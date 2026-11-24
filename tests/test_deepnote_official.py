"""Offline Windows-compatible unit tests for Deepnote's 8 real official benchmark tasks & Notebook 08."""

from __future__ import annotations

from pathlib import Path

import pytest

from dial_skill.benchmarks.deepnote_official import (
    CATEGORY_TO_SKILL,
    DEEPNOTE_STRICT_JUDGE_JSON_SCHEMA,
    DeepnoteRubricJudgment,
    DeliveredSlackMessage,
    OfficialEpisodeResult,
    convert_deepnote_tool_to_openai,
    enforce_deepnote_score_consistency,
    format_initial_context_messages,
    load_deepnote_official_catalog,
    summarize_official_8_results,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEEPNOTE_REPO_DIR = REPO_ROOT / "external" / "agent-initiative-benchmark"
NOTEBOOK_08_PATH = REPO_ROOT / "notebooks" / "08_official_deepnote_8_benchmark.ipynb"


def test_load_deepnote_official_catalog_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_deepnote_official_catalog(tmp_path)


@pytest.mark.skipif(
    not (DEEPNOTE_REPO_DIR / "tasks" / "index.json").is_file(),
    reason="external/agent-initiative-benchmark not cloned locally",
)
def test_load_all_8_real_deepnote_official_tasks() -> None:
    tasks = load_deepnote_official_catalog(DEEPNOTE_REPO_DIR)
    assert len(tasks) == 8
    assert [t.task_id for t in tasks] == [f"task-{i:02d}" for i in range(1, 9)]

    by_id = {t.task_id: t for t in tasks}
    # Task-03 is the genuine cross-service silence trap; all other 7 are action tasks
    assert by_id["task-03"].intent == "silence"
    assert by_id["task-03"].expected_action == "SILENT"
    assert "linear" in by_id["task-03"].delta_services
    assert "search_console" in by_id["task-03"].delta_services

    for tid in ("task-01", "task-02", "task-04", "task-05", "task-06", "task-07", "task-08"):
        assert by_id[tid].intent == "action"
        assert by_id[tid].expected_action == "INTERVENE"
        assert len(by_id[tid].example_messages) >= 1
        assert len(by_id[tid].important_mistakes) >= 1

    # Verify all 8 categories map cleanly into DIAL-Skill active skills
    assert {t.category for t in tasks} == set(CATEGORY_TO_SKILL.keys())
    assert by_id["task-08"].mapped_skill == "guard"
    assert by_id["task-04"].mapped_skill == "check"

    # Subset filter check
    subset = load_deepnote_official_catalog(DEEPNOTE_REPO_DIR, selected_ids=["task-03", "task-08"])
    assert [t.task_id for t in subset] == ["task-03", "task-08"]


def test_format_initial_context_messages_and_tool_schema_conversion() -> None:
    ctx = {
        "snapshot_time": "2026-10-01T01:17:57+00:00",
        "messages": [
            {
                "channel_id": "C123",
                "user_id": "U999",
                "ts": "1790816696.674688",
                "thread_ts": None,
                "text": "Can you pull user emails from PostHog for the writers?",
                "author": {"real_name": "Rachel Kim", "title": "Marketing Lead"},
                "channel": {"name": "marketing", "topic": "Campaigns"},
            },
            {
                "channel_id": "C456",
                "user_id": "U888",
                "ts": "1790816750.000100",
                "thread_ts": "1790816000.000001",
                "text": "Already fixed in PR #42, thanks!",
                "author": {"real_name": "Alex Chen", "title": "SWE"},
                "channel": {"name": "eng-ops"},
            },
        ],
    }
    msgs = format_initial_context_messages(ctx)
    assert len(msgs) == 2
    assert msgs[0].msg_id == "m01"
    assert msgs[0].speaker == "Rachel Kim"
    assert "#marketing (C123)" in msgs[0].text
    assert "user_id=U999" in msgs[0].text
    assert msgs[1].thread_ts == "1790816000.000001"

    openai_tool = convert_deepnote_tool_to_openai(
        {
            "name": "posthog_query_insight",
            "description": "Query a saved PostHog insight.",
            "parameters": {"type": "object", "properties": {"insight_id": {"type": "string"}}},
        }
    )
    assert openai_tool["type"] == "function"
    assert openai_tool["function"]["name"] == "posthog_query_insight"


def test_deepnote_strict_rubric_zero_conditions_and_consistency() -> None:
    # 1. Strict JSON schema has pure integer types (OpenAI strict-mode compatible)
    for k in (
        "intervention_or_silence",
        "correctness_and_evidence",
        "manner_of_intervention",
        "audience_and_location",
        "overall_score",
    ):
        assert DEEPNOTE_STRICT_JUDGE_JSON_SCHEMA["properties"][k]["type"] == "integer"

    # 2. Deterministic silence on silence task -> 10
    s_ok = enforce_deepnote_score_consistency(expected_intent="silence", delivered_count=0, raw_payload={})
    assert s_ok["overall_score"] == 10
    assert s_ok["intervention_or_silence"] == 10
    assert s_ok["correctness_and_evidence"] is None

    # 3. Deterministic silence on action task -> 0
    s_miss = enforce_deepnote_score_consistency(expected_intent="action", delivered_count=0, raw_payload={})
    assert s_miss["overall_score"] == 0
    assert s_miss["serious_mistake_zero"] is True

    # 4. Unwarranted intervention on silence task -> forced overall 0 regardless of polite wording
    s_false = enforce_deepnote_score_consistency(
        expected_intent="silence",
        delivered_count=1,
        raw_payload={
            "intervention_or_silence": 4,
            "correctness_and_evidence": 9,
            "manner_of_intervention": 10,
            "audience_and_location": 10,
            "overall_score": 6,
            "serious_mistake_zero": False,
        },
    )
    assert s_false["intervention_or_silence"] == 0
    assert s_false["overall_score"] == 0
    assert s_false["serious_mistake_zero"] is True

    # 5. Serious mistake flag forces overall 0
    s_pii = enforce_deepnote_score_consistency(
        expected_intent="action",
        delivered_count=1,
        raw_payload={
            "intervention_or_silence": 10,
            "correctness_and_evidence": 8,
            "manner_of_intervention": 9,
            "audience_and_location": 0,
            "overall_score": 5,
            "triggered_important_mistakes": ["Disclosed user emails"],
            "serious_mistake_zero": True,
        },
    )
    assert s_pii["overall_score"] == 0

    # 6. Consistency: cannot get overall=10 if any category < 10
    s_inconsistent = enforce_deepnote_score_consistency(
        expected_intent="action",
        delivered_count=1,
        raw_payload={
            "intervention_or_silence": 10,
            "correctness_and_evidence": 8,
            "manner_of_intervention": 10,
            "audience_and_location": 10,
            "overall_score": 10,
            "serious_mistake_zero": False,
        },
    )
    assert s_inconsistent["overall_score"] <= 9


def test_summarize_official_8_results() -> None:
    eps = [
        OfficialEpisodeResult(
            task_id="task-01",
            title="t1",
            category="Find",
            expected_intent="action",
            method="DIAL_Skill_official",
            model_id="gpt-4.1",
            predicted_action="INTERVENE",
            delivered_messages=[DeliveredSlackMessage(tool="slack_reply_to_thread", text="ok")],
            tool_calls_used=5,
            read_calls_used=4,
            write_calls_used=1,
            model_elapsed_sec=12.0,
            total_cost_usd=0.02,
        ),
        OfficialEpisodeResult(
            task_id="task-03",
            title="t3",
            category="Analyze",
            expected_intent="silence",
            method="DIAL_Skill_official",
            model_id="gpt-4.1",
            predicted_action="SILENT",
            delivered_messages=[],
            tool_calls_used=3,
            read_calls_used=3,
            write_calls_used=0,
            model_elapsed_sec=8.0,
            total_cost_usd=0.01,
        ),
    ]
    judgments = {
        "task-01": DeepnoteRubricJudgment(
            task_id="task-01",
            method="DIAL_Skill_official",
            model_id="gpt-4.1",
            judge_model="gpt-4.1",
            intervention_or_silence=10.0,
            correctness_and_evidence=8.0,
            manner_of_intervention=9.0,
            audience_and_location=10.0,
            overall_score=8.0,
        ),
        "task-03": DeepnoteRubricJudgment(
            task_id="task-03",
            method="DIAL_Skill_official",
            model_id="gpt-4.1",
            judge_model="gpt-4.1",
            intervention_or_silence=10.0,
            overall_score=10.0,
        ),
    }
    summary = summarize_official_8_results(eps, judgments)
    assert summary["n_tasks"] == 2
    assert summary["overall_mean"] == 9.0
    assert summary["silence_accuracy"] == 1.0
    assert summary["action_recall"] == 1.0
    assert summary["avg_read_calls"] == 3.5
    assert summary["avg_write_calls"] == 0.5


def test_notebook_08_validates_cleanly() -> None:
    import ast
    import importlib
    import nbformat

    from scripts.validate_notebooks import _strip_ipython_magics, validate_notebook_file

    assert NOTEBOOK_08_PATH.is_file(), f"Missing {NOTEBOOK_08_PATH}"
    errors = validate_notebook_file(NOTEBOOK_08_PATH)
    assert not errors, f"Notebook 08 validation failed: {errors}"

    # Verify every `from dial_skill... import ...` symbol in Notebook 08 actually exists!
    nb = nbformat.read(NOTEBOOK_08_PATH, as_version=4)
    checked_imports = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(_strip_ipython_magics(cell.source))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("dial_skill"):
                mod = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(mod, alias.name), (
                        f"Notebook 08 imports missing symbol '{alias.name}' from '{node.module}'"
                    )
                    checked_imports += 1
    assert checked_imports >= 10

