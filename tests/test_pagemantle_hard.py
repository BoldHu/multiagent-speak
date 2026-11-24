"""Offline unit tests for Per-Thread Multi-Channel DIAL-Skill v2, PageMantle-Hard, and Notebook 09."""

from __future__ import annotations

import ast
import asyncio
import importlib
from pathlib import Path

import nbformat

from dial_skill.benchmarks.deepnote_official import (
    DeepnoteRubricJudgment,
    DeliveredSlackMessage,
    OfficialEpisodeResult,
    partition_snapshot_into_threads,
)
from dial_skill.benchmarks.pagemantle_hard import (
    PageMantleHardSession,
    build_pagemantle_hard_suite,
    extract_feedback_from_hard_episode,
    load_pagemantle_hard_jsonl,
    verify_hard_split_isolation,
)
from scripts.validate_notebooks import _strip_ipython_magics, validate_notebook_file

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_09_PATH = REPO_ROOT / "notebooks" / "09_pagemantle_hard_benchmark.ipynb"


def test_partition_snapshot_into_isolated_threads() -> None:
    snapshot = {
        "snapshot_time": "2026-10-01T01:15:00+00:00",
        "messages": [
            {"channel_id": "C1", "ts": "100.1", "thread_ts": "90.0", "text": "Decoy 1", "channel": {"name": "mktg"}},
            {"channel_id": "C2", "ts": "101.0", "thread_ts": None, "text": "Real question", "channel": {"name": "prod"}},
            {"channel_id": "C1", "ts": "102.2", "thread_ts": "90.0", "text": "Thanks, settled!", "channel": {"name": "mktg"}},
            {"channel_id": "C2", "ts": "103.5", "thread_ts": "101.0", "text": "Follow-up on real", "channel": {"name": "prod"}},
            {"channel_id": "C3", "ts": "104.0", "thread_ts": "80.0", "text": "Decoy 2 done", "channel": {"name": "fin"}},
        ],
    }
    threads = partition_snapshot_into_threads(snapshot)
    assert len(threads) == 3
    by_ch = {t.channel_name: t for t in threads}
    assert len(by_ch["mktg"].raw_messages) == 2
    assert len(by_ch["prod"].raw_messages) == 2
    assert by_ch["prod"].root_ts == "101.0"
    assert len(by_ch["fin"].raw_messages) == 1


def test_build_pagemantle_hard_56_tasks_isolation_and_roundtrip(tmp_path: Path) -> None:
    built = build_pagemantle_hard_suite(env=None, seed=2026, output_dir=tmp_path)
    eval_tasks = built["pm_hard_eval_32"]
    stream_tasks = built["pm_hard_stream_24"]

    qa = verify_hard_split_isolation(eval_tasks, stream_tasks, strict_counts=True)
    assert qa["leakage_free"] is True
    assert qa["record_overlap_count"] == 0
    assert qa["eval_silence_traps"] == 10
    assert qa["eval_cross_service_silence_traps"] == 6
    assert qa["eval_mixed_intent_guards"] == 3

    # Every task must have >=3 isolated threads (1 target + 2 decoys), explicit mistakes, and all delta services enabled
    for t in (*eval_tasks, *stream_tasks):
        threads = partition_snapshot_into_threads(t.initial_context)
        assert len(threads) >= 3, f"{t.task_id} missing decoy threads"
        assert len(t.important_mistakes) >= 1, f"{t.task_id} missing important_mistakes"
        for svc, rows in t.service_deltas.items():
            if rows:
                assert svc in t.enabled_services, f"{t.task_id} has delta in '{svc}' but '{svc}' not in enabled_services"
        off_t = t.to_official_task()
        assert off_t.task_id == t.task_id
        assert off_t.intent == t.intent

    # JSONL reload roundtrip
    reloaded = load_pagemantle_hard_jsonl(tmp_path / "pm_hard_eval_32.jsonl")
    assert len(reloaded) == 32
    assert reloaded[0].task_id == eval_tasks[0].task_id


def test_pagemantle_hard_session_reads_hidden_deltas_and_tracks_deliveries() -> None:
    built = build_pagemantle_hard_suite(env=None, seed=2026)
    sil_task = next(t for t in built["pm_hard_eval_32"] if t.trap_type == "cross_service_resolved")
    guard_task = next(t for t in built["pm_hard_eval_32"] if t.mixed_intent_guard)

    async def _exercise():
        # 1. Cross-service silence trap: hidden Linear resolution MUST be discoverable via linear_search
        async with PageMantleHardSession(hard_task=sil_task, base_env=None) as sess:
            read_schemas = sess.get_openai_tool_schemas(read_only=True)
            names = {s["function"]["name"] for s in read_schemas}
            assert "slack_reply_to_thread" not in names
            assert "linear_search" in names

            res, _ = await sess.call_tool("linear_search", {"query": "OPS-"}, read_only=True)
            assert res["ok"] is True
            assert len(res["matches"]) >= 1
            assert res["matches"][0]["status"] == "Done"

            # Write blocked during read-only recon
            blocked, _ = await sess.call_tool(
                "slack_reply_to_thread",
                {"channel_id": sil_task.target_channel_id, "thread_ts": sil_task.target_root_ts, "text": "hi"},
                read_only=True,
            )
            assert blocked["ok"] is False
            assert len(sess.delivered_messages) == 0

        # 2. Mixed-intent guard task: PostHog aggregate + Notion policy discoverable, Slack write recorded
        async with PageMantleHardSession(hard_task=guard_task, base_env=None) as sess:
            ph, _ = await sess.call_tool("posthog_search", {"query": "PH-AGG"}, read_only=True)
            assert ph["ok"] is True and len(ph["matches"]) >= 1
            nt, _ = await sess.call_tool("notion_search", {"query": "Customer Data Handling"}, read_only=True)
            assert nt["ok"] is True and len(nt["matches"]) >= 1

            wr, _ = await sess.call_tool(
                "slack_reply_to_thread",
                {
                    "channel_id": guard_task.target_channel_id,
                    "thread_ts": guard_task.target_root_ts,
                    "text": "Aggregate summary provided; declining PII per policy.",
                },
                read_only=False,
            )
            assert wr["ok"] is True
            assert len(sess.delivered_messages) == 1
            assert sess.write_calls_used == 1

    asyncio.run(_exercise())


def test_extract_feedback_from_hard_episode_maps_all_failure_types() -> None:
    built = build_pagemantle_hard_suite(env=None, seed=2026)
    sil_task = next(t for t in built["pm_hard_eval_32"] if t.expected_action == "SILENT")
    act_task = next(t for t in built["pm_hard_eval_32"] if t.expected_action == "INTERVENE")

    # FP intrusion on silence trap
    ep_fp = OfficialEpisodeResult(
        task_id=sil_task.task_id,
        title=sil_task.title,
        category=sil_task.category,
        expected_intent="silence",
        method="DIAL_Skill_v2",
        model_id="gpt-4.1",
        predicted_action="INTERVENE",
        delivered_messages=[DeliveredSlackMessage(tool="slack_reply_to_thread", text="oops")],
    )
    j_fp = DeepnoteRubricJudgment(
        task_id=sil_task.task_id,
        method="DIAL_Skill_v2",
        model_id="gpt-4.1",
        judge_model="gpt-4.1",
        intervention_or_silence=0.0,
        overall_score=0.0,
        serious_mistake_zero=True,
    )
    fb_fp = extract_feedback_from_hard_episode(ep_fp, sil_task, j_fp)
    assert fb_fp.feedback_type == "FP_INTRUSION"

    # FN omission on action task
    ep_fn = OfficialEpisodeResult(
        task_id=act_task.task_id,
        title=act_task.title,
        category=act_task.category,
        expected_intent="action",
        method="DIAL_Skill_v2",
        model_id="gpt-4.1",
        predicted_action="SILENT",
    )
    j_fn = DeepnoteRubricJudgment(
        task_id=act_task.task_id,
        method="DIAL_Skill_v2",
        model_id="gpt-4.1",
        judge_model="gpt-4.1",
        intervention_or_silence=0.0,
        overall_score=0.0,
        serious_mistake_zero=True,
    )
    fb_fn = extract_feedback_from_hard_episode(ep_fn, act_task, j_fn)
    assert fb_fn.feedback_type == "FN_OMISSION"


def test_notebook_09_validates_and_all_dial_skill_imports_exist() -> None:
    assert NOTEBOOK_09_PATH.is_file(), f"Missing {NOTEBOOK_09_PATH}"
    errors = validate_notebook_file(NOTEBOOK_09_PATH)
    assert not errors, f"Notebook 09 validation failed: {errors}"

    nb = nbformat.read(NOTEBOOK_09_PATH, as_version=4)
    checked = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(_strip_ipython_magics(cell.source))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("dial_skill"):
                mod = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(mod, alias.name), (
                        f"Notebook 09 imports missing symbol '{alias.name}' from '{node.module}'"
                    )
                    checked += 1
    assert checked >= 12
