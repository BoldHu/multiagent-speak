"""Milestone 4 tests: When2Speak loader/stratification, 11-service PageMantle harness, 180s timer, and PageMantle-Ext builder."""

from __future__ import annotations

import asyncio
import csv
import json
import time
from pathlib import Path

import pytest

from dial_skill.benchmarks.build_pagemantle_ext import (
    build_pagemantle_ext_suite,
    verify_split_isolation,
    verify_task_db_grounding,
)
from dial_skill.benchmarks.pagemantle_env import ALL_11_SERVICES, ModelBudgetTimer, PageMantleEnv
from dial_skill.benchmarks.when2speak import (
    assert_no_calibration_leakage,
    load_when2speak_split,
)
from dial_skill.schemas import ChatMessage, InterventionTier, PageMantleTask


def _create_mock_pagemantle_dataset(root: Path) -> Path:
    """Create disk-backed mock fixtures across all 11 PageMantle enterprise services."""
    ds = root / "pagemantle"
    ds.mkdir(parents=True, exist_ok=True)
    for svc_idx, svc in enumerate(ALL_11_SERVICES):
        svc_dir = ds / svc
        svc_dir.mkdir(parents=True, exist_ok=True)
        if svc in ("bigquery", "snowflake"):
            csv_path = svc_dir / f"{svc}_metrics.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "metric", "value", "quarter"])
                writer.writeheader()
                for i in range(10):
                    writer.writerow(
                        {
                            "id": f"{svc[:3].upper()}-{svc_idx}-{i}",
                            "metric": f"arr_segment_{i}",
                            "value": str(10000 + i * 500),
                            "quarter": "2026-Q1",
                        }
                    )
        else:
            recs = [
                {
                    "id": f"{svc[:3].upper()}-{svc_idx}-{i}",
                    "title": f"{svc} record {i}",
                    "status": "open" if i % 2 == 0 else "resolved",
                    "owner": f"owner_{i % 3}",
                    "details": f"Verified {svc} fact #{i} for enterprise audit.",
                }
                for i in range(10)
            ]
            (svc_dir / f"{svc}_records.jsonl").write_text(
                "\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8"
            )
    return ds


def test_when2speak_loader_stratified_sampling_and_zero_leakage(tmp_path: Path) -> None:
    # 1. Test official duke-trust-lab/When2Speak OpenAI chat schema (dialogue & token configs)
    from dial_skill.benchmarks.when2speak import normalize_when2speak_rows

    hf_dialogue_rows = [
        {
            "messages": [
                {"role": "system", "content": "Decide when to speak or output >."},
                {"role": "user", "content": "Speaker_0: Let's discuss the jazz concert timeline."},
                {"role": "user", "content": "Speaker_1: Sounds good, I think it was in 1920."},
                {"role": "assistant", "content": ">"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "Decide when to speak or output >."},
                {"role": "user", "content": "Speaker_0: Let's discuss the jazz concert timeline."},
                {"role": "user", "content": "Speaker_1: Sounds good, I think it was in 1920."},
                {"role": "user", "content": "Speaker_2: Wait [AGENT], can you verify that year?"},
                {"role": "assistant", "content": "Actually the first commercial jazz recording was in 1917."},
            ]
        },
    ]
    parsed_dlg = normalize_when2speak_rows(hf_dialogue_rows, split="validation")
    assert len(parsed_dlg) == 2
    assert parsed_dlg[0].label_speak is False
    assert parsed_dlg[0].reference_response is None
    assert parsed_dlg[1].label_speak is True
    assert parsed_dlg[1].reference_response == "Actually the first commercial jazz recording was in 1917."
    # System prompt stripped, speaker prefix parsed, and same opening turn groups into same conversation_id!
    assert all(m.speaker != "system" for t in parsed_dlg for m in t.messages)
    assert parsed_dlg[0].messages[0].speaker == "Speaker_0"
    assert parsed_dlg[0].conversation_id == parsed_dlg[1].conversation_id

    hf_token_rows = [
        {
            "messages": [
                {"role": "user", "content": "Speaker_0: Quick question for [AGENT]."},
                {"role": "assistant", "content": "<"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Speaker_0: Thanks everyone, bye!"},
                {"role": "assistant", "content": ">"},
            ]
        },
    ]
    parsed_tok = normalize_when2speak_rows(hf_token_rows, split="test")
    assert parsed_tok[0].label_speak is True   # `<` = SPEAK in When2Speak
    assert parsed_tok[1].label_speak is False  # `>` = SILENT in When2Speak

    # 2. Test local validation + test JSONLs matching ~13% SPEAK & 5 intervention types
    types = ["clarification", "correction", "coordination", "suggestion", "alert"]
    for split, n_convs in [("validation", 25), ("test", 50)]:
        rows = []
        for c in range(n_convs):
            cid = f"{split}_conv_{c}"
            itype = types[c % len(types)]
            for t in range(10):
                speak = (t == 7) or (t == 9 and c % 3 == 0)
                rows.append(
                    {
                        "conversation_id": cid,
                        "turn_index": t,
                        "messages": [{"speaker": f"u{k%2}", "text": f"Turn {k} in {cid}"} for k in range(t + 1)],
                        "token": "<" if speak else ">",
                        "intervention_type": itype,
                    }
                )
        (tmp_path / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )

    val_turns = load_when2speak_split("val_calib", local_dir=tmp_path, seed=42)
    pilot_turns = load_when2speak_split("pilot", local_dir=tmp_path, seed=42)

    assert len(val_turns) == 200
    assert len(pilot_turns) == 400
    assert all(t.split == "validation" for t in val_turns)
    assert all(t.split == "test" for t in pilot_turns)

    # Check positive ratio is preserved close to ~13%
    pos_rate = sum(1 for t in pilot_turns if t.label_speak) / len(pilot_turns)
    assert 0.08 <= pos_rate <= 0.25

    # Invariant #8 zero leakage passes for val vs test, fails if test leaks into calib
    assert_no_calibration_leakage(val_turns, pilot_turns)
    with pytest.raises(AssertionError, match="Invariant #8"):
        assert_no_calibration_leakage(pilot_turns[:100], pilot_turns)


def test_pagemantle_env_all_11_services_and_180s_timer(tmp_path: Path) -> None:
    ds_dir = _create_mock_pagemantle_dataset(tmp_path)
    env = PageMantleEnv(dataset_dir=ds_dir, max_tool_calls=5, hard_time_cap_sec=180.0)

    assert set(env.available_services()) == set(ALL_11_SERVICES)

    read_only_tools = env.get_openai_tool_schemas(read_only=True)
    all_tools = env.get_openai_tool_schemas(read_only=False)
    assert len(all_tools) == len(read_only_tools) + 1
    assert all(t["function"]["name"] != "slack_post_message" for t in read_only_tools)

    # Search, get_record, and SQL query across services
    res_stripe = env.call_tool("stripe_get_record", {"record_id": "STR-4-0"})
    assert res_stripe["found"] is True
    assert res_stripe["record"]["id"] == "STR-4-0"

    res_snow = env.call_tool("snowflake_query_sql", {"sql": "SELECT id, metric FROM snowflake_metrics LIMIT 3"})
    assert res_snow["row_count"] == 3

    # Verify tool-call budget enforcement
    env.call_tool("linear_search", {"query": "linear"})
    env.call_tool("notion_search", {"query": "notion"})
    env.call_tool("hubspot_search", {"query": "hubspot"})
    over_budget = env.call_tool("slack_search", {"query": "slack"})
    assert "error" in over_budget and "budget" in over_budget["error"].lower()

    # Verify ModelBudgetTimer excludes tool time from model_elapsed_sec (Invariant #7)
    timer = ModelBudgetTimer(hard_cap_sec=180.0)
    timer.start_tool()
    time.sleep(0.03)
    timer.stop_tool()
    assert timer.tool_elapsed_sec >= 0.02
    assert timer.model_elapsed_sec == 0.0
    assert timer.timed_out is False


def test_build_pagemantle_ext_64_tasks_db_grounding_and_isolation(tmp_path: Path) -> None:
    ds_dir = _create_mock_pagemantle_dataset(tmp_path)
    out_dir = tmp_path / "pagemantle_ext"
    env = PageMantleEnv(dataset_dir=ds_dir)

    report = asyncio.run(build_pagemantle_ext_suite(env=env, output_dir=out_dir, seed=42))
    assert report["total_tasks"] == 64
    assert report["db_grounding_passed"] is True
    assert report["isolation"]["official_count"] == 8
    assert report["isolation"]["online_stream_count"] == 24
    assert report["isolation"]["eval_count"] == 32
    assert report["isolation"]["wave_counts"] == {1: 8, 2: 8, 3: 8}
    assert report["isolation"]["leakage_free"] is True
    assert report["isolation"]["record_overlap_count"] == 0

    # Verify JSONLs written cleanly
    assert (out_dir / "pm_official_8.jsonl").is_file()
    assert (out_dir / "pm_online_stream_24.jsonl").is_file()
    assert (out_dir / "pm_eval_32.jsonl").is_file()
    assert (out_dir / "verification_audit.json").is_file()

    # Verify deterministic DB-grounding catches hallucinated record IDs!
    hallucinated = PageMantleTask(
        task_id="fake_01",
        split="pm_eval_32",
        category="diagnose",
        messages=[ChatMessage(msg_id="m0", speaker="a", text="x")],
        expected_action="INTERVENE",
        ideal_tier=InterventionTier.THREAD_BRIEF,
        required_evidence_sources=[{"service": "stripe", "record_id": "FAKE-INV-99999"}],
    )
    ok, missing = verify_task_db_grounding(hallucinated, env)
    assert ok is False
    assert "stripe:FAKE-INV-99999" in missing

    # Verify split-leakage guard catches any eval task placed inside online_stream!
    eval_tasks = [
        PageMantleTask.model_validate_json(line)
        for line in (out_dir / "pm_eval_32.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    official_tasks = [
        PageMantleTask.model_validate_json(line)
        for line in (out_dir / "pm_official_8.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stream_tasks = [
        PageMantleTask.model_validate_json(line)
        for line in (out_dir / "pm_online_stream_24.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with pytest.raises(AssertionError, match="Leakage"):
        verify_split_isolation(official_tasks, [*stream_tasks[:23], eval_tasks[0]], eval_tasks, strict_counts=False)
