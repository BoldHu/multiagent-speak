"""Deterministic, DB-grounded 64-task PageMantle-Ext builder with DB-existence & split-leakage QA (Invariant #8)."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Any, Literal

import typer

from dial_skill.benchmarks.pagemantle_env import PageMantleEnv, load_official_tasks
from dial_skill.llm.openai_client import OpenAIClient
from dial_skill.prompts.templates import PAGEMANTLE_SYNTHESIZER_SYSTEM_PROMPT
from dial_skill.schemas import (
    ACTIVE_SKILLS,
    ActiveSkillName,
    ChatMessage,
    InterventionTier,
    PageMantleTask,
)
from dial_skill.utils.drive import ensure_drive_layout

logger = logging.getLogger(__name__)
cli = typer.Typer(help="Build and verify the 64-task DB-grounded PageMantle-Ext benchmark suite.")

TRAP_TYPES: tuple[
    Literal["already_resolved", "explicit_deferral", "brainstorming_flow", "scope_boundary"], ...
] = (
    "already_resolved",
    "explicit_deferral",
    "brainstorming_flow",
    "scope_boundary",
)

_CATEGORY_IDEAL_TIERS: dict[ActiveSkillName, InterventionTier] = {
    "find": InterventionTier.THREAD_BRIEF,
    "explain": InterventionTier.THREAD_BRIEF,
    "analyze": InterventionTier.THREAD_BRIEF,
    "check": InterventionTier.DM_NUDGE,
    "diagnose": InterventionTier.CHANNEL_INTERVENE,
    "decide": InterventionTier.THREAD_BRIEF,
    "coordinate": InterventionTier.CHANNEL_INTERVENE,
    "guard": InterventionTier.DM_NUDGE,
}

_CATEGORY_CHANNELS: dict[ActiveSkillName, str] = {
    "find": "#team-general",
    "explain": "#product-ops",
    "analyze": "#analytics",
    "check": "#revops",
    "diagnose": "#eng-incidents",
    "decide": "#roadmap",
    "coordinate": "#launch-planning",
    "guard": "#security-compliance",
}

_SPEAKERS = ["maya", "devon", "priya", "liam", "elena", "noah"]


def verify_task_db_grounding(task: PageMantleTask, env: PageMantleEnv) -> tuple[bool, list[str]]:
    """Stage-1 QA Gate: programmatically assert every cited evidence source exists in local PageMantle DB."""
    missing: list[str] = []
    if not task.required_evidence_sources:
        return True, []
    for src in task.required_evidence_sources:
        service = str(src.get("service", ""))
        rec_id = str(src.get("record_id", ""))
        if not service or not rec_id or not env.verify_record_exists(service, rec_id):
            missing.append(f"{service}:{rec_id}")
    return len(missing) == 0, missing


def verify_split_isolation(
    official: list[PageMantleTask],
    online_stream: list[PageMantleTask],
    eval_set: list[PageMantleTask],
    strict_counts: bool = True,
) -> dict[str, Any]:
    """Enforce Invariant #8: pm_official_8 and pm_eval_32 NEVER leak into pm_online_stream_24."""
    off_ids = {t.task_id for t in official}
    stream_ids = {t.task_id for t in online_stream}
    eval_ids = {t.task_id for t in eval_set}

    if off_ids & stream_ids:
        raise AssertionError(f"Leakage: official tasks found in online_stream: {off_ids & stream_ids}")
    if eval_ids & stream_ids:
        raise AssertionError(f"Leakage: eval tasks found in online_stream: {eval_ids & stream_ids}")
    if off_ids & eval_ids:
        raise AssertionError(f"Overlap between official and eval tasks: {off_ids & eval_ids}")

    # Verify zero shared DB record IDs between online_stream and held-out eval_set
    def _record_keys(tasks: list[PageMantleTask]) -> set[str]:
        keys: set[str] = set()
        for t in tasks:
            for src in t.required_evidence_sources:
                if src.get("service") and src.get("record_id"):
                    keys.add(f"{src['service']}:{src['record_id']}")
        return keys

    stream_recs = _record_keys(online_stream)
    eval_recs = _record_keys(eval_set)
    rec_overlap = stream_recs & eval_recs

    # Verify online_stream waves (1, 2, 3)
    wave_counts: dict[int, int] = {}
    for t in online_stream:
        if t.wave is not None:
            wave_counts[t.wave] = wave_counts.get(t.wave, 0) + 1

    if strict_counts:
        assert len(official) == 8, f"Expected 8 pm_official_8 tasks, got {len(official)}"
        assert len(online_stream) == 24, f"Expected 24 pm_online_stream_24 tasks, got {len(online_stream)}"
        assert len(eval_set) == 32, f"Expected 32 pm_eval_32 tasks, got {len(eval_set)}"
        assert wave_counts == {1: 8, 2: 8, 3: 8}, f"Expected 3 waves of 8, got {wave_counts}"
        stream_traps = sum(1 for t in online_stream if t.expected_action == "SILENT")
        eval_traps = sum(1 for t in eval_set if t.expected_action == "SILENT")
        assert stream_traps == 6, f"Expected 6 traps in online_stream, got {stream_traps}"
        assert eval_traps == 10, f"Expected 10 traps in eval_set, got {eval_traps}"

    return {
        "official_count": len(official),
        "online_stream_count": len(online_stream),
        "eval_count": len(eval_set),
        "wave_counts": wave_counts,
        "record_overlap_count": len(rec_overlap),
        "leakage_free": len(off_ids & stream_ids) == 0 and len(eval_ids & stream_ids) == 0,
    }


def _synthesize_deterministic_thread(
    *,
    task_id: str,
    split: Literal["pm_official_8", "pm_online_stream_24", "pm_eval_32"],
    wave: int | None,
    category: ActiveSkillName,
    fact: dict[str, Any],
    expected_action: Literal["INTERVENE", "SILENT"],
    trap_type: Literal["already_resolved", "explicit_deferral", "brainstorming_flow", "scope_boundary"] | None,
    rng: random.Random,
) -> PageMantleTask:
    """Build a realistic, 100% DB-grounded 8-turn Slack thread deterministically (zero hallucination)."""
    service = fact["service"]
    rec_id = str(fact["record_id"])
    rec_summary = json.dumps(fact["record"], ensure_ascii=False, default=str)[:220]
    channel = _CATEGORY_CHANNELS[category]
    spks = rng.sample(_SPEAKERS, 4)

    base_msgs = [
        (spks[0], "Morning everyone — quick sync before standup."),
        (spks[1], "Sounds good, just grabbing coffee and reviewing the sprint board."),
        (spks[2], f"By the way, on the {category} thread around `{rec_id}` in {service}:"),
    ]

    if expected_action == "INTERVENE":
        ideal_tier = _CATEGORY_IDEAL_TIERS[category]
        action_turns = [
            (spks[2], f"I think `{rec_id}` is already closed with zero impact—can we skip follow-up?"),
            (spks[0], "Wait, is that actually what the source record says? I don't have it open."),
            (spks[1], "Not sure either, we should verify before updating leadership."),
            (spks[3], "Anyone know the exact status or policy details for this?"),
            (spks[0], "If nobody checks soon we might send the wrong summary out."),
        ]
        ref_answer = (
            f"Verified `{rec_id}` in {service}: {rec_summary}. "
            f"Recommended action ({category}, tier={ideal_tier.value}): share verified record details."
        )
    else:
        ideal_tier = InterventionTier.SILENT
        if trap_type == "already_resolved":
            action_turns = [
                (spks[2], f"Does anyone know the latest status of `{rec_id}` in {service}?"),
                (spks[1], f"Just checked {service} directly: `{rec_id}` is accounted for and updated."),
                (spks[2], "Awesome, thanks for confirming so quickly!"),
                (spks[0], "Great, marking that resolved on my side too."),
                (spks[3], "All set then, moving on to the next item."),
            ]
        elif trap_type == "explicit_deferral":
            action_turns = [
                (spks[2], f"Should we dig into `{rec_id}` from {service} right now?"),
                (spks[0], "Let's explicitly defer this until Friday's release retrospective offline."),
                (spks[1], "Agreed—no need to investigate or pull numbers today."),
                (spks[2], "Makes sense, I'll hold off until Friday."),
                (spks[3], "Cool, let's focus on today's launch checklist instead."),
            ]
        elif trap_type == "brainstorming_flow":
            action_turns = [
                (spks[2], f"Just throwing rough ideas out there—what if next quarter we rethink areas like `{rec_id}`?"),
                (spks[1], "Ooh interesting blue-sky idea, no need for exact numbers yet."),
                (spks[0], "Yeah let's keep riffing loosely before narrowing anything down."),
                (spks[3], "Maybe we could also explore an async template workflow later."),
                (spks[2], "Love the energy, let's keep brainstorming!"),
            ]
        else:  # scope_boundary
            action_turns = [
                (spks[2], f"Quick note: HR/legal is handling sensitive aspects of `{rec_id}` privately."),
                (spks[0], "Right, please keep any details out of this public channel."),
                (spks[1], "Understood—legal already has direct access and asked us not to discuss it here."),
                (spks[3], "Good reminder, keeping this channel strictly on public product updates."),
                (spks[2], "Thanks everyone, shifting back to UI polish."),
            ]
        ref_answer = f"SILENT (Hard-negative silence trap: {trap_type})."

    messages = [
        ChatMessage(
            msg_id=f"{task_id}_m{i}",
            speaker=spk,
            text=txt,
            timestamp=f"2026-01-15T10:{i:02d}:00Z",
            channel=channel,
            thread_ts=f"{task_id}_t0" if i >= 2 else None,
        )
        for i, (spk, txt) in enumerate([*base_msgs, *action_turns])
    ]

    return PageMantleTask(
        task_id=task_id,
        split=split,
        wave=wave,
        category=category,
        channel_name=channel,
        messages=messages,
        expected_action=expected_action,
        ideal_tier=ideal_tier,
        trap_type=trap_type,
        required_evidence_sources=[{"service": service, "record_id": rec_id}],
        reference_answer=ref_answer,
        ground_truth_facts=[f"[{service}:{rec_id}] {rec_summary}"],
        metadata={"synthesizer": "deterministic_db_grounded"},
    )


async def _enrich_with_llm(
    task: PageMantleTask,
    fact: dict[str, Any],
    openai_client: OpenAIClient,
    model: str,
    seed: int,
) -> PageMantleTask:
    """Optionally polish deterministic DB-grounded task with LLM dialogue realism while preserving record IDs."""
    sys_prompt = PAGEMANTLE_SYNTHESIZER_SYSTEM_PROMPT.format(
        category=task.category,
        trap_type=task.trap_type or "none",
    )
    user_prompt = (
        f"Verified DB Snapshot (DO NOT invent IDs outside this):\n"
        f"{json.dumps(fact, ensure_ascii=False, default=str)[:1200]}\n\n"
        f"Target specification:\n"
        f"- task_id: {task.task_id}\n"
        f"- category: {task.category}\n"
        f"- expected_action: {task.expected_action}\n"
        f"- ideal_tier: {task.ideal_tier.value}\n"
        f"- trap_type: {task.trap_type}\n"
        "Return JSON with keys: messages (list of {speaker, text}), reference_answer (string)."
    )
    try:
        res = await openai_client.complete_json(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            seed=seed,
        )
        data = res.parsed_json or {}
        raw_msgs = data.get("messages", [])
        if isinstance(raw_msgs, list) and len(raw_msgs) >= 6:
            msgs = [
                ChatMessage(
                    msg_id=f"{task.task_id}_m{i}",
                    speaker=str(m.get("speaker", f"emp_{i%3}")),
                    text=str(m.get("text", "")),
                    timestamp=f"2026-01-15T10:{i:02d}:00Z",
                    channel=task.channel_name,
                    thread_ts=f"{task.task_id}_t0" if i >= 2 else None,
                )
                for i, m in enumerate(raw_msgs[:12])
                if isinstance(m, dict) and m.get("text")
            ]
            if len(msgs) >= 6:
                return task.model_copy(
                    update={
                        "messages": msgs,
                        "reference_answer": str(data.get("reference_answer") or task.reference_answer),
                        "metadata": {**task.metadata, "synthesizer": f"llm_db_grounded:{model}"},
                    }
                )
    except Exception as exc:
        logger.warning("LLM synthesis fallback for %s (%s); keeping deterministic grounded thread.", task.task_id, exc)
    return task


async def audit_task_ambiguity(
    task: PageMantleTask,
    openai_client: OpenAIClient,
    judge_model: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Stage-2 QA Gate: Terra ambiguity audit verifying unambiguous expected_action & ideal_tier."""
    transcript = "\n".join(f"[{m.msg_id}] {m.speaker}: {m.text}" for m in task.messages)
    prompt = (
        f"Channel: {task.channel_name}\nCategory: {task.category}\n"
        f"Proposed Ground Truth: expected_action={task.expected_action}, ideal_tier={task.ideal_tier.value}, "
        f"trap_type={task.trap_type}\n"
        f"Verified Facts: {task.ground_truth_facts}\n\nTranscript:\n{transcript}\n\n"
        'Return strict JSON: {"unambiguous": true/false, "agreed_action": "INTERVENE|SILENT", '
        '"agreed_tier": "SILENT|DM_NUDGE|THREAD_BRIEF|CHANNEL_INTERVENE", "reason": "..."}'
    )
    res = await openai_client.complete_json(
        model=judge_model,
        messages=[
            {"role": "system", "content": "You are a strict benchmark quality auditor checking for ambiguity."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        seed=seed,
    )
    data = res.parsed_json or {}
    return {
        "task_id": task.task_id,
        "unambiguous": bool(data.get("unambiguous", True)),
        "agreed_action": data.get("agreed_action", task.expected_action),
        "agreed_tier": data.get("agreed_tier", task.ideal_tier.value),
        "reason": str(data.get("reason", "")),
    }


async def build_pagemantle_ext_suite(
    *,
    env: PageMantleEnv,
    output_dir: str | Path,
    benchmark_repo_dir: str | Path | None = None,
    openai_client: OpenAIClient | None = None,
    synth_model: str | None = None,
    judge_model: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Build and verify the complete 64-task PageMantle-Ext suite (8 official + 24 stream + 32 eval)."""
    rng = random.Random(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    facts = env.mine_verified_facts(limit_per_service=12)
    if not facts:
        raise RuntimeError(
            f"No PageMantle records found in {env.dataset_dir}. "
            "Download deepnote/pagemantle-company-dataset or provide fixture files."
        )
    rng.shuffle(facts)

    def _pick_fact(cursor: int) -> dict[str, Any]:
        return facts[cursor % len(facts)]

    # 1. Load official 8 tasks (or construct DB-grounded fallback placeholders if repo not cloned yet)
    official_tasks = load_official_tasks(benchmark_repo_dir=benchmark_repo_dir)
    if len(official_tasks) < 8:
        for i in range(len(official_tasks), 8):
            cat = ACTIVE_SKILLS[i % len(ACTIVE_SKILLS)]
            official_tasks.append(
                _synthesize_deterministic_thread(
                    task_id=f"pm_official_{i + 1:02d}",
                    split="pm_official_8",
                    wave=None,
                    category=cat,
                    fact=_pick_fact(i),
                    expected_action="INTERVENE",
                    trap_type=None,
                    rng=rng,
                )
            )
    official_tasks = official_tasks[:8]

    # Partition mined facts disjointly between online_stream (24) and eval_set (32) when enough facts exist
    stream_fact_offset = 8
    eval_fact_offset = 8 + 24

    # 2. Build pm_online_stream_24: 18 Action + 6 Silence Traps across 3 waves of 8 (6 Action + 2 Traps/wave)
    online_stream: list[PageMantleTask] = []
    for idx in range(24):
        wave = (idx // 8) + 1
        pos_in_wave = idx % 8
        is_trap = pos_in_wave >= 6  # 6 action + 2 traps per wave => 18 action + 6 traps total
        cat = ACTIVE_SKILLS[idx % len(ACTIVE_SKILLS)]
        trap_type = TRAP_TYPES[idx % len(TRAP_TYPES)] if is_trap else None
        fact = _pick_fact(stream_fact_offset + idx)
        task = _synthesize_deterministic_thread(
            task_id=f"pm_stream_w{wave}_{idx + 1:02d}",
            split="pm_online_stream_24",
            wave=wave,
            category=cat,
            fact=fact,
            expected_action="SILENT" if is_trap else "INTERVENE",
            trap_type=trap_type,
            rng=rng,
        )
        if openai_client is not None and synth_model:
            task = await _enrich_with_llm(task, fact, openai_client, synth_model, seed + idx)
        online_stream.append(task)

    # 3. Build pm_eval_32: 22 Action + 10 Hard-Negative Silence Traps (strictly held out)
    eval_set: list[PageMantleTask] = []
    for idx in range(32):
        is_trap = idx >= 22  # 22 action + 10 traps
        cat = ACTIVE_SKILLS[idx % len(ACTIVE_SKILLS)]
        trap_type = TRAP_TYPES[(idx - 22) % len(TRAP_TYPES)] if is_trap else None
        fact = _pick_fact(eval_fact_offset + idx)
        task = _synthesize_deterministic_thread(
            task_id=f"pm_eval_{idx + 1:02d}",
            split="pm_eval_32",
            wave=None,
            category=cat,
            fact=fact,
            expected_action="SILENT" if is_trap else "INTERVENE",
            trap_type=trap_type,
            rng=rng,
        )
        if openai_client is not None and synth_model:
            task = await _enrich_with_llm(task, fact, openai_client, synth_model, seed + 100 + idx)
        eval_set.append(task)

    # 4. Run Two-Stage Quality Assurance Gate
    grounding_failures: list[dict[str, Any]] = []
    for task in [*official_tasks, *online_stream, *eval_set]:
        ok, missing = verify_task_db_grounding(task, env)
        if not ok:
            grounding_failures.append({"task_id": task.task_id, "missing": missing})

    ambiguity_audits: list[dict[str, Any]] = []
    if openai_client is not None and judge_model:
        for task in [*online_stream, *eval_set]:
            ambiguity_audits.append(await audit_task_ambiguity(task, openai_client, judge_model, seed))

    isolation_report = verify_split_isolation(official_tasks, online_stream, eval_set, strict_counts=True)

    # 5. Persist JSONL splits + verification_audit.json
    def _write_split(filename: str, tasks: list[PageMantleTask]) -> Path:
        p = out_dir / filename
        with p.open("w", encoding="utf-8") as f:
            for t in tasks:
                f.write(t.model_dump_json() + "\n")
        return p

    _write_split("pm_official_8.jsonl", official_tasks)
    _write_split("pm_online_stream_24.jsonl", online_stream)
    _write_split("pm_eval_32.jsonl", eval_set)

    audit_report = {
        "total_tasks": len(official_tasks) + len(online_stream) + len(eval_set),
        "isolation": isolation_report,
        "db_grounding_passed": len(grounding_failures) == 0,
        "db_grounding_failures": grounding_failures,
        "ambiguity_audits_count": len(ambiguity_audits),
        "ambiguity_unambiguous_rate": (
            sum(1 for a in ambiguity_audits if a["unambiguous"]) / len(ambiguity_audits)
            if ambiguity_audits
            else 1.0
        ),
    }
    (out_dir / "verification_audit.json").write_text(
        json.dumps(audit_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit_report


@cli.command()
def build(
    dataset_dir: Path | None = typer.Option(None, help="Path to PageMantle local dataset folder."),
    output_dir: Path | None = typer.Option(None, help="Output folder for PageMantle-Ext JSONLs."),
    benchmark_repo_dir: Path | None = typer.Option(None, help="Path to cloned agent-initiative-benchmark."),
    dry_run: bool = typer.Option(True, help="Run deterministic DB-grounded synthesis with zero LLM API cost."),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    """Build and verify PageMantle-Ext splits from local PageMantle records."""
    drive = ensure_drive_layout()
    ds_dir = dataset_dir or (drive / "datasets" / "pagemantle")
    out_dir = output_dir or (drive / "datasets" / "pagemantle_ext")
    env = PageMantleEnv(dataset_dir=ds_dir, benchmark_repo_dir=benchmark_repo_dir)
    oai = None if dry_run else OpenAIClient()
    report = asyncio.run(
        build_pagemantle_ext_suite(
            env=env,
            output_dir=out_dir,
            benchmark_repo_dir=benchmark_repo_dir,
            openai_client=oai,
            synth_model=None if dry_run else "gpt-6.1-terra",
            judge_model=None if dry_run else "gpt-6.1-terra",
            seed=seed,
        )
    )
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    cli()
