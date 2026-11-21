"""Unified Typer CLI & Python experiment runners with atomic checkpointing, --resume, and cost estimation."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import typer
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from dial_skill.baselines.direct_chat import DirectChatBaseline
from dial_skill.baselines.proact import ProACTBaseline
from dial_skill.baselines.single_gate import SingleGateBaseline
from dial_skill.benchmarks.pagemantle_env import PageMantleEnv
from dial_skill.benchmarks.when2speak import assert_no_calibration_leakage, load_when2speak_split
from dial_skill.config import (
    DialSkillConfig,
    load_dial_skill_config,
    load_experiments_config,
    load_models_config,
)
from dial_skill.core.pipeline import DialSkillAgent
from dial_skill.core.skill_store import SkillStore
from dial_skill.eval.judge import BlindedDualJudge
from dial_skill.eval.metrics import (
    compute_pagemantle_metrics,
    compute_when2speak_metrics,
    evaluate_pilot_go_nogo,
)
from dial_skill.evolution.db_cse import DBCSEUpdater
from dial_skill.evolution.feedback import extract_feedback_from_pagemantle
from dial_skill.llm.jev_client import TriageClient
from dial_skill.llm.openai_client import OpenAIClient
from dial_skill.llm.preflight import load_resolved_models, resolve_model_for_role
from dial_skill.schemas import (
    EpisodeResult,
    InterventionTier,
    PageMantleTask,
    SharedEvidenceSlate,
    When2SpeakTurn,
)
from dial_skill.utils.cache import SQLiteAPICache
from dial_skill.utils.checkpoint import (
    JsonlCheckpointWriter,
    load_completed_ids,
    load_jsonl,
    save_run_metadata,
)
from dial_skill.utils.drive import ensure_drive_layout

logger = logging.getLogger(__name__)
app = typer.Typer(help="DIAL-Skill experiment runner (defaults to pilot mode with Drive resume).")


def estimate_run_cost_usd(
    *,
    mode: Literal["pilot", "main", "full"] = "pilot",
    include_pagemantle: bool = True,
    include_ablations: bool = False,
) -> dict[str, Any]:
    """Estimate API call count and USD spend before launching an experiment sweep."""
    w2s_turns = {"pilot": 400, "main": 2000, "full": 21687}[mode]
    # ~3 calls per turn (Jev + Advocate + Skeptic), reduced ~35% by short-circuiting & caching
    est_w2s_calls = int(w2s_turns * 2.2)
    est_w2s_usd = round(w2s_turns * 0.0022, 2)
    est_pm_calls = (16 * 12) if mode == "pilot" else (64 * 14)
    est_pm_usd = round(est_pm_calls * 0.012, 2) if include_pagemantle else 0.0
    est_ablation_usd = round(est_w2s_usd * 2.5 + est_pm_usd * 1.5, 2) if include_ablations else 0.0
    return {
        "mode": mode,
        "estimated_api_calls": est_w2s_calls + (est_pm_calls if include_pagemantle else 0),
        "estimated_when2speak_usd": est_w2s_usd,
        "estimated_pagemantle_usd": est_pm_usd,
        "estimated_ablations_usd": est_ablation_usd,
        "estimated_total_usd": round(est_w2s_usd + est_pm_usd + est_ablation_usd, 2),
    }


def calibrate_when2speak_thresholds(
    *,
    val_turns: Sequence[When2SpeakTurn],
    val_episodes: Sequence[EpisodeResult],
    eval_turns: Sequence[When2SpeakTurn] | None = None,
    lambda_grid: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    tau_dm_grid: Sequence[float] = (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70),
    theta_floor: float = 0.30,
    max_fir: float | None = None,
    max_mir: float | None = None,
) -> dict[str, float]:
    """Grid-search (lambda, tau_dm) strictly on When2Speak validation split <= 200 turns (Invariant #8)."""
    assert_no_calibration_leakage(list(val_turns), list(eval_turns or []), max_calib_turns=200)
    if len(val_turns) != len(val_episodes):
        raise ValueError("val_turns and val_episodes must align 1-to-1.")

    y_true = [int(t.label_speak) for t in val_turns]
    n_neg = max(1, sum(1 - y for y in y_true))
    n_pos = max(1, sum(y_true))

    best_unconstrained = (-1.0, 1.0, 0.05, 0.0, 0.0)
    best_fir_only = (-1.0, 1.0, 0.05, 0.0, 0.0)
    best_both = (-1.0, 1.0, 0.05, 0.0, 0.0)

    for lam in lambda_grid:
        for tau in tau_dm_grid:
            preds = []
            for ep in val_episodes:
                if ep.gate.short_circuited or ep.gate.advocate_score < theta_floor:
                    preds.append(0)
                else:
                    delta = ep.gate.advocate_score - lam * ep.gate.skeptic_score
                    preds.append(1 if delta >= tau else 0)
            score = float(f1_score(y_true, preds, average="macro", zero_division=0))
            fp = sum(1 for y, p in zip(y_true, preds) if y == 0 and p == 1)
            fn = sum(1 for y, p in zip(y_true, preds) if y == 1 and p == 0)
            fir = fp / n_neg
            mir = fn / n_pos
            cand = (score, float(lam), float(tau), fir, mir)
            if score > best_unconstrained[0]:
                best_unconstrained = cand
            if max_fir is not None and fir <= max_fir:
                if score > best_fir_only[0]:
                    best_fir_only = cand
                if max_mir is not None and mir <= max_mir and score > best_both[0]:
                    best_both = cand

    if max_fir is not None and max_mir is not None and best_both[0] >= 0.0:
        chosen = best_both
    elif max_fir is not None and best_fir_only[0] >= 0.0:
        chosen = best_fir_only
    else:
        chosen = best_unconstrained
    return {
        "best_lambda": chosen[1],
        "best_tau_dm": chosen[2],
        "val_macro_f1": round(chosen[0], 6),
        "val_fir": round(chosen[3], 6),
        "val_mir": round(chosen[4], 6),
        "n_val_turns": len(val_turns),
    }


def sweep_lambda_pareto_from_episodes(
    *,
    turns: Sequence[When2SpeakTurn],
    episodes: Sequence[EpisodeResult],
    lambdas: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5),
    tau_dm: float = 0.05,
    theta_floor: float = 0.30,
) -> list[dict[str, Any]]:
    """Zero-cost FIR-MIR Pareto sweep over lambda reusing cached Stage-2 Advocate/Skeptic scores."""
    y_true = [t.label_speak for t in turns]
    rows: list[dict[str, Any]] = []
    for lam in lambdas:
        rescored: list[EpisodeResult] = []
        for ep in episodes:
            if ep.gate.short_circuited or ep.gate.advocate_score < theta_floor:
                delta = -1.0
                speak = False
            else:
                delta = ep.gate.advocate_score - lam * ep.gate.skeptic_score
                speak = delta >= tau_dm
            new_tier = InterventionTier.THREAD_BRIEF if speak else InterventionTier.SILENT
            new_gate = ep.gate.model_copy(
                update={
                    "margin_delta": round(delta, 6),
                    "final_tier": new_tier,
                    "binary_speak": speak,
                }
            )
            rescored.append(ep.model_copy(update={"gate": new_gate}))
        metrics = compute_when2speak_metrics(y_true, rescored, n_bootstrap=200)
        rows.append(
            {
                "lambda": float(lam),
                "macro_f1": metrics["macro_f1"],
                "fir": metrics["fir"],
                "mir": metrics["mir"],
                "speak_f1": metrics["speak_f1"],
                "auroc": metrics["auroc"],
            }
        )
    return rows


async def run_when2speak_eval(
    *,
    agent: DialSkillAgent | DirectChatBaseline | SingleGateBaseline | ProACTBaseline,
    turns: Sequence[When2SpeakTurn],
    output_jsonl: str | Path,
    resume: bool = True,
    generate_text: bool = False,
    max_concurrency: int = 8,
    show_progress: bool = True,
) -> list[EpisodeResult]:
    """Run When2Speak evaluation with conversation-parallel concurrency, live tqdm, and atomic --resume.

    Conversations run concurrently (up to `max_concurrency`), while turns within each conversation run
    in chronological order so Stage-1 prefix turns hit SQLite cache deterministically with zero waste.
    """
    out_path = Path(output_jsonl)
    writer = JsonlCheckpointWriter(out_path)
    completed = load_completed_ids(out_path, id_field="sample_id") if resume else set()

    pending_by_conv: dict[str, list[When2SpeakTurn]] = defaultdict(list)
    for turn in turns:
        if turn.sample_id not in completed:
            pending_by_conv[turn.conversation_id].append(turn)

    total_pending = sum(len(v) for v in pending_by_conv.values())
    pbar = tqdm(
        total=len(turns),
        initial=len(turns) - total_pending,
        desc=f"W2S[{out_path.stem}]",
        disable=not show_progress,
    )
    sem = asyncio.Semaphore(max(1, max_concurrency))
    write_lock = asyncio.Lock()

    async def _run_conv(conv_turns: list[When2SpeakTurn]) -> None:
        async with sem:
            for turn in conv_turns:
                kwargs: dict[str, Any] = {
                    "sample_id": turn.sample_id,
                    "conversation_id": turn.conversation_id,
                    "channel_name": turn.messages[-1].channel if turn.messages else "#group-chat",
                    "messages": turn.messages,
                    "benchmark": "when2speak",
                }
                if not isinstance(agent, DirectChatBaseline):
                    kwargs["generate_text"] = generate_text
                ep = await agent.run_episode(**kwargs)
                async with write_lock:
                    writer.append(ep)
                    completed.add(turn.sample_id)
                    pbar.update(1)

    try:
        if pending_by_conv:
            await asyncio.gather(*(_run_conv(cturns) for cturns in pending_by_conv.values()))
    finally:
        pbar.close()

    # Return ordered predictions matching turns
    by_id = {
        row["sample_id"]: EpisodeResult.model_validate(row)
        for row in load_jsonl(out_path)
        if "sample_id" in row
    }
    return [by_id[t.sample_id] for t in turns if t.sample_id in by_id]


async def run_pagemantle_eval(
    *,
    agent: DialSkillAgent | DirectChatBaseline | SingleGateBaseline | ProACTBaseline,
    tasks: Sequence[PageMantleTask],
    env: PageMantleEnv,
    output_jsonl: str | Path,
    resume: bool = True,
    include_slate_in_metadata: bool = False,
    show_progress: bool = True,
) -> list[EpisodeResult]:
    """Run PageMantle evaluation with tool-budget reset, live tqdm, atomic checkpointing, and --resume."""
    out_path = Path(output_jsonl)
    writer = JsonlCheckpointWriter(out_path)
    completed = load_completed_ids(out_path, id_field="sample_id") if resume else set()

    pending = [t for t in tasks if t.task_id not in completed]
    pbar = tqdm(
        total=len(tasks),
        initial=len(tasks) - len(pending),
        desc=f"PM[{out_path.stem}]",
        disable=not show_progress,
    )
    try:
        for task in pending:
            env.reset_budget()
            benchmark_tag: Literal["pagemantle_official", "pagemantle_ext"] = (
                "pagemantle_official" if task.split == "pm_official_8" else "pagemantle_ext"
            )
            kwargs: dict[str, Any] = {
                "sample_id": task.task_id,
                "conversation_id": task.task_id,
                "channel_name": task.channel_name,
                "messages": task.messages,
                "benchmark": benchmark_tag,
                "tool_env": env,
            }
            if isinstance(agent, DialSkillAgent):
                kwargs["generate_text"] = True
                kwargs["include_slate_in_metadata"] = include_slate_in_metadata
            elif not isinstance(agent, DirectChatBaseline):
                kwargs["generate_text"] = True

            ep = await agent.run_episode(**kwargs)
            writer.append(ep)
            completed.add(task.task_id)
            pbar.set_postfix(task=task.task_id, tier=ep.gate.final_tier.value)
            pbar.update(1)
    finally:
        pbar.close()

    by_id = {
        row["sample_id"]: EpisodeResult.model_validate(row)
        for row in load_jsonl(out_path)
        if "sample_id" in row
    }
    return [by_id[t.task_id] for t in tasks if t.task_id in by_id]


async def run_online_evolution_stream(
    *,
    agent: DialSkillAgent,
    updater: DBCSEUpdater,
    stream_tasks: Sequence[PageMantleTask],
    env: PageMantleEnv,
    run_dir: str | Path,
    judge: BlindedDualJudge | None = None,
) -> dict[str, Any]:
    """Run sequential online DB-CSE stream strictly on pm_online_stream_24 (Invariant #8)."""
    bad = [t.task_id for t in stream_tasks if t.split != "pm_online_stream_24"]
    if bad:
        raise AssertionError(
            f"Invariant #8 violated: non-stream tasks entered online evolution: {bad[:5]}"
        )

    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_writer = JsonlCheckpointWriter(out_dir / "stream_predictions.jsonl")

    wave_summaries: list[dict[str, Any]] = []
    prompt_token_curve: list[dict[str, int]] = [
        {"step": 0, "skill_store_tokens": agent.skill_store.estimate_serialized_tokens()}
    ]

    agent.skill_store.save(out_dir / "skill_store_v0.json")
    waves = sorted({t.wave or 1 for t in stream_tasks})
    step = 0

    for wave_id in waves:
        wave_tasks = [t for t in stream_tasks if (t.wave or 1) == wave_id]
        wave_eps: list[EpisodeResult] = []
        wave_scores = []

        for task in wave_tasks:
            step += 1
            env.reset_budget()
            ep = await agent.run_episode(
                sample_id=task.task_id,
                conversation_id=task.task_id,
                channel_name=task.channel_name,
                messages=task.messages,
                benchmark="pagemantle_ext",
                tool_env=env,
                generate_text=True,
                include_slate_in_metadata=True,
            )
            pred_writer.append(ep)
            wave_eps.append(ep)

            j_score = await judge.grade_pagemantle_episode(task, ep) if judge is not None else None
            if j_score is not None:
                wave_scores.append(j_score)

            fb = extract_feedback_from_pagemantle(ep, task, j_score)
            slate_dict = ep.metadata.get("slate")
            if slate_dict is not None:
                slate = SharedEvidenceSlate.model_validate(slate_dict)
                await updater.process_episode(
                    slate=slate,
                    gate=ep.gate,
                    feedback=fb,
                    rescore_fn=lambda sl, store: agent.rescore_slate_with_store(
                        slate=sl, candidate_store=store
                    ),
                )
            prompt_token_curve.append(
                {
                    "step": step,
                    "skill_store_tokens": agent.skill_store.estimate_serialized_tokens(),
                    "version": agent.skill_store.version,
                }
            )

        agent.skill_store.save(out_dir / f"skill_store_wave{wave_id}_v{agent.skill_store.version}.json")
        metrics = compute_pagemantle_metrics(wave_tasks, wave_eps, wave_scores or None, n_bootstrap=200)
        wave_summaries.append({"wave": wave_id, "skill_store_version": agent.skill_store.version, **metrics})

    summary = {
        "waves": wave_summaries,
        "dbcse_stats": updater.stats,
        "prompt_token_curve": prompt_token_curve,
        "final_skill_store_version": agent.skill_store.version,
    }
    (out_dir / "online_evolution_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _build_runtime_clients(
    drive_root: Path,
    triage_backend_override: Literal["jev", "openai_fallback"] | None = None,
) -> tuple[OpenAIClient, TriageClient, str, str, str]:
    models_cfg = load_models_config()
    resolved = load_resolved_models(drive_root)
    gate_model = resolve_model_for_role("gate", models_cfg, resolved)
    exec_model = resolve_model_for_role("executor", models_cfg, resolved)
    critic_model = resolve_model_for_role("critic", models_cfg, resolved)
    luna_model = resolve_model_for_role("luna", models_cfg, resolved)

    cache = SQLiteAPICache(drive_root / "caches" / "api_cache.sqlite")
    oai = OpenAIClient(models_cfg=models_cfg, cache=cache)
    backend = triage_backend_override or (resolved.triage_backend if resolved else models_cfg.roles.triage_backend)
    tc = TriageClient(
        models_cfg=models_cfg,
        cache=cache,
        openai_client=oai,
        luna_model=luna_model,
        backend=backend,
    )
    return oai, tc, gate_model, exec_model, critic_model


@app.command()
def pilot(
    config: Path = typer.Option(Path("configs/experiments.yaml"), help="Experiments config YAML."),
    drive_root: Path | None = typer.Option(None, help="Override DRIVE_ROOT."),
    resume: bool = typer.Option(True, help="Skip already completed sample_ids."),
) -> None:
    """Run Phase-1 Pilot Verification (When2Speak-Pilot 400 turns + PageMantle official 8) and print GO/NO-GO."""
    exp_cfg = load_experiments_config(config)
    root = ensure_drive_layout(drive_root)
    est = estimate_run_cost_usd(mode="pilot", include_pagemantle=True)
    typer.echo(f"Estimated Pilot Cost: {json.dumps(est)}")

    run_dir = root / "runs" / f"pilot_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    save_run_metadata(run_dir, {"command": "pilot", "exp_cfg": exp_cfg, "estimate": est})

    oai, tc, gate_model, exec_model, _ = _build_runtime_clients(root)
    dial_cfg = load_dial_skill_config()

    async def _run() -> None:
        w2s_turns = load_when2speak_split("pilot", local_dir=root / "datasets" / "when2speak", seed=exp_cfg["seed"])
        dial_agent = DialSkillAgent(
            triage_client=tc,
            openai_client=oai,
            gate_model=gate_model,
            executor_model=exec_model,
            cfg=dial_cfg,
        )
        direct_agent = DirectChatBaseline(openai_client=oai, model=gate_model)

        dial_eps = await run_when2speak_eval(
            agent=dial_agent,
            turns=w2s_turns,
            output_jsonl=run_dir / "predictions_when2speak_dial.jsonl",
            resume=resume,
        )
        direct_eps = await run_when2speak_eval(
            agent=direct_agent,
            turns=w2s_turns,
            output_jsonl=run_dir / "predictions_when2speak_direct.jsonl",
            resume=resume,
        )
        y_true = [t.label_speak for t in w2s_turns]
        w2s_dial_m = compute_when2speak_metrics(y_true, dial_eps)
        w2s_direct_m = compute_when2speak_metrics(y_true, direct_eps)
        summary = {"when2speak_dial": w2s_dial_m, "when2speak_direct": w2s_direct_m, "triage_meta": tc.get_run_metadata()}
        (run_dir / "pilot_scorecard.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        typer.echo(json.dumps(summary, indent=2))

    asyncio.run(_run())


@app.command()
def estimate(
    mode: Literal["pilot", "main", "full"] = typer.Option("pilot"),
    include_ablations: bool = typer.Option(False),
) -> None:
    """Print estimated API calls and USD cost before launching a sweep."""
    typer.echo(json.dumps(estimate_run_cost_usd(mode=mode, include_ablations=include_ablations), indent=2))


if __name__ == "__main__":
    app()
