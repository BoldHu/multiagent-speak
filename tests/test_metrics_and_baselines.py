"""Milestone 5 tests: B0/B1/B2 baselines, When2Speak & PageMantle metrics, Blinded Dual-Judge, resume & calibration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from dial_skill.baselines.direct_chat import DirectChatBaseline
from dial_skill.baselines.proact import ProACTBaseline
from dial_skill.baselines.single_gate import SingleGateBaseline
from dial_skill.eval.judge import (
    BlindedDualJudge,
    compute_cohens_kappa,
    compute_spearman_rho,
    export_human_spotcheck_csv,
)
from dial_skill.eval.metrics import (
    compute_ece,
    compute_pagemantle_metrics,
    compute_when2speak_metrics,
    evaluate_pilot_go_nogo,
    mcnemar_test,
)
from dial_skill.eval.runners import (
    calibrate_when2speak_thresholds,
    run_when2speak_eval,
    sweep_lambda_pareto_from_episodes,
)
from dial_skill.llm.jev_client import TriageClient
from dial_skill.llm.openai_client import OpenAIClient
from dial_skill.schemas import (
    ChatMessage,
    EpisodeResult,
    GateDecision,
    InterventionTier,
    JudgeRubricScore,
    PageMantleTask,
    When2SpeakTurn,
)
from dial_skill.utils.cache import SQLiteAPICache


class _BaselineFakeOpenAI:
    def __init__(self) -> None:
        self.calls = 0
        outer = self

        class _Completions:
            async def create(self, **kwargs: Any) -> Any:
                outer.calls += 1
                sys_msg = kwargs["messages"][0]["content"]
                user_msg = kwargs["messages"][1]["content"]

                if "speak_probability" in sys_msg:
                    prob = 0.85 if "help" in user_msg.lower() else 0.10
                    content = json.dumps({"speak_probability": prob, "rationale": "confidence check"})
                elif "ProACT" in sys_msg:
                    hit = "help" in user_msg.lower()
                    content = json.dumps(
                        {
                            "breakdown_detected": hit,
                            "breakdown_type": "factual_gap" if hit else "none",
                            "should_intervene": hit,
                            "target_skill": "find" if hit else "none",
                            "rationale": "breakdown check",
                        }
                    )
                elif "blinded evaluator grading a proactive workplace" in sys_msg:
                    content = json.dumps(
                        {
                            "intervention_or_silence": 10.0,
                            "content_and_evidence": 8.0,
                            "clarity_and_tone": 9.0,
                            "audience_and_location": 8.0,
                            "overall_score": 8.75,
                            "justifications": {
                                "intervention_or_silence": "Right timing.",
                                "content_and_evidence": "Accurate.",
                                "clarity_and_tone": "Concise.",
                                "audience_and_location": "Good channel.",
                            },
                        }
                    )
                elif "exact token: SILENT" in sys_msg:
                    content = "[Thread] Here is the answer!" if "help" in user_msg.lower() else "SILENT"
                else:
                    content = "Here is the verified finding."

                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
                    usage=SimpleNamespace(prompt_tokens=100, completion_tokens=30),
                )

        self.chat = SimpleNamespace(completions=_Completions())


def test_b0_b1_b2_baselines_execute_and_respect_silence(tmp_path: Path) -> None:
    fake = _BaselineFakeOpenAI()
    cache = SQLiteAPICache(tmp_path / "bl.sqlite")
    oai = OpenAIClient(api_key="k", cache=cache, async_client=fake)

    msg_need = [ChatMessage(msg_id="m1", speaker="alice", text="Can someone help find the billing doc?")]
    msg_quiet = [ChatMessage(msg_id="m2", speaker="bob", text="Sounds good, see you tomorrow!")]

    b0 = DirectChatBaseline(openai_client=oai, model="gpt-6-luna")
    b1_llm = SingleGateBaseline(openai_client=oai, gate_model="gpt-6-luna", gate_source="llm")
    b2 = ProACTBaseline(openai_client=oai, gate_model="gpt-6-luna")

    http_jev = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "answers": {
                        "decision_change": {"probability": 0.9},
                        "stance": {"choice": "need_signal", "probabilities": {"need_signal": 0.9, "background_chatter": 0.1}},
                        "skill_hint": {"choice": "find", "probabilities": {"find": 0.8, "none": 0.2}},
                        "urgency": {"score": 0.7},
                    }
                },
            )
        )
    )
    tc = TriageClient(cache=SQLiteAPICache(tmp_path / "j.sqlite"), jev_api_key="k", http_client=http_jev)
    b1_jev = SingleGateBaseline(openai_client=oai, gate_model="gpt-6-luna", triage_client=tc, gate_source="jev")

    async def _run() -> dict[str, EpisodeResult]:
        out = {
            "b0_need": await b0.run_episode(sample_id="1", conversation_id="c1", channel_name="#g", messages=msg_need, benchmark="when2speak"),
            "b0_quiet": await b0.run_episode(sample_id="2", conversation_id="c2", channel_name="#g", messages=msg_quiet, benchmark="when2speak"),
            "b1_need": await b1_llm.run_episode(sample_id="3", conversation_id="c1", channel_name="#g", messages=msg_need, benchmark="when2speak", generate_text=False),
            "b1_quiet": await b1_llm.run_episode(sample_id="4", conversation_id="c2", channel_name="#g", messages=msg_quiet, benchmark="when2speak", generate_text=False),
            "b1_jev": await b1_jev.run_episode(sample_id="5", conversation_id="c1", channel_name="#g", messages=msg_need, benchmark="when2speak", generate_text=False),
            "b2_need": await b2.run_episode(sample_id="6", conversation_id="c1", channel_name="#g", messages=msg_need, benchmark="when2speak", generate_text=True),
            "b2_quiet": await b2.run_episode(sample_id="7", conversation_id="c2", channel_name="#g", messages=msg_quiet, benchmark="when2speak", generate_text=True),
        }
        await http_jev.aclose()
        return out

    res = asyncio.run(_run())
    assert res["b0_need"].gate.binary_speak is True
    assert res["b0_need"].gate.final_tier == InterventionTier.THREAD_BRIEF
    assert res["b0_quiet"].gate.binary_speak is False
    assert res["b1_need"].gate.binary_speak is True
    assert res["b1_quiet"].gate.binary_speak is False
    assert res["b1_jev"].gate.binary_speak is True
    assert res["b2_need"].gate.binary_speak is True
    assert res["b2_need"].gate.selected_skill == "find"
    assert res["b2_need"].gate.final_tier == InterventionTier.CHANNEL_INTERVENE
    assert res["b2_quiet"].gate.binary_speak is False
    cache.close()


def test_when2speak_pagemantle_metrics_mcnemar_and_pilot_scorecard() -> None:
    def _mk_ep(sid: str, speak: bool, delta: float, tier: InterventionTier) -> EpisodeResult:
        return EpisodeResult(
            sample_id=sid,
            benchmark="when2speak",
            method="dial_skill",
            model_id="luna",
            gate=GateDecision(
                slate_hash="h",
                selected_skill="find",
                advocate_score=0.8 if speak else 0.2,
                skeptic_score=0.2 if speak else 0.8,
                margin_delta=delta,
                disagreement_u=0.2,
                raw_tier=tier,
                final_tier=tier,
                binary_speak=speak,
            ),
            prompt_tokens_raw=500,
            prompt_tokens_slate=200,
            total_cost_usd=0.002,
        )

    y_true = [True, True, True, False, False, False, False, False, False, False]
    eps = [
        _mk_ep("0", True, 0.6, InterventionTier.THREAD_BRIEF),
        _mk_ep("1", True, 0.4, InterventionTier.THREAD_BRIEF),
        _mk_ep("2", False, -0.1, InterventionTier.SILENT),
        *[_mk_ep(str(i), False, -0.5, InterventionTier.SILENT) for i in range(3, 10)],
    ]
    m = compute_when2speak_metrics(y_true, eps, n_bootstrap=200)
    assert m["tp"] == 2 and m["fn"] == 1 and m["fp"] == 0 and m["tn"] == 7
    assert m["fir"] == 0.0
    assert abs(m["mir"] - 1 / 3) < 1e-4
    assert m["auroc"] > 0.9
    assert m["token_reduction_pct"] == 60.0
    assert 0.0 <= compute_ece([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2]) <= 0.2

    mc = mcnemar_test(y_true, [e.gate.binary_speak for e in eps], [False] * 10)
    assert mc["n_a_correct_b_wrong"] == 2
    assert mc["n_b_correct_a_wrong"] == 0

    # PageMantle metrics
    tasks = [
        PageMantleTask(
            task_id="pm1",
            split="pm_eval_32",
            category="guard",
            messages=[ChatMessage(msg_id="m", speaker="a", text="x")],
            expected_action="INTERVENE",
            ideal_tier=InterventionTier.DM_NUDGE,
        ),
        PageMantleTask(
            task_id="pm2",
            split="pm_eval_32",
            category="check",
            messages=[ChatMessage(msg_id="m", speaker="a", text="x")],
            expected_action="SILENT",
            ideal_tier=InterventionTier.SILENT,
            trap_type="already_resolved",
        ),
    ]
    pm_eps = [
        _mk_ep("pm1", True, 0.3, InterventionTier.DM_NUDGE),
        _mk_ep("pm2", False, -0.4, InterventionTier.SILENT),
    ]
    judges = [
        JudgeRubricScore(sample_id="pm1", judge_model="terra", intervention_or_silence=10, content_and_evidence=8, clarity_and_tone=9, audience_and_location=10, overall_score=9.15),
        JudgeRubricScore(sample_id="pm2", judge_model="terra", intervention_or_silence=10, content_and_evidence=10, clarity_and_tone=10, audience_and_location=10, overall_score=10.0),
    ]
    pm_m = compute_pagemantle_metrics(tasks, pm_eps, judges, n_bootstrap=200)
    assert pm_m["silence_accuracy"] == 1.0
    assert pm_m["action_recall"] == 1.0
    assert pm_m["channel_match_rate"] == 1.0
    assert pm_m["overall_score"] == 9.575

    # Pilot GO/NO-GO scorecard
    go = evaluate_pilot_go_nogo(
        w2s_dial={"macro_f1": 0.72, "fir": 0.06, "mir": 0.28, "margin_gap": 0.35, "auroc": 0.86},
        w2s_direct={"macro_f1": 0.58},
        pm_dial={"overall_score": 7.6, "silence_accuracy": 0.85, "timeout_rate": 0.0},
        pm_direct={"overall_score": 6.4},
        triage_long_chat_token_reduction=0.55,
        triage_decisive_recall=0.95,
        dbcse_anchor_regressions=0,
        dbcse_heldout_delta=0.45,
    )
    assert go["verdict"] == "GO"
    assert go["passed_checks"] == 5


def test_blinded_dual_judge_agreement_and_spotcheck_csv(tmp_path: Path) -> None:
    fake = _BaselineFakeOpenAI()
    oai = OpenAIClient(api_key="k", cache=SQLiteAPICache(tmp_path / "j.sqlite"), async_client=fake)
    judge = BlindedDualJudge(openai_client=oai, primary_judge_model="gpt-6.1-terra")

    task = PageMantleTask(
        task_id="pm_1",
        split="pm_official_8",
        category="diagnose",
        messages=[ChatMessage(msg_id="m0", speaker="alice", text="Invoice failed")],
        expected_action="INTERVENE",
        ideal_tier=InterventionTier.THREAD_BRIEF,
        reference_answer="Card expired on INV-42.",
    )
    ep = EpisodeResult(
        sample_id="pm_1",
        benchmark="pagemantle_official",
        method="secret_dial_skill_v2",
        model_id="gpt-6-luna",
        gate=GateDecision(
            slate_hash="h",
            selected_skill="diagnose",
            advocate_score=0.9,
            skeptic_score=0.1,
            margin_delta=0.8,
            disagreement_u=0.1,
            raw_tier=InterventionTier.THREAD_BRIEF,
            final_tier=InterventionTier.THREAD_BRIEF,
            binary_speak=True,
        ),
        response_text="Card expired on INV-42.",
    )
    # Verify method name is NEVER leaked into blinded prompt!
    p0 = judge._build_blinded_prompt(task, ep, 0)
    assert "secret_dial_skill_v2" not in p0
    assert "CANDIDATE_" in p0

    score = asyncio.run(judge.grade_pagemantle_episode(task, ep))
    assert score.overall_score == 8.75
    assert fake.calls == 2  # Dual passes!

    assert compute_cohens_kappa([1, 1, 0, 0, 1], [1, 1, 0, 0, 1]) == 1.0
    assert compute_spearman_rho([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])["rho"] == 1.0

    from dial_skill.eval.judge import _coerce_rubric_score

    assert _coerce_rubric_score({"score": 8.5, "reason": "good"}) == 8.5
    assert _coerce_rubric_score("9.0/10") == 9.0
    assert _coerce_rubric_score(12.0) == 10.0

    csv_path = export_human_spotcheck_csv(
        episodes=[ep],
        tasks=[task],
        judge_scores=[score],
        out_csv=tmp_path / "human_calibration_30.csv",
    )
    assert csv_path.is_file()
    content = csv_path.read_text(encoding="utf-8")
    assert "secret_dial_skill_v2" not in content
    assert "SPOT_01_" in content


def test_runner_resume_idempotency_calibration_and_pareto_sweep(tmp_path: Path) -> None:
    fake = _BaselineFakeOpenAI()
    oai = OpenAIClient(api_key="k", cache=SQLiteAPICache(tmp_path / "r.sqlite"), async_client=fake)
    b1 = SingleGateBaseline(openai_client=oai, gate_model="gpt-6-luna", gate_source="llm")

    turns = [
        When2SpeakTurn(
            sample_id=f"s_{i}",
            conversation_id=f"c_{i}",
            turn_index=0,
            split="validation",
            messages=[ChatMessage(msg_id=f"m_{i}", speaker="a", text="Please help" if i % 2 == 0 else "Morning")],
            label_speak=(i % 2 == 0),
        )
        for i in range(4)
    ]
    out_jsonl = tmp_path / "preds.jsonl"

    # Run first 2 turns, then resume over all 4 turns -> total OpenAI calls must be exactly 4 (0 duplicates!)
    eps_part = asyncio.run(run_when2speak_eval(agent=b1, turns=turns[:2], output_jsonl=out_jsonl, resume=True))
    assert len(eps_part) == 2
    assert fake.calls == 2

    eps_all = asyncio.run(run_when2speak_eval(agent=b1, turns=turns, output_jsonl=out_jsonl, resume=True))
    assert len(eps_all) == 4
    assert fake.calls == 4  # Idempotent --resume skipped s_0 and s_1!

    cal = calibrate_when2speak_thresholds(val_turns=turns, val_episodes=eps_all)
    cal_rev = calibrate_when2speak_thresholds(val_turns=turns, val_episodes=list(reversed(eps_all)))
    assert cal["val_macro_f1"] == 1.0
    assert cal_rev == cal

    pareto = sweep_lambda_pareto_from_episodes(turns=turns, episodes=eps_all, lambdas=[0.5, 1.0, 1.5])
    pareto_rev = sweep_lambda_pareto_from_episodes(turns=turns, episodes=list(reversed(eps_all)), lambdas=[0.5, 1.0, 1.5])
    assert len(pareto) == 3
    assert all("fir" in row and "mir" in row for row in pareto)
    assert pareto_rev == pareto
