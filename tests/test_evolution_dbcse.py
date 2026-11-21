"""Milestone 3 tests: DB-CSE feedback extraction, bounded numeric updates, anti-overfitting guardrails, EscrowGate 0-regression, O(1) cap, rollback, and Reflexion/AWM baselines."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dial_skill.config import load_dial_skill_config
from dial_skill.core.skill_store import SkillStore
from dial_skill.evolution.baselines import AWMWorkflowUpdater, ReflexionAppendUpdater
from dial_skill.evolution.db_cse import DBCSEUpdater, validate_generalized_predicate
from dial_skill.evolution.feedback import extract_feedback_from_pagemantle, extract_feedback_from_when2speak
from dial_skill.llm.openai_client import OpenAIClient
from dial_skill.schemas import (
    AnchorEpisode,
    ChatMessage,
    EpisodeFeedback,
    EpisodeResult,
    GateDecision,
    InterventionTier,
    JudgeRubricScore,
    PageMantleTask,
    SharedEvidenceSlate,
)
from dial_skill.utils.cache import SQLiteAPICache
from dial_skill.utils.checkpoint import load_jsonl


class _FakeCriticOpenAI:
    def __init__(
        self,
        attribution: dict[str, Any],
        proposal: dict[str, Any],
        consolidated: dict[str, Any] | None = None,
        text_reply: str = "Next time verify ownership before replying in public channel.",
    ) -> None:
        self.attribution = attribution
        self.proposal = proposal
        self.consolidated = consolidated
        self.text_reply = text_reply
        outer = self

        class _Completions:
            async def create(self, **kwargs: Any) -> Any:
                sys_msg = kwargs["messages"][0]["content"]
                if "Counterfactual Attribution Critic" in sys_msg:
                    content = json.dumps(outer.attribution)
                elif "Propose EXACTLY ONE atomic" in sys_msg:
                    content = json.dumps(outer.proposal)
                elif "strict capacity cap" in sys_msg:
                    if outer.consolidated is not None:
                        content = json.dumps(outer.consolidated)
                    else:
                        content = json.dumps({"consolidated_rules": [], "action_taken": "pruned"})
                else:
                    content = outer.text_reply
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
                    usage=SimpleNamespace(prompt_tokens=100, completion_tokens=30),
                )

        self.chat = SimpleNamespace(completions=_Completions())


def _make_slate(store: SkillStore, skill: str = "check", text: str = "SLATE") -> SharedEvidenceSlate:
    return SharedEvidenceSlate(
        conversation_id="c1",
        channel_name="#ops",
        env_mode="pagemantle",
        latest_msg_id="m1",
        selected_messages=[ChatMessage(msg_id="m1", speaker="u1", text="test")],
        triage_annotations={},
        skill_priors={k: (0.9 if k == skill else 0.0) for k in ["find", "explain", "analyze", "check", "diagnose", "decide", "coordinate", "guard", "none"]},
        candidate_skills=[store.require(skill)],
        canonical_text=text,
        slate_hash=f"hash_{text}",
    )


def _make_gate(tier: InterventionTier, margin: float, skill: str = "check") -> GateDecision:
    return GateDecision(
        short_circuited=False,
        slate_hash="h",
        selected_skill=skill,  # type: ignore[arg-type]
        advocate_score=0.75,
        skeptic_score=0.30,
        margin_delta=margin,
        disagreement_u=0.30,
        raw_tier=tier,
        final_tier=tier,
        binary_speak=(tier != InterventionTier.SILENT),
    )


def test_feedback_extraction_all_classes() -> None:
    task_silent = PageMantleTask(
        task_id="t_trap",
        split="pm_online_stream_24",
        wave=1,
        category="check",
        messages=[ChatMessage(msg_id="m1", speaker="a", text="x")],
        expected_action="SILENT",
        ideal_tier=InterventionTier.SILENT,
        trap_type="already_resolved",
    )
    task_action = PageMantleTask(
        task_id="t_act",
        split="pm_online_stream_24",
        wave=1,
        category="guard",
        messages=[ChatMessage(msg_id="m1", speaker="a", text="x")],
        expected_action="INTERVENE",
        ideal_tier=InterventionTier.DM_NUDGE,
    )

    def _ep(tier: InterventionTier, skill: str = "check") -> EpisodeResult:
        return EpisodeResult(
            sample_id="s1",
            benchmark="pagemantle_ext",
            method="dial_skill",
            model_id="luna",
            gate=_make_gate(tier, 0.3, skill=skill),
        )

    assert extract_feedback_from_pagemantle(_ep(InterventionTier.SILENT), task_silent).feedback_type == "TN_VALIDATED"
    assert extract_feedback_from_pagemantle(_ep(InterventionTier.THREAD_BRIEF), task_silent).feedback_type == "FP_INTRUSION"
    assert extract_feedback_from_pagemantle(_ep(InterventionTier.SILENT), task_action).feedback_type == "FN_OMISSION"
    assert extract_feedback_from_pagemantle(_ep(InterventionTier.CHANNEL_INTERVENE, "guard"), task_action).feedback_type == "FP_SCOPE_OVERREACH"

    low_ev_judge = JudgeRubricScore(
        sample_id="s1",
        judge_model="terra",
        intervention_or_silence=10.0,
        content_and_evidence=4.0,
        clarity_and_tone=8.0,
        audience_and_location=9.0,
        overall_score=7.4,
    )
    assert extract_feedback_from_pagemantle(_ep(InterventionTier.DM_NUDGE, "guard"), task_action, low_ev_judge).feedback_type == "CONTENT_EVIDENCE_GAP"
    assert extract_feedback_from_when2speak(_ep(InterventionTier.THREAD_BRIEF), label_speak=True).feedback_type == "TP_VALIDATED"
    assert extract_feedback_from_when2speak(_ep(InterventionTier.THREAD_BRIEF), label_speak=False).feedback_type == "FP_INTRUSION"


def test_anti_overfitting_guardrail() -> None:
    existing = ["Verification confirms the stated claim is already accurate."]
    ok, reason = validate_generalized_predicate(
        "Stay silent when an owner explicitly defers verification until after a planned release.",
        existing,
    )
    assert ok is True and reason is None

    # Reject ticket IDs, channels, user handles, proper nouns, >20 words, and duplicates
    assert validate_generalized_predicate("Never interrupt about ticket ENG-412.", existing)[0] is False
    assert validate_generalized_predicate("Stay silent in #billing-ops channel.", existing)[0] is False
    assert validate_generalized_predicate("Defer when @alice says she will check.", existing)[0] is False
    assert validate_generalized_predicate("Defer when Alice checks Acme Corp billing.", existing)[0] is False
    assert validate_generalized_predicate("word " * 25, existing)[0] is False
    assert validate_generalized_predicate(existing[0], existing)[0] is False


def test_bounded_numeric_clips_and_tp_decay(tmp_path: Path) -> None:
    store = SkillStore.from_json_file()
    fake = _FakeCriticOpenAI({}, {})
    oai = OpenAIClient(api_key="k", cache=SQLiteAPICache(tmp_path / "c.sqlite"), async_client=fake)
    updater = DBCSEUpdater(skill_store=store, openai_client=oai, critic_model="gpt-6.1-terra")

    for _ in range(20):
        updater.apply_numeric_update("check", "FP_INTRUSION")
    assert store.require("check").margin_offset == 0.35  # clipped at +0.35

    for _ in range(30):
        updater.apply_numeric_update("check", "FN_OMISSION")
    assert store.require("check").margin_offset == -0.35  # clipped at -0.35

    for _ in range(10):
        updater.apply_numeric_update("check", "FP_SCOPE_OVERREACH")
    assert store.require("check").deescalation_bias == 2.0  # clipped at 2.0

    before = store.require("check").margin_offset
    updater.apply_numeric_update("check", "TP_VALIDATED")
    assert abs(store.require("check").margin_offset - round(before * 0.98, 6)) < 1e-6


def test_escrow_gate_promotes_safe_patch_and_blocks_regression(tmp_path: Path) -> None:
    store = SkillStore.from_json_file()
    log_path = tmp_path / "skill_evolution_log.jsonl"
    cfg = load_dial_skill_config()

    safe_rule = "Stay silent when a team lead explicitly defers verification to an offline review."
    fake = _FakeCriticOpenAI(
        attribution={
            "root_cause": "VETO_MISSING",
            "affected_skill": "check",
            "counterfactual_explanation": "Missing explicit deferral veto.",
            "needs_symbolic_patch": True,
        },
        proposal={
            "target_field": "veto_conditions",
            "proposed_predicate": safe_rule,
            "rationale": "Avoid interrupting explicitly deferred threads.",
        },
    )
    oai = OpenAIClient(api_key="k", cache=SQLiteAPICache(tmp_path / "esc.sqlite"), async_client=fake)
    updater = DBCSEUpdater(
        skill_store=store,
        openai_client=oai,
        critic_model="gpt-6.1-terra",
        cfg=cfg,
        log_path=log_path,
    )

    # Seed 3 verified anchors (2 positive, 1 negative)
    for idx, exp_speak in enumerate([True, True, False]):
        updater.anchor_buffer.add(
            AnchorEpisode(
                sample_id=f"anchor_{idx}",
                skill_name="check",
                expected_speak=exp_speak,
                expected_tier=InterventionTier.THREAD_BRIEF if exp_speak else InterventionTier.SILENT,
                slate=_make_slate(store, "check", f"ANCHOR_{idx}"),
                original_margin=0.40 if exp_speak else -0.20,
            )
        )

    failed_slate = _make_slate(store, "check", "FAILED_TRAP")
    failed_gate = _make_gate(InterventionTier.THREAD_BRIEF, 0.30, "check")
    fb = EpisodeFeedback(
        sample_id="fail_1",
        skill_name="check",
        feedback_type="FP_INTRUSION",
        critique="Interrupted an explicitly deferred check.",
        expected_speak=False,
        expected_tier=InterventionTier.SILENT,
        predicted_tier=InterventionTier.THREAD_BRIEF,
    )

    # Case A: Bad candidate causes regression on ANCHOR_0 -> EscrowGate MUST BLOCK!
    async def _regressing_rescore(sl: SharedEvidenceSlate, cand: SkillStore) -> GateDecision:
        if sl.canonical_text == "FAILED_TRAP":
            return _make_gate(InterventionTier.SILENT, -0.10, "check")
        if sl.canonical_text == "ANCHOR_0":
            return _make_gate(InterventionTier.SILENT, -0.10, "check")  # REGRESSION! (expected_speak=True)
        if sl.canonical_text == "ANCHOR_1":
            return _make_gate(InterventionTier.THREAD_BRIEF, 0.40, "check")
        return _make_gate(InterventionTier.SILENT, -0.20, "check")

    entry_blocked = asyncio.run(
        updater.process_episode(
            slate=failed_slate,
            gate=failed_gate,
            feedback=fb,
            rescore_fn=_regressing_rescore,
        )
    )
    assert entry_blocked.promoted is False
    assert entry_blocked.escrow_passed is False
    assert entry_blocked.escrow_details["anchor_regressions"] == 1
    assert safe_rule not in store.require("check").veto_conditions
    assert store.version == 0

    # Save snapshot before promotion to test rollback later
    snap_v0 = tmp_path / "skill_store_v0.json"
    store.save(snap_v0)

    # Case B: Safe candidate fixes failure AND has 0 regressions on all 3 anchors -> PROMOTE!
    async def _clean_rescore(sl: SharedEvidenceSlate, cand: SkillStore) -> GateDecision:
        if sl.canonical_text == "FAILED_TRAP":
            return _make_gate(InterventionTier.SILENT, -0.15, "check")
        if sl.canonical_text in ("ANCHOR_0", "ANCHOR_1"):
            return _make_gate(InterventionTier.THREAD_BRIEF, 0.40, "check")
        return _make_gate(InterventionTier.SILENT, -0.20, "check")

    entry_promoted = asyncio.run(
        updater.process_episode(
            slate=failed_slate,
            gate=failed_gate,
            feedback=fb,
            rescore_fn=_clean_rescore,
        )
    )
    assert entry_promoted.promoted is True
    assert entry_promoted.escrow_passed is True
    assert entry_promoted.escrow_details["anchor_regressions"] == 0
    assert safe_rule in store.require("check").veto_conditions
    assert store.version == 1

    # Verify audit log persisted both events
    logs = load_jsonl(log_path)
    assert len(logs) == 2

    # Verify clean rollback to v0
    updater.rollback_from_snapshot(snap_v0)
    assert store.version == 0
    assert safe_rule not in store.require("check").veto_conditions


def test_o1_prompt_cap_enforced_via_consolidate_or_prune(tmp_path: Path) -> None:
    store = SkillStore.from_json_file()
    # Fill check.veto_conditions up to hard cap of 5 rules
    card = store.require("check")
    full_vetos = [
        *card.veto_conditions,
        "Defer when participants explicitly pause verification until a later sync.",
        "Stay silent when two domain owners are already checking source records live.",
        "Avoid interrupting casual informal estimation with zero operational commitment.",
    ]
    assert len(full_vetos) == 5
    store.set_skill(card.model_copy(update={"veto_conditions": full_vetos}))

    new_rule = "Stay silent when an analyst already posted the verified source figure."
    fake = _FakeCriticOpenAI(
        attribution={
            "root_cause": "VETO_MISSING",
            "affected_skill": "check",
            "counterfactual_explanation": "Redundant check after analyst already verified.",
            "needs_symbolic_patch": True,
        },
        proposal={
            "target_field": "veto_conditions",
            "proposed_predicate": new_rule,
            "rationale": "Avoid redundant confirmations.",
        },
        consolidated={
            "consolidated_rules": [*full_vetos[:4], new_rule],
            "action_taken": "merged",
        },
    )
    oai = OpenAIClient(api_key="k", cache=SQLiteAPICache(tmp_path / "o1.sqlite"), async_client=fake)
    updater = DBCSEUpdater(skill_store=store, openai_client=oai, critic_model="gpt-6.1-terra")

    async def _fix_rescore(sl: SharedEvidenceSlate, cand: SkillStore) -> GateDecision:
        return _make_gate(InterventionTier.SILENT, -0.20, "check")

    entry = asyncio.run(
        updater.process_episode(
            slate=_make_slate(store, "check", "OVER_CAP"),
            gate=_make_gate(InterventionTier.THREAD_BRIEF, 0.25, "check"),
            feedback=EpisodeFeedback(
                sample_id="cap_1",
                skill_name="check",
                feedback_type="FP_INTRUSION",
                critique="Redundant reply.",
                expected_speak=False,
                expected_tier=InterventionTier.SILENT,
                predicted_tier=InterventionTier.THREAD_BRIEF,
            ),
            rescore_fn=_fix_rescore,
        )
    )
    assert entry.promoted is True
    assert entry.pruning_action == "merged"
    assert len(store.require("check").veto_conditions) == 5  # Hard O(1) cap preserved!
    store.validate_o1_caps()


def test_reflexion_and_awm_online_baselines(tmp_path: Path) -> None:
    fake = _FakeCriticOpenAI({}, {}, text_reply="Next time check Linear before speaking in #general.")
    oai = OpenAIClient(api_key="k", cache=SQLiteAPICache(tmp_path / "bl.sqlite"), async_client=fake)

    refl = ReflexionAppendUpdater(oai, "gpt-6.1-terra")
    awm = AWMWorkflowUpdater(oai, "gpt-6.1-terra")

    ep = EpisodeResult(
        sample_id="e1",
        benchmark="pagemantle_ext",
        method="reflexion",
        model_id="luna",
        gate=_make_gate(InterventionTier.CHANNEL_INTERVENE, 0.5, "diagnose"),
        response_text="Checked billing.",
        metadata={"tool_trace": [{"tool": "stripe_get_invoice", "arguments": {"id": "1"}}]},
    )
    fb = EpisodeFeedback(
        sample_id="e1",
        skill_name="diagnose",
        feedback_type="FP_INTRUSION",
        critique="Too noisy.",
        predicted_tier=InterventionTier.CHANNEL_INTERVENE,
    )

    t0 = refl.estimate_memory_tokens()
    asyncio.run(refl.process_episode(ep, fb))
    t1 = refl.estimate_memory_tokens()
    asyncio.run(refl.process_episode(ep.model_copy(update={"sample_id": "e2"}), fb))
    t2 = refl.estimate_memory_tokens()
    assert t2 > t1 > t0  # Linear O(T) growth

    asyncio.run(awm.process_episode(ep, fb))
    assert len(awm.workflows) == 1
    assert awm.estimate_memory_tokens() > 0


def test_dbcse_targets_affected_skill_for_numeric_and_escrow_replay(tmp_path: Path) -> None:
    """Numeric updates AND EscrowGate replay must target attribution.affected_skill even when absent from slate."""
    store = SkillStore.from_json_file()
    new_trigger = "Intervene when a teammate cites an unverified metric that contradicts warehouse records."
    fake = _FakeCriticOpenAI(
        attribution={
            "root_cause": "TRIGGER_MISSING",
            "affected_skill": "check",
            "counterfactual_explanation": "Verification trigger belongs on check skill.",
            "needs_symbolic_patch": True,
        },
        proposal={
            "target_field": "trigger_conditions",
            "proposed_predicate": new_trigger,
            "rationale": "Catch unverified claims.",
        },
    )
    oai = OpenAIClient(api_key="k", cache=SQLiteAPICache(tmp_path / "aff.sqlite"), async_client=fake)
    updater = DBCSEUpdater(skill_store=store, openai_client=oai, critic_model="gpt-6.1-terra")

    # Original slate only had `find` in candidate_skills, NOT `check`!
    slate = _make_slate(store, "find", "SLATE_WITHOUT_CHECK")
    gate = _make_gate(InterventionTier.SILENT, -0.05, "find")
    fb = EpisodeFeedback(
        sample_id="fn_mismatch",
        skill_name="find",
        feedback_type="FN_OMISSION",
        critique="Missed factual check.",
        expected_speak=True,
        expected_tier=InterventionTier.THREAD_BRIEF,
        predicted_tier=InterventionTier.SILENT,
    )

    seen_candidate_names: list[list[str]] = []

    async def _rescore(sl: SharedEvidenceSlate, cand: SkillStore) -> GateDecision:
        seen_candidate_names.append([c.name for c in sl.candidate_skills])
        # Verify the newly patched `check` skill is visible during replay!
        assert any(c.name == "check" and new_trigger in c.trigger_conditions for c in sl.candidate_skills)
        return _make_gate(InterventionTier.THREAD_BRIEF, 0.25, "check")

    entry = asyncio.run(
        updater.process_episode(slate=slate, gate=gate, feedback=fb, rescore_fn=_rescore)
    )
    assert entry.promoted is True
    assert entry.skill_name == "check"
    # Numeric update MUST have landed on `check`, NOT on `find`!
    assert store.require("find").margin_offset == 0.0
    assert store.require("check").margin_offset == -0.06
    assert new_trigger in store.require("check").trigger_conditions
    assert seen_candidate_names == [["find", "check"]]

