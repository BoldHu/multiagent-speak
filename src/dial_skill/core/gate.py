"""Stage 2: Symmetric-Evidence Adversarial Margin Gating & 4-Tier Dynamic De-escalation."""

from __future__ import annotations

import asyncio
from typing import Any

from dial_skill.config import DialSkillConfig
from dial_skill.core.skill_store import SkillStore
from dial_skill.llm.openai_client import LLMCallResult, OpenAIClient
from dial_skill.prompts.templates import (
    ADVOCATE_JSON_SCHEMA,
    SINGLE_COMBINED_GATE_JSON_SCHEMA,
    SKEPTIC_JSON_SCHEMA,
    build_advocate_messages,
    build_single_combined_messages,
    build_skeptic_messages,
)
from dial_skill.schemas import (
    ACTIVE_SKILLS,
    INT_TO_TIER,
    TIER_TO_INT,
    AdvocateOutput,
    GateDecision,
    InterventionTier,
    SharedEvidenceSlate,
    SkepticOutput,
    SkillName,
)


def _select_candidate_skill(
    advocate: AdvocateOutput | None,
    skill_priors: dict[SkillName, float],
    prior_floor: float,
    disable_skill_routing: bool,
) -> SkillName:
    if disable_skill_routing:
        return "none"
    if (
        advocate is not None
        and advocate.recommended_skill != "none"
        and float(skill_priors.get(advocate.recommended_skill, 0.0)) >= prior_floor
    ):
        return advocate.recommended_skill
    best_skill = max(ACTIVE_SKILLS, key=lambda k: float(skill_priors.get(k, 0.0)))
    if float(skill_priors.get(best_skill, 0.0)) > 0.0:
        return best_skill
    if advocate is not None and advocate.recommended_skill != "none":
        return advocate.recommended_skill
    return "none"


def compute_gate_decision(
    *,
    slate: SharedEvidenceSlate,
    advocate: AdvocateOutput | None,
    skeptic: SkepticOutput | None,
    skill_store: SkillStore,
    cfg: DialSkillConfig,
) -> GateDecision:
    """Deterministic Stage-2 margin, disagreement, and 4-tier dynamic de-escalation computation."""
    # Invariant #1 runtime check
    if not cfg.ablation.asymmetric_evidence and advocate is not None and skeptic is not None:
        assert advocate.slate_hash == skeptic.slate_hash, (
            f"SSES Invariant violated: advocate.slate_hash={advocate.slate_hash} "
            f"!= skeptic.slate_hash={skeptic.slate_hash}"
        )

    gate_mode = cfg.ablation.gate_mode
    s_adv = advocate.advocate_score if advocate is not None else 0.5
    s_skep = skeptic.skeptic_score if skeptic is not None else 0.0

    # Ablation A4 score projection
    if gate_mode == "advocate_only":
        s_skep = 0.0
    elif gate_mode == "skeptic_only":
        s_adv = max(0.0, min(1.0, 1.0 - s_skep))

    selected_skill = _select_candidate_skill(
        advocate=advocate,
        skill_priors=slate.skill_priors,
        prior_floor=cfg.gate.skill_prior_override_floor,
        disable_skill_routing=cfg.ablation.disable_skill_routing,
    )
    skill_card = None if cfg.ablation.disable_skill_routing else skill_store.get(selected_skill)
    delta_k = skill_card.margin_offset if skill_card is not None else 0.0
    gamma_k = skill_card.deescalation_bias if skill_card is not None else 0.0

    # Differential margin & epistemic disagreement
    if gate_mode == "advocate_only":
        margin_delta = s_adv - 0.5 - delta_k
    elif gate_mode == "skeptic_only":
        margin_delta = (1.0 - s_skep) - 0.5 - delta_k
    else:
        margin_delta = s_adv - cfg.gate.lambda_disruption * s_skep - delta_k

    disagreement_u = min(s_adv, s_skep)
    deescalation_reasons: list[str] = []

    # 1. Hard Need Floor
    if s_adv < cfg.gate.theta_floor:
        raw_tier = InterventionTier.SILENT
        final_tier = InterventionTier.SILENT
        deescalation_reasons.append("below_advocate_floor")
    else:
        # 2. Margin Ladder
        if margin_delta < cfg.gate.tau_dm:
            raw_tier = InterventionTier.SILENT
        elif margin_delta < cfg.gate.tau_thread:
            raw_tier = InterventionTier.DM_NUDGE
        elif margin_delta < cfg.gate.tau_channel:
            raw_tier = InterventionTier.THREAD_BRIEF
        else:
            raw_tier = InterventionTier.CHANNEL_INTERVENE

        raw_int = TIER_TO_INT[raw_tier]
        if raw_int == 0:
            final_tier = InterventionTier.SILENT
        elif cfg.ablation.binary_only:
            # Ablation A5: binary SILENT / CHANNEL_INTERVENE only
            final_tier = InterventionTier.CHANNEL_INTERVENE
            deescalation_reasons.append("ablation_a5_binary_only")
        else:
            deesc_steps = 0
            # Epistemic disagreement tension de-escalation (disabled in Ablation A6)
            if (
                not cfg.ablation.disable_disagreement_deesc
                and raw_int >= 2
                and disagreement_u >= cfg.gate.tau_disagreement
            ):
                deesc_steps += 1
                deescalation_reasons.append(
                    f"epistemic_disagreement_u={disagreement_u:.2f}>={cfg.gate.tau_disagreement:.2f}"
                )

            # Skeptic channel ceiling de-escalation
            if skeptic is not None and gate_mode != "advocate_only":
                skep_max_int = TIER_TO_INT[skeptic.max_acceptable_channel]
                if skep_max_int < raw_int and s_skep >= cfg.gate.skeptic_channel_cap_threshold:
                    deesc_steps += 1
                    deescalation_reasons.append(
                        f"skeptic_cap={skeptic.max_acceptable_channel.value}"
                    )

            # Online learned skill de-escalation bias
            gamma_steps = int(round(gamma_k))
            if gamma_steps > 0:
                deesc_steps += gamma_steps
                deescalation_reasons.append(f"skill_deescalation_bias={gamma_k:.2f}")

            tier_int = max(1, raw_int - deesc_steps)

            # Dual-consensus channel ceiling: never broadcast louder than BOTH Advocate's minimum
            # effective channel AND Skeptic's maximum acceptable channel.
            if (
                advocate is not None
                and skeptic is not None
                and gate_mode not in ("advocate_only", "skeptic_only")
            ):
                adv_min_int = TIER_TO_INT[InterventionTier(advocate.minimum_effective_channel)]
                skep_max_int = TIER_TO_INT[skeptic.max_acceptable_channel]
                consensus_cap = max(1, max(adv_min_int, skep_max_int))
                if tier_int > consensus_cap:
                    tier_int = consensus_cap
                    deescalation_reasons.append(
                        f"dual_consensus_channel_cap={INT_TO_TIER[consensus_cap].value}"
                    )

            # Skill privacy/channel ceiling
            if skill_card is not None:
                max_policy_int = max(1, TIER_TO_INT[skill_card.channel_policy.max_tier])
                if tier_int > max_policy_int:
                    tier_int = max_policy_int
                    deescalation_reasons.append(
                        f"skill_privacy_ceiling={skill_card.channel_policy.max_tier.value}"
                    )

            final_tier = INT_TO_TIER[tier_int]

    return GateDecision(
        short_circuited=False,
        slate_hash=slate.slate_hash,
        selected_skill=selected_skill,
        advocate_score=round(s_adv, 6),
        skeptic_score=round(s_skep, 6),
        margin_delta=round(margin_delta, 6),
        disagreement_u=round(disagreement_u, 6),
        raw_tier=raw_tier,
        final_tier=final_tier,
        binary_speak=(final_tier != InterventionTier.SILENT),
        deescalation_reasons=deescalation_reasons,
        advocate=advocate,
        skeptic=skeptic,
    )


class DialecticGate:
    """Async Stage-2 evaluator running independent parallel Advocate & Skeptic calls."""

    def __init__(
        self,
        openai_client: OpenAIClient,
        gate_model: str,
        cfg: DialSkillConfig,
        seed: int = 42,
    ) -> None:
        self.openai_client = openai_client
        self.gate_model = gate_model
        self.cfg = cfg
        self.seed = seed

    async def _run_advocate(self, slate: SharedEvidenceSlate) -> tuple[AdvocateOutput, LLMCallResult]:
        res = await self.openai_client.complete_json(
            model=self.gate_model,
            messages=build_advocate_messages(slate.canonical_text, env_mode=slate.env_mode),
            json_schema=ADVOCATE_JSON_SCHEMA,
            schema_name="dial_skill_advocate",
            temperature=self.cfg.gate.temperature,
            seed=self.seed,
        )
        data = dict(res.parsed_json or {})
        out = AdvocateOutput(
            slate_hash=slate.slate_hash,
            advocate_score=float(data.get("advocate_score", 0.0)),
            unmitigated_loss=str(data.get("unmitigated_loss", "")),
            recommended_skill=data.get("recommended_skill", "none"),
            minimum_effective_channel=data.get("minimum_effective_channel", "THREAD_BRIEF"),
            cited_evidence_ids=list(data.get("cited_evidence_ids", [])),
        )
        return out, res

    async def _run_skeptic(self, slate: SharedEvidenceSlate) -> tuple[SkepticOutput, LLMCallResult]:
        res = await self.openai_client.complete_json(
            model=self.gate_model,
            messages=build_skeptic_messages(slate.canonical_text, env_mode=slate.env_mode),
            json_schema=SKEPTIC_JSON_SCHEMA,
            schema_name="dial_skill_skeptic",
            temperature=self.cfg.gate.temperature,
            seed=self.seed,
        )
        data = dict(res.parsed_json or {})
        out = SkepticOutput(
            slate_hash=slate.slate_hash,
            skeptic_score=float(data.get("skeptic_score", 0.0)),
            disruption_risk=str(data.get("disruption_risk", "")),
            matched_veto_conditions=list(data.get("matched_veto_conditions", [])),
            max_acceptable_channel=InterventionTier(data.get("max_acceptable_channel", "SILENT")),
            cited_evidence_ids=list(data.get("cited_evidence_ids", [])),
        )
        return out, res

    async def _run_single_combined(
        self, slate: SharedEvidenceSlate
    ) -> tuple[AdvocateOutput, SkepticOutput, LLMCallResult]:
        res = await self.openai_client.complete_json(
            model=self.gate_model,
            messages=build_single_combined_messages(slate.canonical_text, env_mode=slate.env_mode),
            json_schema=SINGLE_COMBINED_GATE_JSON_SCHEMA,
            schema_name="dial_skill_single_combined",
            temperature=self.cfg.gate.temperature,
            seed=self.seed,
        )
        data = dict(res.parsed_json or {})
        adv = AdvocateOutput(
            slate_hash=slate.slate_hash,
            advocate_score=float(data.get("advocate_score", 0.0)),
            unmitigated_loss=str(data.get("unmitigated_loss", "")),
            recommended_skill=data.get("recommended_skill", "none"),
            minimum_effective_channel=data.get("minimum_effective_channel", "THREAD_BRIEF"),
            cited_evidence_ids=list(data.get("cited_evidence_ids", [])),
        )
        skep = SkepticOutput(
            slate_hash=slate.slate_hash,
            skeptic_score=float(data.get("skeptic_score", 0.0)),
            disruption_risk=str(data.get("disruption_risk", "")),
            matched_veto_conditions=list(data.get("matched_veto_conditions", [])),
            max_acceptable_channel=InterventionTier(data.get("max_acceptable_channel", "SILENT")),
            cited_evidence_ids=list(data.get("cited_evidence_ids", [])),
        )
        return adv, skep, res

    async def evaluate(
        self,
        *,
        advocate_slate: SharedEvidenceSlate,
        skeptic_slate: SharedEvidenceSlate,
        skill_store: SkillStore,
    ) -> tuple[GateDecision, float, float]:
        """Run Stage-2 gating; returns (GateDecision, cost_usd, model_elapsed_sec)."""
        # Enforce Non-Negotiable Invariant #1: Symmetric Shared Evidence Slate
        if not self.cfg.ablation.asymmetric_evidence:
            assert advocate_slate.slate_hash == skeptic_slate.slate_hash, (
                f"Invariant #1 violated: Advocate slate_hash ({advocate_slate.slate_hash}) != "
                f"Skeptic slate_hash ({skeptic_slate.slate_hash}). Enable cfg.ablation.asymmetric_evidence "
                "explicitly for Ablation A3."
            )
            assert advocate_slate.canonical_text == skeptic_slate.canonical_text, (
                "Invariant #1 violated: Advocate and Skeptic received different canonical_text."
            )

        mode = self.cfg.ablation.gate_mode
        cost = 0.0
        elapsed = 0.0
        adv_out: AdvocateOutput | None = None
        skep_out: SkepticOutput | None = None

        if mode == "dual_margin":
            # Execute Advocate and Skeptic independently in parallel
            (adv_out, adv_res), (skep_out, skep_res) = await asyncio.gather(
                self._run_advocate(advocate_slate),
                self._run_skeptic(skeptic_slate),
            )
            cost = adv_res.cost_usd + skep_res.cost_usd
            elapsed = max(adv_res.elapsed_sec, skep_res.elapsed_sec)
        elif mode == "advocate_only":
            adv_out, adv_res = await self._run_advocate(advocate_slate)
            cost = adv_res.cost_usd
            elapsed = adv_res.elapsed_sec
        elif mode == "skeptic_only":
            skep_out, skep_res = await self._run_skeptic(skeptic_slate)
            cost = skep_res.cost_usd
            elapsed = skep_res.elapsed_sec
        elif mode == "single_combined":
            adv_out, skep_out, comb_res = await self._run_single_combined(advocate_slate)
            cost = comb_res.cost_usd
            elapsed = comb_res.elapsed_sec
        else:
            raise ValueError(f"Unsupported gate_mode: {mode}")

        decision = compute_gate_decision(
            slate=advocate_slate,
            advocate=adv_out,
            skeptic=skep_out,
            skill_store=skill_store,
            cfg=self.cfg,
        )
        return decision, cost, elapsed
