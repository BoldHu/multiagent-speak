"""Online Dual-Boundary Counterfactual Skill Evolution (DB-CSE) with EscrowGate & O(1) Caps (Invariants #3 & #4)."""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from dial_skill.config import DialSkillConfig, load_dial_skill_config
from dial_skill.core.skill_store import (
    MAX_INVESTIGATION_STEPS,
    MAX_TRIGGER_CONDITIONS,
    MAX_VETO_CONDITIONS,
    SkillStore,
)
from dial_skill.llm.openai_client import OpenAIClient
from dial_skill.prompts.templates import (
    ATTRIBUTION_CRITIC_JSON_SCHEMA,
    ATTRIBUTION_CRITIC_SYSTEM_PROMPT,
    CONSOLIDATE_RULES_JSON_SCHEMA,
    CONSOLIDATE_RULES_SYSTEM_PROMPT,
    PATCH_PROPOSER_JSON_SCHEMA,
    PATCH_PROPOSER_SYSTEM_PROMPT,
)
from dial_skill.schemas import (
    ACTIVE_SKILLS,
    TIER_TO_INT,
    ActiveSkillName,
    AnchorEpisode,
    AttributionResult,
    EpisodeFeedback,
    EvolutionLogEntry,
    GateDecision,
    InterventionTier,
    PatchProposal,
    SharedEvidenceSlate,
    SkillCard,
)
from dial_skill.utils.checkpoint import JsonlCheckpointWriter

logger = logging.getLogger(__name__)

# Allowed enterprise service & domain capitalized tokens that are NOT entity-specific overfitting
_ALLOWED_PROPER_TOKENS = {
    "AI",
    "API",
    "CRM",
    "DM",
    "DM_NUDGE",
    "THREAD_BRIEF",
    "CHANNEL_INTERVENE",
    "SILENT",
    "PII",
    "SLA",
    "SSO",
    "SQL",
    "VIP",
    "OKR",
    "KPI",
    "CEO",
    "CTO",
    "CFO",
    "PM",
    "QA",
    "HR",
    "IT",
    "UI",
    "UX",
    "URL",
    "ID",
    "IDs",
    "Slack",
    "Notion",
    "Linear",
    "HubSpot",
    "Stripe",
    "PostHog",
    "Google",
    "Analytics",
    "Search",
    "Console",
    "Calendar",
    "BigQuery",
    "Snowflake",
    "English",
}

_TICKET_OR_ID_RE = re.compile(r"\b[A-Z]{2,6}-\d{1,6}\b|#\d{2,}\b")
_CHANNEL_RE = re.compile(r"(?:^|\s)#[a-z0-9][a-z0-9_-]{1,}\b")
_HANDLE_RE = re.compile(r"(?:^|\s)@[a-zA-Z0-9._-]{2,}\b")
_DATE_RE = re.compile(
    r"\b(?:20\d{2}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\b|Q[1-4]\s+20\d{2})\b",
    re.IGNORECASE,
)
_EMAIL_OR_URL_RE = re.compile(r"https?://|\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_DOLLAR_AMOUNT_RE = re.compile(r"\$\d[\d,]*(?:\.\d+)?")

RescoreFn = Callable[[SharedEvidenceSlate, SkillStore], Awaitable[GateDecision]]


def validate_generalized_predicate(
    predicate: str,
    existing_rules: list[str],
    max_words: int = 20,
) -> tuple[bool, str | None]:
    """Reject empty, overly long, duplicate, or entity-memorizing predicates."""
    text = " ".join(predicate.strip().split())
    if not text:
        return False, "empty_predicate"

    words = text.split()
    if len(words) > max_words:
        return False, f"too_long:{len(words)}_words>{max_words}"

    if _TICKET_OR_ID_RE.search(text):
        return False, "contains_specific_ticket_or_record_id"
    if _CHANNEL_RE.search(text):
        return False, "contains_specific_channel_name"
    if _HANDLE_RE.search(text):
        return False, "contains_specific_user_handle"
    if _DATE_RE.search(text):
        return False, "contains_specific_calendar_date"
    if _EMAIL_OR_URL_RE.search(text):
        return False, "contains_email_or_url"
    if _DOLLAR_AMOUNT_RE.search(text):
        return False, "contains_specific_dollar_amount"

    # Check mid-sentence capitalized proper nouns not in allowed enterprise tool/domain vocabulary
    for raw_tok in words[1:]:
        clean = re.sub(r"^[^A-Za-z0-9_]+|[^A-Za-z0-9_]+$", "", raw_tok)
        if not clean:
            continue
        if clean[0].isupper() and clean not in _ALLOWED_PROPER_TOKENS:
            return False, f"contains_specific_proper_noun:{clean}"

    # Check duplicate or near-duplicate against existing rules
    norm_new = set(re.findall(r"[a-z0-9]+", text.lower()))
    for existing in existing_rules:
        if existing.strip().lower() == text.lower():
            return False, "exact_duplicate"
        norm_old = set(re.findall(r"[a-z0-9]+", existing.lower()))
        if norm_new and norm_old:
            jaccard = len(norm_new & norm_old) / len(norm_new | norm_old)
            if jaccard >= 0.80:
                return False, f"near_duplicate_jaccard={jaccard:.2f}"

    return True, None


class AnchorBuffer:
    """Per-skill rolling memory of verified TP/TN anchor episodes (max size 8, pos/neg balanced)."""

    def __init__(self, max_per_skill: int = 8) -> None:
        self.max_per_skill = max_per_skill
        self._buffers: dict[str, list[AnchorEpisode]] = {}

    def add(self, anchor: AnchorEpisode) -> None:
        buf = self._buffers.setdefault(anchor.skill_name, [])
        # Replace existing sample_id if already present
        buf[:] = [a for a in buf if a.sample_id != anchor.sample_id]
        buf.append(anchor)
        if len(buf) > self.max_per_skill:
            # Maintain diversity: drop oldest from majority polarity if possible
            pos = [a for a in buf if a.expected_speak]
            neg = [a for a in buf if not a.expected_speak]
            if len(pos) > len(neg) and len(pos) > 1:
                oldest_pos = pos[0].sample_id
                buf[:] = [a for a in buf if a.sample_id != oldest_pos]
            elif len(neg) > len(pos) and len(neg) > 1:
                oldest_neg = neg[0].sample_id
                buf[:] = [a for a in buf if a.sample_id != oldest_neg]
            else:
                buf.pop(0)

    def get_all(self, skill_name: str) -> list[AnchorEpisode]:
        return list(self._buffers.get(skill_name, []))

    def sample(self, skill_name: str, m: int = 3, seed: int = 42) -> list[AnchorEpisode]:
        """Sample up to M diverse anchors (preferring both positive and negative when available)."""
        buf = self._buffers.get(skill_name, [])
        if len(buf) <= m:
            return list(buf)
        rng = random.Random(seed)
        pos = [a for a in buf if a.expected_speak]
        neg = [a for a in buf if not a.expected_speak]
        chosen: list[AnchorEpisode] = []
        if pos and neg:
            chosen.append(rng.choice(pos))
            chosen.append(rng.choice(neg))
        remaining = [a for a in buf if a.sample_id not in {c.sample_id for c in chosen}]
        rng.shuffle(remaining)
        for item in remaining:
            if len(chosen) >= m:
                break
            chosen.append(item)
        return chosen[:m]


@dataclass
class EscrowResult:
    passed: bool
    fixes_failure: bool
    anchor_regressions: int
    anchors_checked: int
    failed_old_margin: float
    failed_new_margin: float
    failed_old_tier: str
    failed_new_tier: str
    regressed_anchor_ids: list[str] = field(default_factory=list)


class EscrowGate:
    """Counterfactual Replay Escrow Gate verifying failure fix AND 0 anchor regressions (Invariant #4)."""

    def __init__(self, cfg: DialSkillConfig) -> None:
        self.cfg = cfg

    async def verify(
        self,
        *,
        rescore_fn: RescoreFn,
        candidate_store: SkillStore,
        failed_slate: SharedEvidenceSlate,
        failed_old_decision: GateDecision,
        feedback: EpisodeFeedback,
        anchors: list[AnchorEpisode],
    ) -> EscrowResult:
        new_failed = await rescore_fn(failed_slate, candidate_store)
        min_shift = self.cfg.evolution.escrow_min_delta_shift

        # 1. Check whether candidate_store fixes the failed episode
        if feedback.feedback_type == "FP_INTRUSION":
            fixes_failure = (not new_failed.binary_speak) or (
                failed_old_decision.margin_delta - new_failed.margin_delta >= min_shift
            )
        elif feedback.feedback_type == "FN_OMISSION":
            fixes_failure = new_failed.binary_speak or (
                new_failed.margin_delta - failed_old_decision.margin_delta >= min_shift
            )
        elif feedback.feedback_type == "FP_SCOPE_OVERREACH":
            fixes_failure = (
                TIER_TO_INT[new_failed.final_tier] < TIER_TO_INT[failed_old_decision.final_tier]
                and new_failed.binary_speak
            ) or (failed_old_decision.margin_delta - new_failed.margin_delta >= min_shift)
        elif feedback.feedback_type == "CONTENT_EVIDENCE_GAP":
            # Investigation plan patch: timing should remain valid SPEAK without breaking gating
            fixes_failure = new_failed.binary_speak
        else:
            fixes_failure = True

        # 2. Replay sampled anchor episodes — require STRICTLY ZERO regressions!
        regressed_ids: list[str] = []
        for anchor in anchors:
            anchor_decision = await rescore_fn(anchor.slate, candidate_store)
            if anchor_decision.binary_speak != anchor.expected_speak:
                regressed_ids.append(anchor.sample_id)

        passed = fixes_failure and len(regressed_ids) == 0
        return EscrowResult(
            passed=passed,
            fixes_failure=fixes_failure,
            anchor_regressions=len(regressed_ids),
            anchors_checked=len(anchors),
            failed_old_margin=failed_old_decision.margin_delta,
            failed_new_margin=new_failed.margin_delta,
            failed_old_tier=failed_old_decision.final_tier.value,
            failed_new_tier=new_failed.final_tier.value,
            regressed_anchor_ids=regressed_ids,
        )


def _clip(value: float, lo: float, hi: float) -> float:
    return round(max(lo, min(hi, value)), 6)


class DBCSEUpdater:
    """Online Dual-Boundary Counterfactual Skill Evolution engine (zero weight updates)."""

    def __init__(
        self,
        *,
        skill_store: SkillStore,
        openai_client: OpenAIClient,
        critic_model: str,
        cfg: DialSkillConfig | None = None,
        log_path: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        self.skill_store = skill_store
        self.openai_client = openai_client
        self.critic_model = critic_model
        self.cfg = cfg or load_dial_skill_config()
        self.seed = seed
        self.anchor_buffer = AnchorBuffer(max_per_skill=self.cfg.evolution.anchor_buffer_max)
        self.escrow_gate = EscrowGate(self.cfg)
        self._log_writer = JsonlCheckpointWriter(log_path) if log_path is not None else None

        # Telemetry for Exp 3 reporting
        self.stats: dict[str, int] = {
            "episodes_processed": 0,
            "numeric_updates": 0,
            "patches_proposed": 0,
            "patches_rejected_guardrail": 0,
            "patches_blocked_escrow": 0,
            "patches_promoted": 0,
            "pruning_events": 0,
        }

    def apply_numeric_update(
        self,
        skill_name: ActiveSkillName,
        feedback_type: str,
    ) -> dict[str, float]:
        """Apply bounded scalar calibration updates to (delta_k, gamma_k)."""
        card = self.skill_store.require(skill_name)
        evo = self.cfg.evolution
        old_delta = card.margin_offset
        old_gamma = card.deescalation_bias
        new_delta = old_delta
        new_gamma = old_gamma

        if feedback_type == "FP_INTRUSION":
            new_delta = _clip(old_delta + evo.eta_fp, -evo.delta_clip, evo.delta_clip)
        elif feedback_type == "FN_OMISSION":
            new_delta = _clip(old_delta - evo.eta_fn, -evo.delta_clip, evo.delta_clip)
        elif feedback_type == "FP_SCOPE_OVERREACH":
            new_gamma = _clip(old_gamma + evo.eta_scope, 0.0, evo.gamma_clip)
        elif feedback_type == "TP_VALIDATED":
            new_delta = _clip(old_delta * evo.tp_decay, -evo.delta_clip, evo.delta_clip)

        updated_card = card.model_copy(
            update={"margin_offset": new_delta, "deescalation_bias": new_gamma}
        )
        self.skill_store.set_skill(updated_card, bump_version=False)
        self.stats["numeric_updates"] += 1
        return {
            "delta_before": old_delta,
            "delta_after": new_delta,
            "gamma_before": old_gamma,
            "gamma_after": new_gamma,
        }

    async def diagnose_attribution(
        self,
        *,
        slate: SharedEvidenceSlate,
        gate: GateDecision,
        feedback: EpisodeFeedback,
        skill_card: SkillCard,
    ) -> AttributionResult:
        user_prompt = (
            f"Episode Feedback Type: {feedback.feedback_type}\n"
            f"Feedback Critique: {feedback.critique}\n"
            f"Expected Tier: {feedback.expected_tier.value if feedback.expected_tier else 'unknown'}\n"
            f"Predicted Tier: {gate.final_tier.value} (margin={gate.margin_delta:.3f}, "
            f"adv={gate.advocate_score:.3f}, skep={gate.skeptic_score:.3f})\n\n"
            f"Current Skill Card `{skill_card.name}`:\n"
            f"- Triggers: {skill_card.trigger_conditions}\n"
            f"- Vetos: {skill_card.veto_conditions}\n"
            f"- Investigation Plan: {skill_card.investigation_plan}\n\n"
            f"Shared Evidence Slate:\n{slate.canonical_text}"
        )
        res = await self.openai_client.complete_json(
            model=self.critic_model,
            messages=[
                {"role": "system", "content": ATTRIBUTION_CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            json_schema=ATTRIBUTION_CRITIC_JSON_SCHEMA,
            schema_name="dial_skill_attribution_critic",
            temperature=0.0,
            seed=self.seed,
        )
        data = dict(res.parsed_json or {})
        affected = data.get("affected_skill", skill_card.name)
        if affected not in ACTIVE_SKILLS:
            affected = skill_card.name
        return AttributionResult(
            root_cause=data.get("root_cause", "CALIBRATION_SHIFT"),
            affected_skill=affected,
            counterfactual_explanation=str(data.get("counterfactual_explanation", "")),
            needs_symbolic_patch=bool(data.get("needs_symbolic_patch", False)),
        )

    async def propose_symbolic_patch(
        self,
        *,
        skill_card: SkillCard,
        target_field: Literal["trigger_conditions", "veto_conditions", "investigation_plan"],
        attribution: AttributionResult,
        feedback: EpisodeFeedback,
    ) -> PatchProposal:
        sys_prompt = PATCH_PROPOSER_SYSTEM_PROMPT.format(
            skill_name=skill_card.name,
            target_field=target_field,
        )
        existing = getattr(skill_card, target_field)
        user_prompt = (
            f"Diagnosed Root Cause: {attribution.root_cause}\n"
            f"Counterfactual Explanation: {attribution.counterfactual_explanation}\n"
            f"Feedback Critique: {feedback.critique}\n"
            f"Existing `{target_field}`: {existing}\n"
            "Propose ONE atomic, generalized rule (<= 20 words, zero proper nouns/ticket IDs)."
        )
        res = await self.openai_client.complete_json(
            model=self.critic_model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            json_schema=PATCH_PROPOSER_JSON_SCHEMA,
            schema_name="dial_skill_patch_proposer",
            temperature=0.0,
            seed=self.seed,
        )
        data = dict(res.parsed_json or {})
        return PatchProposal(
            target_field=target_field,
            proposed_predicate=str(data.get("proposed_predicate", "")).strip(),
            rationale=str(data.get("rationale", "")).strip(),
        )

    async def _consolidate_or_prune(
        self,
        *,
        skill_card: SkillCard,
        target_field: Literal["trigger_conditions", "veto_conditions", "investigation_plan"],
        candidate_rules: list[str],
        max_rules: int,
    ) -> tuple[list[str], str]:
        """Enforce hard O(1) capacity cap via LLM merge/prune with deterministic utility fallback."""
        if len(candidate_rules) <= max_rules:
            return candidate_rules, "none"

        scored_lines: list[str] = []
        for idx, rule in enumerate(candidate_rules):
            st = skill_card.rule_stats.get(rule, {"helpful_hits": 0, "regressions": 0})
            util = int(st.get("helpful_hits", 0)) - 2 * int(st.get("regressions", 0))
            scored_lines.append(
                f"{idx + 1}. {rule} (helpful_hits={st.get('helpful_hits', 0)}, "
                f"regressions={st.get('regressions', 0)}, utility={util})"
            )

        sys_prompt = CONSOLIDATE_RULES_SYSTEM_PROMPT.format(
            skill_name=skill_card.name,
            max_rules=max_rules,
            target_field=target_field,
            candidate_count=len(candidate_rules),
        )
        try:
            res = await self.openai_client.complete_json(
                model=self.critic_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": "\n".join(scored_lines)},
                ],
                json_schema=CONSOLIDATE_RULES_JSON_SCHEMA,
                schema_name="dial_skill_consolidate_rules",
                temperature=0.0,
                seed=self.seed,
            )
            data = dict(res.parsed_json or {})
            consolidated = [str(r).strip() for r in data.get("consolidated_rules", []) if str(r).strip()]
            action = str(data.get("action_taken", "merged"))
            if len(consolidated) == max_rules:
                self.stats["pruning_events"] += 1
                return consolidated, action
        except Exception as exc:
            logger.warning("Consolidation LLM failed (%s); falling back to utility pruning.", exc)

        # Deterministic utility-based pruning fallback (keeps newest proposed rule + highest-utility existing)
        newest = candidate_rules[-1]
        older = candidate_rules[:-1]

        def _utility(rule: str) -> tuple[int, int]:
            st = skill_card.rule_stats.get(rule, {"helpful_hits": 0, "regressions": 0})
            return (int(st.get("helpful_hits", 0)) - 2 * int(st.get("regressions", 0)), older.index(rule))

        kept_older = sorted(older, key=_utility, reverse=True)[: max_rules - 1]
        kept_older_ordered = [r for r in older if r in kept_older]
        self.stats["pruning_events"] += 1
        return [*kept_older_ordered, newest], "pruned_utility_fallback"

    @staticmethod
    def _target_field_for_root_cause(
        root_cause: str, feedback_type: str
    ) -> Literal["trigger_conditions", "veto_conditions", "investigation_plan"]:
        if root_cause == "VETO_MISSING":
            return "veto_conditions"
        if root_cause == "TRIGGER_MISSING":
            return "trigger_conditions"
        if root_cause == "INVESTIGATION_GAP":
            return "investigation_plan"
        # ROUTING_CONFUSION: disambiguate via veto on FP or trigger on FN
        if feedback_type in ("FP_INTRUSION", "FP_SCOPE_OVERREACH"):
            return "veto_conditions"
        return "trigger_conditions"

    async def process_episode(
        self,
        *,
        slate: SharedEvidenceSlate,
        gate: GateDecision,
        feedback: EpisodeFeedback,
        rescore_fn: RescoreFn,
    ) -> EvolutionLogEntry:
        """Process one completed episode: anchor maintenance, numeric update, attribution, patch, EscrowGate."""
        self.stats["episodes_processed"] += 1
        version_before = self.skill_store.version

        skill_name: ActiveSkillName = (
            feedback.skill_name if feedback.skill_name in ACTIVE_SKILLS else "find"
        )  # type: ignore[assignment]

        # 1. Validated TP / TN -> record into Anchor Buffer (+ mild TP decay) and skip critic
        if feedback.feedback_type in ("TP_VALIDATED", "TN_VALIDATED"):
            anchor = AnchorEpisode(
                sample_id=feedback.sample_id,
                skill_name=skill_name,
                expected_speak=(feedback.feedback_type == "TP_VALIDATED"),
                expected_tier=feedback.expected_tier
                or (InterventionTier.THREAD_BRIEF if feedback.feedback_type == "TP_VALIDATED" else InterventionTier.SILENT),
                slate=slate,
                original_margin=gate.margin_delta,
            )
            self.anchor_buffer.add(anchor)

            # Update helpful rule utility stats when Skeptic cited a veto on a verified TN
            if feedback.feedback_type == "TN_VALIDATED" and gate.skeptic:
                card = self.skill_store.require(skill_name)
                stats_copy = {k: dict(v) for k, v in card.rule_stats.items()}
                for veto in gate.skeptic.matched_veto_conditions:
                    entry = stats_copy.setdefault(veto, {"helpful_hits": 0, "regressions": 0})
                    entry["helpful_hits"] = int(entry.get("helpful_hits", 0)) + 1
                self.skill_store.set_skill(card.model_copy(update={"rule_stats": stats_copy}))

            numeric_upd = (
                self.apply_numeric_update(skill_name, "TP_VALIDATED")
                if feedback.feedback_type == "TP_VALIDATED"
                else {}
            )
            log_entry = EvolutionLogEntry(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                version_before=version_before,
                version_after=self.skill_store.version,
                sample_id=feedback.sample_id,
                skill_name=skill_name,
                feedback_type=feedback.feedback_type,
                numeric_update=numeric_upd,
                promoted=False,
            )
            self._write_log(log_entry)
            return log_entry

        # 2. Counterfactual Attribution Critic FIRST so numeric & symbolic updates target the exact affected_skill
        current_card = self.skill_store.require(skill_name)
        attribution = await self.diagnose_attribution(
            slate=slate,
            gate=gate,
            feedback=feedback,
            skill_card=current_card,
        )
        affected_skill = attribution.affected_skill

        # 3. Immediate bounded numeric calibration applied strictly to `affected_skill`
        numeric_upd = self.apply_numeric_update(affected_skill, feedback.feedback_type)
        target_card = self.skill_store.require(affected_skill)

        symbolic_causes = {"VETO_MISSING", "TRIGGER_MISSING", "INVESTIGATION_GAP", "ROUTING_CONFUSION"}
        if attribution.root_cause not in symbolic_causes or not attribution.needs_symbolic_patch:
            log_entry = EvolutionLogEntry(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                version_before=version_before,
                version_after=self.skill_store.version,
                sample_id=feedback.sample_id,
                skill_name=affected_skill,
                feedback_type=feedback.feedback_type,
                root_cause=attribution.root_cause,
                numeric_update=numeric_upd,
                promoted=False,
                rejection_reason="numeric_or_scope_only_no_symbolic_patch",
            )
            self._write_log(log_entry)
            return log_entry

        # 4. Generalized Symbolic Patch Proposal + Anti-Overfitting Guardrail
        target_field = self._target_field_for_root_cause(attribution.root_cause, feedback.feedback_type)
        proposal = await self.propose_symbolic_patch(
            skill_card=target_card,
            target_field=target_field,
            attribution=attribution,
            feedback=feedback,
        )
        self.stats["patches_proposed"] += 1

        existing_rules: list[str] = list(getattr(target_card, target_field))
        valid, reject_reason = validate_generalized_predicate(proposal.proposed_predicate, existing_rules)
        if not valid:
            self.stats["patches_rejected_guardrail"] += 1
            log_entry = EvolutionLogEntry(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                version_before=version_before,
                version_after=self.skill_store.version,
                sample_id=feedback.sample_id,
                skill_name=affected_skill,
                feedback_type=feedback.feedback_type,
                root_cause=attribution.root_cause,
                numeric_update=numeric_upd,
                patch_proposed=proposal,
                promoted=False,
                rejection_reason=f"guardrail:{reject_reason}",
            )
            self._write_log(log_entry)
            return log_entry

        # 5. Construct candidate SkillStore enforcing hard O(1) cap via Consolidate-or-Prune
        cap = (
            min(self.cfg.evolution.max_investigation_steps, MAX_INVESTIGATION_STEPS)
            if target_field == "investigation_plan"
            else min(
                self.cfg.evolution.max_rules_per_side,
                MAX_TRIGGER_CONDITIONS if target_field == "trigger_conditions" else MAX_VETO_CONDITIONS,
            )
        )
        extended_rules = [*existing_rules, proposal.proposed_predicate]
        pruning_action: str | None = None
        if len(extended_rules) > cap:
            final_rules, pruning_action = await self._consolidate_or_prune(
                skill_card=target_card,
                target_field=target_field,
                candidate_rules=extended_rules,
                max_rules=cap,
            )
        else:
            final_rules = extended_rules

        candidate_store = self.skill_store.clone()
        candidate_card = target_card.model_copy(update={target_field: final_rules})
        candidate_store.set_skill(candidate_card, bump_version=True)

        # Ensure EscrowGate replays with the patched affected_skill visible in candidate_skills
        failed_slate_for_escrow = slate
        if not any(c.name == affected_skill for c in slate.candidate_skills):
            priors_copy = dict(slate.skill_priors)
            priors_copy[affected_skill] = max(
                self.cfg.gate.skill_prior_override_floor,
                float(priors_copy.get(affected_skill, 0.0)),
            )
            failed_slate_for_escrow = slate.model_copy(
                update={
                    "candidate_skills": [*slate.candidate_skills, candidate_card],
                    "skill_priors": priors_copy,
                }
            )

        # 6. Counterfactual Replay Escrow Gate (Invariant #4)
        escrow_passed = True
        escrow_details: dict[str, Any] = {"escrow_enabled": self.cfg.evolution.enable_escrow_gate}
        if self.cfg.evolution.enable_escrow_gate:
            anchors = self.anchor_buffer.sample(
                affected_skill,
                m=self.cfg.evolution.escrow_anchors_m,
                seed=self.seed + self.stats["episodes_processed"],
            )
            escrow_res = await self.escrow_gate.verify(
                rescore_fn=rescore_fn,
                candidate_store=candidate_store,
                failed_slate=failed_slate_for_escrow,
                failed_old_decision=gate,
                feedback=feedback,
                anchors=anchors,
            )
            escrow_passed = escrow_res.passed
            escrow_details.update(
                {
                    "fixes_failure": escrow_res.fixes_failure,
                    "anchor_regressions": escrow_res.anchor_regressions,
                    "anchors_checked": escrow_res.anchors_checked,
                    "failed_old_margin": escrow_res.failed_old_margin,
                    "failed_new_margin": escrow_res.failed_new_margin,
                    "regressed_anchor_ids": escrow_res.regressed_anchor_ids,
                }
            )

        if escrow_passed:
            self.skill_store.set_skill(candidate_card, bump_version=True)
            self.stats["patches_promoted"] += 1
            rejection = None
        else:
            self.stats["patches_blocked_escrow"] += 1
            rejection = (
                "escrow_anchor_regression"
                if escrow_details.get("anchor_regressions", 0) > 0
                else "escrow_did_not_fix_failure"
            )

        log_entry = EvolutionLogEntry(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            version_before=version_before,
            version_after=self.skill_store.version,
            sample_id=feedback.sample_id,
            skill_name=affected_skill,
            feedback_type=feedback.feedback_type,
            root_cause=attribution.root_cause,
            numeric_update=numeric_upd,
            patch_proposed=proposal,
            escrow_passed=escrow_passed,
            escrow_details=escrow_details,
            promoted=escrow_passed,
            pruning_action=pruning_action,
            rejection_reason=rejection,
        )
        self._write_log(log_entry)
        return log_entry

    def rollback_from_snapshot(self, snapshot_path: str | Path) -> SkillStore:
        """Restore SkillStore cleanly from a previously saved JSON version snapshot."""
        restored = SkillStore.from_json_file(snapshot_path)
        self.skill_store.skills = restored.skills
        self.skill_store.version = restored.version
        return self.skill_store

    def _write_log(self, entry: EvolutionLogEntry) -> None:
        if self._log_writer is not None:
            self._log_writer.append(entry)
