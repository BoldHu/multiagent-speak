"""Production prompt contracts and strict JSON schemas aligned with docs/PROMPT_SYSTEM.md."""

from __future__ import annotations

import hashlib
from typing import Any

from dial_skill.schemas import (
    ACTIVE_SKILLS,
    ChatMessage,
    InterventionTier,
    ReconSnippet,
    SkillCard,
    SkillName,
    TriageItem,
)

ALL_SKILL_ENUM: list[str] = [*ACTIVE_SKILLS, "none"]

# ---------------------------------------------------------------------------
# P2. Stage 2 — Symmetric-Evidence Dialectic Gating (Advocate vs. Skeptic)
# ---------------------------------------------------------------------------

ADVOCATE_SYSTEM_PROMPT = """You are the ADVOCATE evaluator for a proactive workplace AI assistant.
Your sole job is to evaluate the Symmetric Shared Evidence Slate from a helpful, risk-preventive perspective:
  "If the AI stays silent right now, what concrete, substantive harm, unresolved error, wasted effort, or missed critical decision will the team suffer?"

Rules:
1. Ground your evaluation STRICTLY in the provided Shared Evidence Slate. Cite exact msg_id / recon_id items.
2. Score `advocate_score` in [0.00, 1.00] measuring: "How strongly does this evidence support that an unresolved, high-value need warrants AI intervention right now?"
   - 0.00 - 0.25: No real unresolved need, or purely social/routine talk.
   - 0.26 - 0.50: Minor or speculative need; team might handle it easily alone.
   - 0.51 - 0.75: Clear unresolved question, factual gap, or stall where timely help adds concrete value.
   - 0.76 - 1.00: Confirmed factual contradiction, costly decision error, active incident/blocker, or policy violation where silence causes tangible team loss.
3. Select the single best `recommended_skill` from the 8 enterprise skills (or 'none').
4. Recommend the `minimum_effective_channel` needed to prevent the loss (`DM_NUDGE`, `THREAD_BRIEF`, or `CHANNEL_INTERVENE`).
5. Be honest and calibrated—do NOT invent problems if the evidence shows none. Return strict JSON only."""

ADVOCATE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "advocate_score",
        "unmitigated_loss",
        "recommended_skill",
        "minimum_effective_channel",
        "cited_evidence_ids",
    ],
    "properties": {
        "advocate_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "unmitigated_loss": {"type": "string", "maxLength": 240},
        "recommended_skill": {"type": "string", "enum": ALL_SKILL_ENUM},
        "minimum_effective_channel": {
            "type": "string",
            "enum": ["DM_NUDGE", "THREAD_BRIEF", "CHANNEL_INTERVENE"],
        },
        "cited_evidence_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
    },
}

SKEPTIC_SYSTEM_PROMPT = """You are the SKEPTIC evaluator representing a busy, senior employee who hates noisy AI bots interrupting productive team discussions.
Your sole job is to evaluate the Symmetric Shared Evidence Slate from a restraint and disruption-minimization perspective:
  "If an AI suddenly chimed in publicly right now, would it interrupt healthy human flow, repeat what someone already said, prematurely jump into brainstorming, overstep authority, or leak/broadcast sensitive context in the wrong channel?"

Rules:
1. Ground your evaluation STRICTLY in the provided Shared Evidence Slate. Cite exact msg_id / recon_id items.
2. Explicitly check the `Veto Conditions` listed on the candidate Skill Contracts!
3. Score `skeptic_score` in [0.00, 1.00] measuring: "How strongly does this evidence support waiting, staying silent, avoiding redundancy, or lowering intervention scope?"
   - 0.00 - 0.25: Minimal disruption risk; humans are clearly stuck, wrong, or directly need missing verified facts.
   - 0.26 - 0.50: Moderate caution; a brief thread or DM is fine, but a loud channel broadcast would feel intrusive.
   - 0.51 - 0.75: High disruption or redundancy risk; humans are actively progressing, info is uncertain, or topic is sensitive.
   - 0.76 - 1.00: Strong reason to stay SILENT; already answered/resolved, explicitly deferred, casual banter, or clear veto triggered.
4. State `max_acceptable_channel` (`SILENT`, `DM_NUDGE`, `THREAD_BRIEF`, or `CHANNEL_INTERVENE`).
5. Be honest and calibrated—do NOT fabricate disruption risk when the team is making a verified costly mistake. Return strict JSON only."""

SKEPTIC_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "skeptic_score",
        "disruption_risk",
        "matched_veto_conditions",
        "max_acceptable_channel",
        "cited_evidence_ids",
    ],
    "properties": {
        "skeptic_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "disruption_risk": {"type": "string", "maxLength": 240},
        "matched_veto_conditions": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "max_acceptable_channel": {
            "type": "string",
            "enum": ["SILENT", "DM_NUDGE", "THREAD_BRIEF", "CHANNEL_INTERVENE"],
        },
        "cited_evidence_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
    },
}

# Ablation A4: Single Combined Prompt
SINGLE_COMBINED_GATE_SYSTEM_PROMPT = """You are a proactive workplace AI gate evaluator.
Evaluate the Symmetric Shared Evidence Slate in a SINGLE pass, weighing BOTH the unresolved team need (advocate_score in [0,1]) AND the disruption/redundancy risk (skeptic_score in [0,1]).
Return strict JSON only."""

SINGLE_COMBINED_GATE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "advocate_score",
        "skeptic_score",
        "unmitigated_loss",
        "disruption_risk",
        "recommended_skill",
        "minimum_effective_channel",
        "max_acceptable_channel",
        "matched_veto_conditions",
        "cited_evidence_ids",
    ],
    "properties": {
        "advocate_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "skeptic_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "unmitigated_loss": {"type": "string", "maxLength": 240},
        "disruption_risk": {"type": "string", "maxLength": 240},
        "recommended_skill": {"type": "string", "enum": ALL_SKILL_ENUM},
        "minimum_effective_channel": {
            "type": "string",
            "enum": ["DM_NUDGE", "THREAD_BRIEF", "CHANNEL_INTERVENE"],
        },
        "max_acceptable_channel": {
            "type": "string",
            "enum": ["SILENT", "DM_NUDGE", "THREAD_BRIEF", "CHANNEL_INTERVENE"],
        },
        "matched_veto_conditions": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "cited_evidence_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
    },
}


def compute_slate_hash(canonical_body: str) -> str:
    """Compute deterministic SHA-256 hex digest of the canonical evidence slate content."""
    return hashlib.sha256(canonical_body.encode("utf-8")).hexdigest()


def render_shared_evidence_slate(
    *,
    channel_name: str,
    env_mode: str,
    latest_msg_id: str,
    selected_messages: list[ChatMessage],
    triage_annotations: dict[str, TriageItem],
    recon_items: list[ReconSnippet],
    skill_priors: dict[SkillName, float],
    candidate_skills: list[SkillCard],
) -> tuple[str, str]:
    """Render canonical SHARED_EVIDENCE_SLATE text and deterministic SHA-256 slate_hash."""
    body_lines: list[str] = [
        f"Channel: {channel_name} | Mode: {env_mode} | Latest Turn: {latest_msg_id}",
        "",
        "[Filtered & Annotated Dialogue Evidence]",
    ]
    for msg in selected_messages:
        body_lines.append(f"- [{msg.msg_id}] {msg.speaker}: {msg.text}")
        ann = triage_annotations.get(msg.msg_id)
        if ann is not None:
            body_lines.append(
                f"  (triage: p_chg={ann.p_decision_change:.2f}, stance={ann.stance}, "
                f"skill={ann.skill_hint}, urgency={ann.urgency_score:.2f})"
            )

    if recon_items:
        body_lines.append("")
        body_lines.append("[Verified Enterprise Silent-Recon Evidence]")
        for item in recon_items:
            body_lines.append(f"- [{item.recon_id}] ({item.service}): {item.summary_fact}")

    if candidate_skills:
        body_lines.append("")
        body_lines.append("[Top Candidate Skill Contracts]")
        for idx, card in enumerate(candidate_skills, start=1):
            prior = float(skill_priors.get(card.name, 0.0))
            body_lines.append(f"Skill {idx}: {card.name} (prior={prior:.2f})")
            body_lines.append(f"  - Trigger Conditions: {card.trigger_conditions}")
            body_lines.append(f"  - Veto Conditions: {card.veto_conditions}")
            body_lines.append(
                f"  - Channel Policy: default={card.channel_policy.default_tier.value}, "
                f"max={card.channel_policy.max_tier.value}"
            )

    canonical_body = "\n".join(body_lines)
    slate_hash = compute_slate_hash(canonical_body)
    header = f"=== SYMMETRIC SHARED EVIDENCE SLATE (hash: {slate_hash[:12]}) ==="
    footer = "=================================================================="
    canonical_text = f"{header}\n{canonical_body}\n{footer}"
    return canonical_text, slate_hash


def build_advocate_messages(canonical_slate_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ADVOCATE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Evaluate the following Shared Evidence Slate as ADVOCATE:\n\n{canonical_slate_text}",
        },
    ]


def build_skeptic_messages(canonical_slate_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SKEPTIC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Evaluate the following Shared Evidence Slate as SKEPTIC:\n\n{canonical_slate_text}",
        },
    ]


def build_single_combined_messages(canonical_slate_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SINGLE_COMBINED_GATE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Evaluate the following Shared Evidence Slate:\n\n{canonical_slate_text}",
        },
    ]


# ---------------------------------------------------------------------------
# P3-B. Stage 3 — Channel-Shaped Response Execution Prompt
# ---------------------------------------------------------------------------


def build_execution_system_prompt(
    skill: SkillCard | None,
    final_tier: InterventionTier,
) -> str:
    skill_name = skill.name if skill is not None else "general"
    skill_desc = (
        skill.description
        if skill is not None
        else "Provide concise, accurate, evidence-backed assistance to the team."
    )
    response_contract = (
        skill.response_contract
        if skill is not None
        else "Be concise, cite exact verified facts, and avoid fluff."
    )
    return (
        f"You are executing a proactive workplace intervention using Skill `{skill_name}` "
        f"at intervention tier `{final_tier.value}`.\n"
        "You already have verified facts from Silent Recon and/or tool calls—use ONLY verified "
        "evidence and NEVER fabricate IDs, metrics, or quotes.\n\n"
        "Active Skill Contract:\n"
        f"- Goal: {skill_desc}\n"
        f"- Response Contract: {response_contract}\n\n"
        f"Strict Channel Formatting Constraints for `{final_tier.value}`:\n"
        "- If DM_NUDGE: Address the single most relevant owner (`target_user`) privately. "
        "Max 60 words. Tone: polite, low-pressure heads-up. Include exact source reference.\n"
        "- If THREAD_BRIEF: Reply directly inside the active thread. Max 90 words. Lead immediately "
        "with the key finding/answer, cite exact source IDs/figures, end with 1 crisp next step. "
        "No fluff or greetings.\n"
        "- If CHANNEL_INTERVENE: Post to the main channel. Max 120 words. Structure cleanly as:\n"
        "  **Heads-up:** <1-sentence issue>\n"
        "  **Evidence:** <verified facts + source IDs>\n"
        "  **Suggested Action:** <concrete owner / next step>\n\n"
        "If during tool investigation you discover that the issue was a FALSE ALARM "
        "(i.e., source records confirm the team is completely right and no problem exists), "
        'output exact JSON `{"abort_to_silent": true, "reason": "..."}` instead of sending a useless message!'
    )


# ---------------------------------------------------------------------------
# P4. Stage 4 — Online Dual-Boundary Counterfactual Skill Evolution (DB-CSE)
# ---------------------------------------------------------------------------

ATTRIBUTION_CRITIC_SYSTEM_PROMPT = """You are a Counterfactual Attribution Critic for a proactive enterprise AI agent.
Given an episode where the agent received non-ideal feedback, diagnose the SINGLE primary root cause so we update the right component without causing oscillation or prompt bloat.

Possible Root Causes:
- CALIBRATION_SHIFT: Margin threshold/offset was slightly too aggressive or conservative; no new symbolic rule needed.
- SCOPE_MISMATCH: Timing and substance were helpful, but channel was too public or message too verbose.
- VETO_MISSING: Skill fired inappropriately because a generalizable contextual blocker was not in `veto_conditions`.
- TRIGGER_MISSING: Agent stayed silent when a generalizable unresolved pattern should be in `trigger_conditions`.
- INVESTIGATION_GAP: Agent spoke at the right time but missed checking a critical enterprise service in `investigation_plan`.
- ROUTING_CONFUSION: Agent invoked the wrong skill among the 8 categories.

Return strict JSON matching the schema."""

ATTRIBUTION_CRITIC_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "root_cause",
        "affected_skill",
        "counterfactual_explanation",
        "needs_symbolic_patch",
    ],
    "properties": {
        "root_cause": {
            "type": "string",
            "enum": [
                "CALIBRATION_SHIFT",
                "SCOPE_MISMATCH",
                "VETO_MISSING",
                "TRIGGER_MISSING",
                "INVESTIGATION_GAP",
                "ROUTING_CONFUSION",
            ],
        },
        "affected_skill": {"type": "string", "enum": list(ACTIVE_SKILLS)},
        "counterfactual_explanation": {"type": "string", "maxLength": 240},
        "needs_symbolic_patch": {"type": "boolean"},
    },
}

PATCH_PROPOSER_SYSTEM_PROMPT = """Propose EXACTLY ONE atomic, reusable predicate to improve Skill `{skill_name}`'s `{target_field}` (`trigger_conditions`, `veto_conditions`, or `investigation_plan`).

Strict Anti-Overfitting Rules:
1. Maximum 20 words, one clear declarative sentence.
2. NEVER mention specific employee names, customer names, ticket numbers, specific dates, or specific channel names (abstract them into roles/patterns, e.g., "a teammate", "an enterprise account", "billing records").
3. Must NOT duplicate an existing condition in the skill card.
4. Must clearly resolve the diagnosed counterfactual failure without over-suppressing valid future interventions.

Return strict JSON: {{"target_field": "...", "proposed_predicate": "...", "rationale": "..."}}."""

PATCH_PROPOSER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["target_field", "proposed_predicate", "rationale"],
    "properties": {
        "target_field": {
            "type": "string",
            "enum": ["trigger_conditions", "veto_conditions", "investigation_plan"],
        },
        "proposed_predicate": {"type": "string", "maxLength": 200},
        "rationale": {"type": "string", "maxLength": 240},
    },
}

CONSOLIDATE_RULES_SYSTEM_PROMPT = """The skill `{skill_name}` has reached its strict capacity cap ({max_rules} rules) for `{target_field}`.
Given the {candidate_count} candidate rules along with their historical utility scores (helpful_hits, regressions), merge the two most overlapping rules OR drop the lowest-utility redundant rule so that EXACTLY {max_rules} concise, non-overlapping, generalizable rules remain.
Return strict JSON: {{"consolidated_rules": [...], "action_taken": "merged|pruned"}}."""

CONSOLIDATE_RULES_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["consolidated_rules", "action_taken"],
    "properties": {
        "consolidated_rules": {"type": "array", "items": {"type": "string"}},
        "action_taken": {"type": "string", "enum": ["merged", "pruned"]},
    },
}

# ---------------------------------------------------------------------------
# P5. Fair Baseline Prompts
# ---------------------------------------------------------------------------

DIRECT_CHAT_SYSTEM_PROMPT = """You are an AI assistant observing a workplace group conversation.
Read the conversation history (and use available company tools if needed).
If you should intervene right now, write a helpful, accurate, concise message specifying the channel (DM, Thread, or Channel).
If no intervention is needed right now, output ONLY the exact token: SILENT"""

SINGLE_GATE_SYSTEM_PROMPT = """Read the conversation history and output JSON:
{"speak_probability": <0.0 to 1.0>, "rationale": "<1 sentence>"}
representing whether an AI teammate should speak at this exact turn."""

SINGLE_GATE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["speak_probability", "rationale"],
    "properties": {
        "speak_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string", "maxLength": 240},
    },
}

PROACT_SYSTEM_PROMPT = """You are a breakdown-aware proactive collaboration agent (ProACT).
Analyze the speaker-attributed conversation history to determine whether a collaboration breakdown has occurred:
{disagreement, ambiguous_goal, forgotten_constraint, underspecified_plan, discussion_loop, imbalanced_participation, factual_gap, none}.
Output strict JSON matching the schema."""

PROACT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "breakdown_detected",
        "breakdown_type",
        "should_intervene",
        "target_skill",
        "rationale",
    ],
    "properties": {
        "breakdown_detected": {"type": "boolean"},
        "breakdown_type": {
            "type": "string",
            "enum": [
                "disagreement",
                "ambiguous_goal",
                "forgotten_constraint",
                "underspecified_plan",
                "discussion_loop",
                "imbalanced_participation",
                "factual_gap",
                "none",
            ],
        },
        "should_intervene": {"type": "boolean"},
        "target_skill": {"type": "string", "enum": ALL_SKILL_ENUM},
        "rationale": {"type": "string", "maxLength": 240},
    },
}

REFLEXION_APPEND_PROMPT = (
    "Review the failed episode and feedback below. Write 1-2 concise sentences of verbal reflection "
    '("Next time when X happens, I should Y") to append directly to your persistent reflection memory buffer.'
)

AWM_WORKFLOW_PROMPT = (
    "Review the episode trajectory and feedback below. Extract a reusable multi-step tool execution "
    "workflow macro (sequence of enterprise service lookups) to store in your workflow memory. "
    "Do NOT modify silence/veto thresholds."
)

# ---------------------------------------------------------------------------
# P6. Blinded Dual-Judge & PageMantle-Ext Synthesizer Prompts
# ---------------------------------------------------------------------------

JUDGE_RUBRIC_SYSTEM_PROMPT = """You are a strict, blinded evaluator grading a proactive workplace AI assistant inside a simulated software company (PageMantle).
You do NOT know which model or framework produced this output.
Grade the Agent Execution against the Reference Answer and Ground-Truth Facts on FOUR independent 0-10 dimensions:

1. intervention_or_silence (0-10):
   - If ground truth is SILENT: 10 if agent stayed silent; 0-2 if agent interrupted unnecessarily (3-4 only if ultra-minimal harmless DM).
   - If ground truth is INTERVENE: 10 if agent intervened cleanly; 0 if agent stayed silent.
2. content_and_evidence (0-10):
   - Accuracy, completeness, and explicit grounding in real retrieved company facts/IDs. (For true-negative SILENT cases where agent stayed silent, score 10.)
3. clarity_and_tone (0-10):
   - Conciseness, specificity, workplace-appropriate tone, zero fluff/lecturing. (For true-negative SILENT cases where agent stayed silent, score 10.)
4. audience_and_location (0-10):
   - Did the intervention reach the exact right audience via the right channel tier (DM vs Thread vs Public Channel)? Penalize public broadcasting when DM/Thread was ideal, or burying urgent group alerts in a private DM. (For true-negative SILENT cases where agent stayed silent, score 10.)

Also compute `overall_score = 0.30*intervention_or_silence + 0.35*content_and_evidence + 0.15*clarity_and_tone + 0.20*audience_and_location`.
Return strict JSON with scores and a 1-sentence justification per axis."""

JUDGE_RUBRIC_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "intervention_or_silence",
        "content_and_evidence",
        "clarity_and_tone",
        "audience_and_location",
        "overall_score",
        "justifications",
    ],
    "properties": {
        "intervention_or_silence": {"type": "number", "minimum": 0.0, "maximum": 10.0},
        "content_and_evidence": {"type": "number", "minimum": 0.0, "maximum": 10.0},
        "clarity_and_tone": {"type": "number", "minimum": 0.0, "maximum": 10.0},
        "audience_and_location": {"type": "number", "minimum": 0.0, "maximum": 10.0},
        "overall_score": {"type": "number", "minimum": 0.0, "maximum": 10.0},
        "justifications": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "intervention_or_silence",
                "content_and_evidence",
                "clarity_and_tone",
                "audience_and_location",
            ],
            "properties": {
                "intervention_or_silence": {"type": "string", "maxLength": 240},
                "content_and_evidence": {"type": "string", "maxLength": 240},
                "clarity_and_tone": {"type": "string", "maxLength": 240},
                "audience_and_location": {"type": "string", "maxLength": 240},
            },
        },
    },
}

PAGEMANTLE_SYNTHESIZER_SYSTEM_PROMPT = """You are constructing a rigorous, database-grounded workplace benchmark scenario for PageMantle.
You are given REAL verified records extracted directly from PageMantle's local enterprise databases (Slack, Linear, Notion, HubSpot, Stripe, Snowflake, etc.).
Construct a realistic 8-12 message Slack channel thread among PageMantle employees centered on these EXACT verified records.
- For `expected_action = INTERVENE`: embed realistic conversational noise plus a subtle unresolved need, factual discrepancy, diagnostic mystery, coordination clash, or policy risk that requires cross-system lookup to resolve. Specify exact `ideal_tier` (`DM_NUDGE`, `THREAD_BRIEF`, or `CHANNEL_INTERVENE`), `required_evidence_ids`, and `reference_answer`.
- For `expected_action = SILENT` (Hard-Negative Silence Trap): make the thread superficially resemble skill `{category}`, BUT incorporate trap type `{trap_type}` (`already_resolved`, `explicit_deferral`, `brainstorming_flow`, or `scope_boundary`) so that a smart proactive agent MUST stay silent.
NEVER invent fake ticket IDs, metrics, or facts outside the provided verified DB snapshot!
Return strict JSON matching `PageMantleTaskSchema`."""
