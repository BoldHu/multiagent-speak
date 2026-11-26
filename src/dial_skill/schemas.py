"""Core Pydantic v2 data contracts for DIAL-Skill."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class InterventionTier(str, Enum):
    SILENT = "SILENT"                        # Tier 0
    DM_NUDGE = "DM_NUDGE"                    # Tier 1
    THREAD_BRIEF = "THREAD_BRIEF"            # Tier 2
    CHANNEL_INTERVENE = "CHANNEL_INTERVENE"  # Tier 3


TIER_TO_INT: dict[InterventionTier, int] = {
    InterventionTier.SILENT: 0,
    InterventionTier.DM_NUDGE: 1,
    InterventionTier.THREAD_BRIEF: 2,
    InterventionTier.CHANNEL_INTERVENE: 3,
}
INT_TO_TIER: dict[int, InterventionTier] = {v: k for k, v in TIER_TO_INT.items()}

SkillName = Literal[
    "find",
    "explain",
    "analyze",
    "check",
    "diagnose",
    "decide",
    "coordinate",
    "guard",
    "none",
]

ActiveSkillName = Literal[
    "find",
    "explain",
    "analyze",
    "check",
    "diagnose",
    "decide",
    "coordinate",
    "guard",
]

ACTIVE_SKILLS: tuple[ActiveSkillName, ...] = (
    "find",
    "explain",
    "analyze",
    "check",
    "diagnose",
    "decide",
    "coordinate",
    "guard",
)

StanceLabel = Literal[
    "need_signal",
    "defer_or_resolved_signal",
    "scope_risk_signal",
    "background_chatter",
]

STANCE_LABELS: tuple[StanceLabel, ...] = (
    "need_signal",
    "defer_or_resolved_signal",
    "scope_risk_signal",
    "background_chatter",
)

FeedbackType = Literal[
    "TP_VALIDATED",
    "TN_VALIDATED",
    "FP_INTRUSION",
    "FP_SCOPE_OVERREACH",
    "FN_OMISSION",
    "CONTENT_EVIDENCE_GAP",
]

RootCauseType = Literal[
    "CALIBRATION_SHIFT",
    "SCOPE_MISMATCH",
    "VETO_MISSING",
    "TRIGGER_MISSING",
    "INVESTIGATION_GAP",
    "ROUTING_CONFUSION",
]


class ChatMessage(BaseModel):
    msg_id: str
    speaker: str
    text: str
    timestamp: str = ""
    channel: str = "#general"
    thread_ts: str | None = None


class TriageItem(BaseModel):
    msg_id: str
    backend: Literal["jev", "openai_fallback"]
    p_decision_change: float = Field(ge=0.0, le=1.0)
    stance: StanceLabel
    stance_probs: dict[StanceLabel, float]
    skill_hint: SkillName
    skill_probs: dict[SkillName, float]
    urgency_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ReconSnippet(BaseModel):
    recon_id: str
    service: str
    query: str
    summary_fact: str


class ChannelPolicy(BaseModel):
    default_tier: InterventionTier
    max_tier: InterventionTier
    deescalation_rule: str


class SkillCard(BaseModel):
    name: SkillName
    description: str
    trigger_conditions: list[str] = Field(max_length=5)
    veto_conditions: list[str] = Field(max_length=5)
    channel_policy: ChannelPolicy
    investigation_plan: list[str] = Field(max_length=4)
    max_tool_calls: int = Field(default=6, ge=1, le=15)
    response_contract: str
    margin_offset: float = Field(default=0.0, ge=-0.35, le=0.35)
    deescalation_bias: float = Field(default=0.0, ge=0.0, le=2.0)
    rule_stats: dict[str, dict[str, int]] = Field(default_factory=dict)


class SharedEvidenceSlate(BaseModel):
    conversation_id: str
    channel_name: str
    env_mode: Literal["when2speak", "pagemantle"]
    latest_msg_id: str
    total_turns: int = 0
    unique_speakers: int = 0
    prior_agent_turns: int = 0
    selected_messages: list[ChatMessage]
    triage_annotations: dict[str, TriageItem]
    recon_items: list[ReconSnippet] = Field(default_factory=list)
    skill_priors: dict[SkillName, float]
    candidate_skills: list[SkillCard]
    canonical_text: str
    slate_hash: str  # sha256(canonical_text) — MUST match between Advocate & Skeptic!


class AdvocateOutput(BaseModel):
    slate_hash: str
    advocate_score: float = Field(ge=0.0, le=1.0)
    unmitigated_loss: str
    recommended_skill: SkillName
    minimum_effective_channel: Literal["DM_NUDGE", "THREAD_BRIEF", "CHANNEL_INTERVENE"]
    cited_evidence_ids: list[str]


class SkepticOutput(BaseModel):
    slate_hash: str
    skeptic_score: float = Field(ge=0.0, le=1.0)
    disruption_risk: str
    matched_veto_conditions: list[str]
    max_acceptable_channel: InterventionTier
    cited_evidence_ids: list[str]


class GateDecision(BaseModel):
    short_circuited: bool = False
    slate_hash: str
    selected_skill: SkillName
    advocate_score: float
    skeptic_score: float
    margin_delta: float  # S_adv - lambda * S_skep - delta_k
    disagreement_u: float  # min(S_adv, S_skep)
    raw_tier: InterventionTier
    final_tier: InterventionTier
    binary_speak: bool  # final_tier != SILENT
    deescalation_reasons: list[str] = Field(default_factory=list)
    advocate: AdvocateOutput | None = None
    skeptic: SkepticOutput | None = None


class EpisodeResult(BaseModel):
    sample_id: str
    benchmark: Literal[
        "when2speak",
        "pagemantle_official",
        "pagemantle_ext",
        "pagemantle_synthetic_sanity",
    ]
    method: str
    model_id: str
    gate: GateDecision
    response_text: str | None = None
    target_channel: str | None = None
    target_user: str | None = None
    tool_calls_used: int = 0
    model_elapsed_sec: float = 0.0
    timed_out: bool = False
    prompt_tokens_raw: int = 0
    prompt_tokens_slate: int = 0
    total_cost_usd: float = 0.0
    uncached_cost_usd: float = 0.0
    triage_backend_used: str = "jev"
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Benchmark & Evolution Contracts
# ---------------------------------------------------------------------------


class When2SpeakTurn(BaseModel):
    sample_id: str
    conversation_id: str
    turn_index: int
    split: Literal["train", "validation", "test"]
    messages: list[ChatMessage]
    label_speak: bool
    intervention_type: str = "unknown"
    reference_response: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PageMantleTask(BaseModel):
    task_id: str
    split: Literal[
        "pm_official_8",
        "pm_synthetic_sanity_8",
        "pm_online_stream_24",
        "pm_eval_32",
    ]
    wave: int | None = None  # 1, 2, or 3 for pm_online_stream_24
    category: ActiveSkillName
    channel_name: str = "#general"
    messages: list[ChatMessage]
    expected_action: Literal["INTERVENE", "SILENT"]
    ideal_tier: InterventionTier
    trap_type: Literal[
        "already_resolved",
        "explicit_deferral",
        "brainstorming_flow",
        "scope_boundary",
        None,
    ] = None
    required_evidence_sources: list[dict[str, Any]] = Field(default_factory=list)
    reference_answer: str = ""
    ground_truth_facts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgeRubricScore(BaseModel):
    sample_id: str
    judge_model: str
    intervention_or_silence: float = Field(ge=0.0, le=10.0)
    content_and_evidence: float = Field(ge=0.0, le=10.0)
    clarity_and_tone: float = Field(ge=0.0, le=10.0)
    audience_and_location: float = Field(ge=0.0, le=10.0)
    overall_score: float = Field(ge=0.0, le=10.0)
    justifications: dict[str, str] = Field(default_factory=dict)


class EpisodeFeedback(BaseModel):
    sample_id: str
    skill_name: SkillName
    feedback_type: FeedbackType
    critique: str = ""
    expected_speak: bool | None = None
    expected_tier: InterventionTier | None = None
    predicted_tier: InterventionTier = InterventionTier.SILENT
    overall_score: float | None = None
    rubric_scores: dict[str, float] = Field(default_factory=dict)


class AttributionResult(BaseModel):
    root_cause: RootCauseType
    affected_skill: ActiveSkillName
    counterfactual_explanation: str
    needs_symbolic_patch: bool


class PatchProposal(BaseModel):
    target_field: Literal["trigger_conditions", "veto_conditions", "investigation_plan"]
    proposed_predicate: str
    rationale: str = ""


class AnchorEpisode(BaseModel):
    sample_id: str
    skill_name: ActiveSkillName
    expected_speak: bool
    expected_tier: InterventionTier
    slate: SharedEvidenceSlate
    original_margin: float


class EvolutionLogEntry(BaseModel):
    timestamp_utc: str
    version_before: int
    version_after: int
    sample_id: str
    skill_name: SkillName
    feedback_type: FeedbackType
    root_cause: RootCauseType | None = None
    numeric_update: dict[str, float] = Field(default_factory=dict)
    patch_proposed: PatchProposal | None = None
    escrow_passed: bool | None = None
    escrow_details: dict[str, Any] = Field(default_factory=dict)
    promoted: bool = False
    pruning_action: str | None = None
    rejection_reason: str | None = None
