"""DIAL-Skill: Decision-Informed Adversarial-Margin Gating with Online Dual-Boundary Skill Evolution."""

__version__ = "0.1.0"

from dial_skill.schemas import (
    ACTIVE_SKILLS,
    INT_TO_TIER,
    TIER_TO_INT,
    AdvocateOutput,
    ChannelPolicy,
    ChatMessage,
    EpisodeResult,
    FeedbackType,
    GateDecision,
    InterventionTier,
    ReconSnippet,
    SharedEvidenceSlate,
    SkepticOutput,
    SkillCard,
    SkillName,
    StanceLabel,
    TriageItem,
)

__all__ = [
    "__version__",
    "ACTIVE_SKILLS",
    "INT_TO_TIER",
    "TIER_TO_INT",
    "AdvocateOutput",
    "ChannelPolicy",
    "ChatMessage",
    "EpisodeResult",
    "FeedbackType",
    "GateDecision",
    "InterventionTier",
    "ReconSnippet",
    "SharedEvidenceSlate",
    "SkepticOutput",
    "SkillCard",
    "SkillName",
    "StanceLabel",
    "TriageItem",
]
