"""Typed social-operational feedback extractor for PageMantle & When2Speak."""

from __future__ import annotations

from dial_skill.schemas import (
    TIER_TO_INT,
    EpisodeFeedback,
    EpisodeResult,
    InterventionTier,
    JudgeRubricScore,
    PageMantleTask,
    SkillName,
)


def extract_feedback_from_pagemantle(
    episode: EpisodeResult,
    task: PageMantleTask,
    judge_score: JudgeRubricScore | None = None,
) -> EpisodeFeedback:
    """Map an executed PageMantle episode + rubric/ground-truth into typed social-operational feedback."""
    pred_tier = episode.gate.final_tier
    pred_speak = pred_tier != InterventionTier.SILENT
    expected_speak = task.expected_action == "INTERVENE"
    skill_name: SkillName = (
        episode.gate.selected_skill if episode.gate.selected_skill != "none" else task.category
    )

    rubric_map: dict[str, float] = {}
    overall: float | None = None
    if judge_score is not None:
        overall = judge_score.overall_score
        rubric_map = {
            "intervention_or_silence": judge_score.intervention_or_silence,
            "content_and_evidence": judge_score.content_and_evidence,
            "clarity_and_tone": judge_score.clarity_and_tone,
            "audience_and_location": judge_score.audience_and_location,
        }

    # Case 1: Ground truth is SILENT (Silence Trap)
    if not expected_speak:
        if not pred_speak:
            return EpisodeFeedback(
                sample_id=episode.sample_id,
                skill_name=skill_name,
                feedback_type="TN_VALIDATED",
                critique="Correctly stayed silent on silence trap.",
                expected_speak=False,
                expected_tier=InterventionTier.SILENT,
                predicted_tier=pred_tier,
                overall_score=overall,
                rubric_scores=rubric_map,
            )
        trap_note = f" (trap_type={task.trap_type})" if task.trap_type else ""
        return EpisodeFeedback(
            sample_id=episode.sample_id,
            skill_name=skill_name,
            feedback_type="FP_INTRUSION",
            critique=f"Interrupted unnecessarily at {pred_tier.value} when silence was required{trap_note}.",
            expected_speak=False,
            expected_tier=InterventionTier.SILENT,
            predicted_tier=pred_tier,
            overall_score=overall,
            rubric_scores=rubric_map,
        )

    # Case 2: Ground truth is INTERVENE
    if not pred_speak:
        return EpisodeFeedback(
            sample_id=episode.sample_id,
            skill_name=skill_name,
            feedback_type="FN_OMISSION",
            critique=(
                f"Missed required {task.category} intervention (ideal tier: {task.ideal_tier.value})."
            ),
            expected_speak=True,
            expected_tier=task.ideal_tier,
            predicted_tier=pred_tier,
            overall_score=overall,
            rubric_scores=rubric_map,
        )

    # Both expected and predicted SPEAK -> check scope overreach vs evidence gap vs validated TP
    tier_overreach = TIER_TO_INT[pred_tier] > TIER_TO_INT[task.ideal_tier]
    judge_scope_low = (
        judge_score is not None
        and judge_score.audience_and_location < 6.5
        and judge_score.content_and_evidence >= 6.0
    )
    if tier_overreach or judge_scope_low:
        return EpisodeFeedback(
            sample_id=episode.sample_id,
            skill_name=skill_name,
            feedback_type="FP_SCOPE_OVERREACH",
            critique=(
                f"Intervened at {pred_tier.value} when {task.ideal_tier.value} was appropriate; "
                "scope was too public or verbose."
            ),
            expected_speak=True,
            expected_tier=task.ideal_tier,
            predicted_tier=pred_tier,
            overall_score=overall,
            rubric_scores=rubric_map,
        )

    if judge_score is not None and judge_score.content_and_evidence < 6.5:
        just = judge_score.justifications.get("content_and_evidence", "")
        return EpisodeFeedback(
            sample_id=episode.sample_id,
            skill_name=skill_name,
            feedback_type="CONTENT_EVIDENCE_GAP",
            critique=f"Timing was right, but evidence was incomplete or inaccurate: {just}".strip(),
            expected_speak=True,
            expected_tier=task.ideal_tier,
            predicted_tier=pred_tier,
            overall_score=overall,
            rubric_scores=rubric_map,
        )

    return EpisodeFeedback(
        sample_id=episode.sample_id,
        skill_name=skill_name,
        feedback_type="TP_VALIDATED",
        critique="Helpful, accurate, well-scoped intervention.",
        expected_speak=True,
        expected_tier=task.ideal_tier,
        predicted_tier=pred_tier,
        overall_score=overall,
        rubric_scores=rubric_map,
    )


def extract_feedback_from_when2speak(
    episode: EpisodeResult,
    label_speak: bool,
    fallback_skill: SkillName = "find",
) -> EpisodeFeedback:
    """Extract typed feedback from When2Speak validation labels (never test!)."""
    pred_speak = episode.gate.binary_speak
    pred_tier = episode.gate.final_tier
    skill_name: SkillName = (
        episode.gate.selected_skill if episode.gate.selected_skill != "none" else fallback_skill
    )

    if label_speak and pred_speak:
        fb_type = "TP_VALIDATED"
        critique = "Correct proactive intervention on When2Speak validation turn."
    elif (not label_speak) and (not pred_speak):
        fb_type = "TN_VALIDATED"
        critique = "Correct silence on When2Speak validation turn."
    elif (not label_speak) and pred_speak:
        fb_type = "FP_INTRUSION"
        critique = "False interruption on a turn where staying silent was optimal."
    else:
        fb_type = "FN_OMISSION"
        critique = "Missed a necessary proactive intervention turn."

    return EpisodeFeedback(
        sample_id=episode.sample_id,
        skill_name=skill_name,
        feedback_type=fb_type,  # type: ignore[arg-type]
        critique=critique,
        expected_speak=label_speak,
        expected_tier=InterventionTier.THREAD_BRIEF if label_speak else InterventionTier.SILENT,
        predicted_tier=pred_tier,
    )
