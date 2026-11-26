"""Blinded Dual-Judge evaluator (Deepnote 4-axis rubric, Dialogue quality, Cohen's kappa, Spearman rho, CSV export)."""

from __future__ import annotations

import csv
import hashlib
import random
import re
from pathlib import Path
from typing import Any, Sequence

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from dial_skill.llm.openai_client import OpenAIClient
from dial_skill.prompts.templates import JUDGE_RUBRIC_JSON_SCHEMA, JUDGE_RUBRIC_SYSTEM_PROMPT
from dial_skill.schemas import EpisodeResult, JudgeRubricScore, PageMantleTask

RUBRIC_WEIGHTS = {
    "intervention_or_silence": 0.30,
    "content_and_evidence": 0.35,
    "clarity_and_tone": 0.15,
    "audience_and_location": 0.20,
}

DIALOGUE_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["helpfulness", "conciseness", "appropriateness", "overall_score", "justification"],
    "properties": {
        "helpfulness": {"type": "number", "minimum": 0.0, "maximum": 10.0},
        "conciseness": {"type": "number", "minimum": 0.0, "maximum": 10.0},
        "appropriateness": {"type": "number", "minimum": 0.0, "maximum": 10.0},
        "overall_score": {"type": "number", "minimum": 0.0, "maximum": 10.0},
        "justification": {"type": "string", "maxLength": 240},
    },
}


def _coerce_rubric_score(val: Any) -> float:
    """Safely extract a 0-10 float from raw judge JSON (handles flat numbers, strings, or {'score': x} dicts)."""
    if isinstance(val, (int, float)):
        return max(0.0, min(10.0, float(val)))
    if isinstance(val, dict):
        for key in ("score", "rating", "value", "points", "grade"):
            if key in val and isinstance(val[key], (int, float, str)):
                return _coerce_rubric_score(val[key])
        for v in val.values():
            if isinstance(v, (int, float)):
                return max(0.0, min(10.0, float(v)))
        return 0.0
    if isinstance(val, str):
        m = re.search(r"-?\d+(?:\.\d+)?", val)
        if m:
            return max(0.0, min(10.0, float(m.group(0))))
    return 0.0


def compute_weighted_overall(
    intervention_or_silence: float,
    content_and_evidence: float,
    clarity_and_tone: float,
    audience_and_location: float,
) -> float:
    score = (
        RUBRIC_WEIGHTS["intervention_or_silence"] * intervention_or_silence
        + RUBRIC_WEIGHTS["content_and_evidence"] * content_and_evidence
        + RUBRIC_WEIGHTS["clarity_and_tone"] * clarity_and_tone
        + RUBRIC_WEIGHTS["audience_and_location"] * audience_and_location
    )
    return round(max(0.0, min(10.0, score)), 4)


def compute_cohens_kappa(ratings_a: Sequence[int | str], ratings_b: Sequence[int | str]) -> float:
    if len(ratings_a) != len(ratings_b) or not ratings_a:
        return 0.0
    if len(set(ratings_a)) == 1 and set(ratings_a) == set(ratings_b):
        return 1.0
    return round(float(cohen_kappa_score(ratings_a, ratings_b)), 6)


def compute_spearman_rho(scores_a: Sequence[float], scores_b: Sequence[float]) -> dict[str, float]:
    if len(scores_a) != len(scores_b) or len(scores_a) < 2:
        return {"rho": 0.0, "p_value": 1.0}
    res = spearmanr(scores_a, scores_b)
    rho = float(res.statistic) if res.statistic == res.statistic else 0.0
    pval = float(res.pvalue) if res.pvalue == res.pvalue else 1.0
    return {"rho": round(rho, 6), "p_value": round(pval, 8)}


def _anonymize_candidate_tag(sample_id: str, method: str, seed: int) -> str:
    h = hashlib.sha256(f"{seed}:{sample_id}:{method}".encode("utf-8")).hexdigest()[:8]
    return f"CANDIDATE_{h.upper()}"


class BlindedDualJudge:
    """Blinded Dual-Judge grading anonymized agent outputs at temperature=0.0 across two rubric passes."""

    def __init__(
        self,
        *,
        openai_client: OpenAIClient,
        primary_judge_model: str,
        secondary_judge_model: str | None = None,
        secondary_openai_client: OpenAIClient | None = None,
        seed: int = 42,
    ) -> None:
        self.openai_client = openai_client
        self.primary_judge_model = primary_judge_model
        self.secondary_judge_model = secondary_judge_model or primary_judge_model
        self.secondary_openai_client = secondary_openai_client
        self.seed = seed

    def _build_blinded_prompt(
        self,
        task: PageMantleTask,
        episode: EpisodeResult,
        pass_index: int,
    ) -> str:
        blind_tag = _anonymize_candidate_tag(task.task_id, episode.method, self.seed + pass_index)
        transcript = "\n".join(f"[{m.msg_id}] {m.speaker}: {m.text}" for m in task.messages)
        facts_str = "\n".join(f"- {f}" for f in task.ground_truth_facts) or "(see reference answer)"
        agent_action = (
            "SILENT (did not send any message)"
            if not episode.gate.binary_speak
            else (
                f"INTERVENED at tier={episode.gate.final_tier.value}, "
                f"channel={episode.target_channel or task.channel_name}, "
                f"target_user={episode.target_user or 'group'}\n"
                f"Message:\n{episode.response_text or '(empty)'}"
            )
        )

        # Dual-pass ordering variation reduces prompt-order bias while remaining strictly blinded
        if pass_index == 0:
            return (
                f"Blinded Submission ID: {blind_tag}\n"
                f"Channel: {task.channel_name} | Category: {task.category}\n"
                f"Ground Truth Timing: {task.expected_action} | Ideal Channel Tier: {task.ideal_tier.value}\n\n"
                f"Conversation Transcript:\n{transcript}\n\n"
                f"Verified Ground-Truth Facts:\n{facts_str}\n\n"
                f"Reference Answer:\n{task.reference_answer}\n\n"
                f"Blinded Agent Execution:\n{agent_action}"
            )
        return (
            f"Blinded Submission ID: {blind_tag}\n"
            f"Blinded Agent Execution:\n{agent_action}\n\n"
            f"Reference Answer:\n{task.reference_answer}\n\n"
            f"Verified Ground-Truth Facts:\n{facts_str}\n\n"
            f"Ground Truth Timing: {task.expected_action} | Ideal Channel Tier: {task.ideal_tier.value}\n"
            f"Channel: {task.channel_name} | Category: {task.category}\n\n"
            f"Conversation Transcript:\n{transcript}"
        )

    async def grade_pagemantle_episode(
        self,
        task: PageMantleTask,
        episode: EpisodeResult,
    ) -> JudgeRubricScore:
        """Run two blinded deterministic passes (temperature=0.0) across primary & secondary (optionally cross-family) judges."""
        client_model_pairs = [
            (self.openai_client, self.primary_judge_model),
            (self.secondary_openai_client or self.openai_client, self.secondary_judge_model),
        ]
        axis_totals = {k: 0.0 for k in RUBRIC_WEIGHTS}
        justifications: dict[str, str] = {}

        for pass_idx, (judge_client, judge_model) in enumerate(client_model_pairs):
            user_prompt = self._build_blinded_prompt(task, episode, pass_idx)
            res = await judge_client.complete_json(
                model=judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_RUBRIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                json_schema=JUDGE_RUBRIC_JSON_SCHEMA,
                schema_name="blinded_pagemantle_rubric",
                temperature=0.0,
                seed=self.seed + pass_idx,
            )
            data = res.parsed_json or {}
            scores_src = data.get("scores") if isinstance(data.get("scores"), dict) else data
            for axis in RUBRIC_WEIGHTS:
                raw_val = data.get(axis, scores_src.get(axis, 0.0))
                axis_totals[axis] += _coerce_rubric_score(raw_val)
            if pass_idx == 0 and isinstance(data.get("justifications"), dict):
                justifications = {k: str(v) for k, v in data["justifications"].items()}

        avg = {k: round(v / len(client_model_pairs), 4) for k, v in axis_totals.items()}
        overall = compute_weighted_overall(
            avg["intervention_or_silence"],
            avg["content_and_evidence"],
            avg["clarity_and_tone"],
            avg["audience_and_location"],
        )
        return JudgeRubricScore(
            sample_id=task.task_id,
            judge_model=f"dual:{self.primary_judge_model}+{self.secondary_judge_model}",
            intervention_or_silence=avg["intervention_or_silence"],
            content_and_evidence=avg["content_and_evidence"],
            clarity_and_tone=avg["clarity_and_tone"],
            audience_and_location=avg["audience_and_location"],
            overall_score=overall,
            justifications=justifications,
        )

    async def grade_dialogue_response(
        self,
        *,
        sample_id: str,
        transcript: str,
        candidate_response: str,
        reference_response: str | None = None,
        method_tag_for_blinding: str = "candidate",
    ) -> dict[str, Any]:
        """Blinded Dual-Pass (optionally Cross-Family) Dialogue Quality Judge for When2Speak SPEAK turns."""
        use_dual = (
            self.secondary_openai_client is not None
            or self.secondary_judge_model != self.primary_judge_model
        )
        pairs = (
            [
                (self.openai_client, self.primary_judge_model),
                (self.secondary_openai_client or self.openai_client, self.secondary_judge_model),
            ]
            if use_dual
            else [(self.openai_client, self.primary_judge_model)]
        )

        sys_prompt = (
            "You are a strict, blinded evaluator grading a proactive AI participant (`[AGENT]`) in a "
            "multi-party group conversation on three 0-10 axes plus holistic overall_score (0-10):\n"
            "1. helpfulness (0-10): substantive accuracy, relevance, and value added relative to the Reference Response.\n"
            "2. conciseness (0-10): crisp, natural group-chat brevity (penalize stiff corporate headers, bullet bloat, or rambling).\n"
            "3. appropriateness (0-10): natural conversational register, non-intrusiveness, and collaborative tone.\n"
            "Return strict JSON."
        )

        per_judge: list[dict[str, Any]] = []
        for idx, (j_client, j_model) in enumerate(pairs):
            blind_id = _anonymize_candidate_tag(sample_id, method_tag_for_blinding, self.seed + idx)
            if idx % 2 == 0:
                user_prompt = (
                    f"Blinded Submission: {blind_id}\n"
                    f"Conversation History:\n{transcript}\n\n"
                    f"Gold Reference Response:\n{reference_response or '(none)'}\n\n"
                    f"Blinded Candidate Response:\n{candidate_response}"
                )
            else:
                user_prompt = (
                    f"Blinded Submission: {blind_id}\n"
                    f"Blinded Candidate Response:\n{candidate_response}\n\n"
                    f"Gold Reference Response:\n{reference_response or '(none)'}\n\n"
                    f"Conversation History:\n{transcript}"
                )
            res = await j_client.complete_json(
                model=j_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                json_schema=DIALOGUE_JUDGE_SCHEMA,
                schema_name="blinded_dialogue_quality",
                temperature=0.0,
                seed=self.seed + idx,
            )
            raw = dict(res.parsed_json or {})
            per_judge.append(
                {
                    "judge_model": j_model,
                    "helpfulness": _coerce_rubric_score(raw.get("helpfulness", 0.0)),
                    "conciseness": _coerce_rubric_score(raw.get("conciseness", 0.0)),
                    "appropriateness": _coerce_rubric_score(raw.get("appropriateness", 0.0)),
                    "overall_score": _coerce_rubric_score(raw.get("overall_score", 0.0)),
                    "justification": str(raw.get("justification", "")),
                }
            )

        n_j = len(per_judge)
        return {
            "sample_id": sample_id,
            "judge_model": "+".join(p["judge_model"] for p in per_judge),
            "helpfulness": round(sum(p["helpfulness"] for p in per_judge) / n_j, 4),
            "conciseness": round(sum(p["conciseness"] for p in per_judge) / n_j, 4),
            "appropriateness": round(sum(p["appropriateness"] for p in per_judge) / n_j, 4),
            "overall_score": round(sum(p["overall_score"] for p in per_judge) / n_j, 4),
            "justification": per_judge[0]["justification"],
            "per_judge": per_judge,
        }


def export_human_spotcheck_csv(
    *,
    episodes: Sequence[EpisodeResult],
    tasks: Sequence[PageMantleTask],
    judge_scores: Sequence[JudgeRubricScore],
    out_csv: str | Path,
    n_samples: int = 30,
    seed: int = 42,
) -> Path:
    """Export anonymized stratified episodes to CSV for 30-minute human calibration spot-check."""
    task_map = {t.task_id: t for t in tasks}
    judge_map = {j.sample_id: j for j in judge_scores}

    paired = [
        (ep, task_map[ep.sample_id], judge_map.get(ep.sample_id))
        for ep in episodes
        if ep.sample_id in task_map
    ]
    rng = random.Random(seed)
    rng.shuffle(paired)
    chosen = paired[: min(n_samples, len(paired))]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "blind_id",
        "sample_id",
        "category",
        "expected_action",
        "ideal_tier",
        "predicted_tier",
        "transcript_preview",
        "agent_response",
        "reference_answer",
        "judge_overall_0_10",
        "human_overall_0_10",
        "human_timing_binary_ok",
        "human_notes",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (ep, task, j) in enumerate(chosen, start=1):
            transcript_preview = " | ".join(f"{m.speaker}: {m.text}" for m in task.messages[-4:])
            writer.writerow(
                {
                    "blind_id": f"SPOT_{idx:02d}_{_anonymize_candidate_tag(task.task_id, ep.method, seed)}",
                    "sample_id": task.task_id,
                    "category": task.category,
                    "expected_action": task.expected_action,
                    "ideal_tier": task.ideal_tier.value,
                    "predicted_tier": ep.gate.final_tier.value,
                    "transcript_preview": transcript_preview[:600],
                    "agent_response": (ep.response_text or "SILENT")[:600],
                    "reference_answer": task.reference_answer[:400],
                    "judge_overall_0_10": j.overall_score if j else "",
                    "human_overall_0_10": "",
                    "human_timing_binary_ok": "",
                    "human_notes": "",
                }
            )
    return out_path
