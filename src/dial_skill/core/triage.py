"""Stage 1: System-1 Decision-Change Triage, fast-silence short-circuit, and Balanced SSES builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dial_skill.config import DialSkillConfig
from dial_skill.core.skill_store import SkillStore
from dial_skill.llm.jev_client import ALL_SKILL_NAMES, TriageClient
from dial_skill.prompts.templates import render_shared_evidence_slate
from dial_skill.schemas import (
    ACTIVE_SKILLS,
    ChatMessage,
    ReconSnippet,
    SharedEvidenceSlate,
    SkillCard,
    SkillName,
    TriageItem,
)


def estimate_tokens(text: str) -> int:
    """Fast, deterministic token estimator (~4 chars/token) avoiding heavy tokenizer init overhead."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


@dataclass
class TriageOutput:
    short_circuited: bool
    slate: SharedEvidenceSlate
    advocate_slate: SharedEvidenceSlate
    skeptic_slate: SharedEvidenceSlate
    all_triage_items: dict[str, TriageItem]
    prompt_tokens_raw: int
    prompt_tokens_slate: int
    triage_backend_used: str


def _uniform_skill_priors() -> dict[SkillName, float]:
    val = round(1.0 / len(ACTIVE_SKILLS), 6)
    priors: dict[SkillName, float] = {k: val for k in ACTIVE_SKILLS}
    priors["none"] = 0.0
    return priors


def _compute_recency_skill_priors(
    selected_indices: list[int],
    messages: list[ChatMessage],
    annotations: dict[str, TriageItem],
    latest_idx: int,
    beta: float,
) -> dict[SkillName, float]:
    eps = 1e-8
    weighted: dict[SkillName, float] = {k: 0.0 for k in ALL_SKILL_NAMES}
    denom = 0.0
    for idx in selected_indices:
        msg = messages[idx]
        ann = annotations.get(msg.msg_id)
        if ann is None:
            continue
        w = (beta ** max(0, latest_idx - idx)) * ann.p_decision_change
        denom += w
        for skill_k in ALL_SKILL_NAMES:
            weighted[skill_k] += w * float(ann.skill_probs.get(skill_k, 0.0))
    if denom <= 0.0:
        return {k: (1.0 if k == "none" else 0.0) for k in ALL_SKILL_NAMES}
    return {k: round(v / (denom + eps), 6) for k, v in weighted.items()}


def _pick_top_candidate_skills(
    skill_priors: dict[SkillName, float],
    skill_store: SkillStore,
    top_k: int = 2,
) -> list[SkillCard]:
    ranked = sorted(
        ACTIVE_SKILLS,
        key=lambda k: float(skill_priors.get(k, 0.0)),
        reverse=True,
    )
    cards: list[SkillCard] = []
    for name in ranked[:top_k]:
        card = skill_store.get(name)
        if card is not None:
            cards.append(card)
    return cards


def rebuild_slate_with_recon(
    slate: SharedEvidenceSlate,
    recon_items: list[ReconSnippet],
) -> SharedEvidenceSlate:
    """Attach Stage-1.5 Silent Recon items to a slate and recompute canonical_text & slate_hash."""
    merged_recon = [*slate.recon_items, *recon_items]
    canonical_text, slate_hash = render_shared_evidence_slate(
        channel_name=slate.channel_name,
        env_mode=slate.env_mode,
        latest_msg_id=slate.latest_msg_id,
        selected_messages=slate.selected_messages,
        triage_annotations=slate.triage_annotations,
        recon_items=merged_recon,
        skill_priors=slate.skill_priors,
        candidate_skills=slate.candidate_skills,
    )
    return slate.model_copy(
        update={
            "recon_items": merged_recon,
            "canonical_text": canonical_text,
            "slate_hash": slate_hash,
        }
    )


class TriageStage:
    """Stage-1 incremental message triage and Symmetric Shared Evidence Slate (SSES) builder."""

    def __init__(
        self,
        triage_client: TriageClient,
        cfg: DialSkillConfig,
    ) -> None:
        self.triage_client = triage_client
        self.cfg = cfg

    def _select_balanced_indices(
        self,
        messages: list[ChatMessage],
        annotations: dict[str, TriageItem],
    ) -> list[int]:
        """Select <= K_slate turns retaining BOTH need_signal AND defer/scope signals (Invariant #2)."""
        n = len(messages)
        k_slate = max(2, self.cfg.triage.k_slate)
        if n <= k_slate:
            return list(range(n))

        # Always include latest 2 turns for immediate conversational continuity
        pinned = {n - 2, n - 1}
        older_indices = list(range(0, n - 2))

        budget_rem = k_slate - len(pinned)
        need_quota = budget_rem // 2
        defer_quota = budget_rem - need_quota

        def _need_score(i: int) -> float:
            ann = annotations[messages[i].msg_id]
            return ann.p_decision_change * float(ann.stance_probs.get("need_signal", 0.0))

        def _defer_scope_score(i: int) -> float:
            ann = annotations[messages[i].msg_id]
            return ann.p_decision_change * (
                float(ann.stance_probs.get("defer_or_resolved_signal", 0.0))
                + float(ann.stance_probs.get("scope_risk_signal", 0.0))
            )

        need_ranked = sorted(older_indices, key=_need_score, reverse=True)
        defer_ranked = sorted(older_indices, key=_defer_scope_score, reverse=True)

        selected: set[int] = set(pinned)

        # Interleave need and defer/scope selection so neither side starves the other
        need_added = 0
        defer_added = 0
        ni = 0
        di = 0
        while (need_added < need_quota or defer_added < defer_quota) and (
            ni < len(need_ranked) or di < len(defer_ranked)
        ):
            if need_added < need_quota and ni < len(need_ranked):
                cand = need_ranked[ni]
                ni += 1
                if cand not in selected:
                    selected.add(cand)
                    need_added += 1
            if defer_added < defer_quota and di < len(defer_ranked):
                cand = defer_ranked[di]
                di += 1
                if cand not in selected:
                    selected.add(cand)
                    defer_added += 1

        return sorted(selected)

    def _select_asymmetric_indices(
        self,
        messages: list[ChatMessage],
        annotations: dict[str, TriageItem],
        side: Literal["advocate", "skeptic"],
    ) -> list[int]:
        """Ablation A3 ONLY: construct an asymmetric view biased toward one side."""
        n = len(messages)
        k_slate = max(2, self.cfg.triage.k_slate)
        if n <= k_slate:
            return list(range(n))
        pinned = {n - 2, n - 1}
        older = list(range(0, n - 2))
        if side == "advocate":
            ranked = sorted(
                older,
                key=lambda i: annotations[messages[i].msg_id].p_decision_change
                * float(annotations[messages[i].msg_id].stance_probs.get("need_signal", 0.0)),
                reverse=True,
            )
        else:
            ranked = sorted(
                older,
                key=lambda i: annotations[messages[i].msg_id].p_decision_change
                * (
                    float(annotations[messages[i].msg_id].stance_probs.get("defer_or_resolved_signal", 0.0))
                    + float(annotations[messages[i].msg_id].stance_probs.get("scope_risk_signal", 0.0))
                ),
                reverse=True,
            )
        chosen = set(pinned)
        for idx in ranked:
            if len(chosen) >= k_slate:
                break
            chosen.add(idx)
        return sorted(chosen)

    def _build_slate(
        self,
        *,
        conversation_id: str,
        channel_name: str,
        env_mode: Literal["when2speak", "pagemantle"],
        messages: list[ChatMessage],
        selected_indices: list[int],
        annotations: dict[str, TriageItem],
        skill_store: SkillStore,
    ) -> SharedEvidenceSlate:
        selected_messages = [messages[i] for i in selected_indices]
        selected_anns = {
            m.msg_id: annotations[m.msg_id] for m in selected_messages if m.msg_id in annotations
        }
        latest_idx = len(messages) - 1
        latest_msg_id = messages[latest_idx].msg_id

        if annotations:
            skill_priors = _compute_recency_skill_priors(
                selected_indices,
                messages,
                annotations,
                latest_idx=latest_idx,
                beta=self.cfg.triage.recency_beta,
            )
        else:
            skill_priors = _uniform_skill_priors()

        candidate_skills = (
            []
            if self.cfg.ablation.disable_skill_routing
            else _pick_top_candidate_skills(skill_priors, skill_store, top_k=2)
        )

        canonical_text, slate_hash = render_shared_evidence_slate(
            channel_name=channel_name,
            env_mode=env_mode,
            latest_msg_id=latest_msg_id,
            selected_messages=selected_messages,
            triage_annotations=selected_anns,
            recon_items=[],
            skill_priors=skill_priors,
            candidate_skills=candidate_skills,
        )
        return SharedEvidenceSlate(
            conversation_id=conversation_id,
            channel_name=channel_name,
            env_mode=env_mode,
            latest_msg_id=latest_msg_id,
            selected_messages=selected_messages,
            triage_annotations=selected_anns,
            recon_items=[],
            skill_priors=skill_priors,
            candidate_skills=candidate_skills,
            canonical_text=canonical_text,
            slate_hash=slate_hash,
        )

    async def run(
        self,
        *,
        conversation_id: str,
        channel_name: str,
        env_mode: Literal["when2speak", "pagemantle"],
        messages: list[ChatMessage],
        skill_store: SkillStore,
    ) -> TriageOutput:
        if not messages:
            raise ValueError("TriageStage requires at least 1 message")

        raw_chat_text = "\n".join(f"[{m.msg_id}] {m.speaker}: {m.text}" for m in messages)
        prompt_tokens_raw = estimate_tokens(raw_chat_text)

        # Ablation A1: w/o Jev Triage — pass full raw chat directly to Advocate & Skeptic
        if self.cfg.ablation.disable_triage:
            slate = self._build_slate(
                conversation_id=conversation_id,
                channel_name=channel_name,
                env_mode=env_mode,
                messages=messages,
                selected_indices=list(range(len(messages))),
                annotations={},
                skill_store=skill_store,
            )
            return TriageOutput(
                short_circuited=False,
                slate=slate,
                advocate_slate=slate,
                skeptic_slate=slate,
                all_triage_items={},
                prompt_tokens_raw=prompt_tokens_raw,
                prompt_tokens_slate=estimate_tokens(slate.canonical_text),
                triage_backend_used="disabled_a1",
            )

        backend_override = self.cfg.ablation.triage_backend_override
        annotations: dict[str, TriageItem] = {}
        backends_seen: set[str] = set()
        for idx, msg in enumerate(messages):
            prior = messages[max(0, idx - 2) : idx]
            item = await self.triage_client.score_message(
                channel_name=channel_name,
                prior_messages=prior,
                target_message=msg,
                backend_override=backend_override,
            )
            annotations[msg.msg_id] = item
            backends_seen.add(item.backend)

        triage_backend_used = (
            "openai_fallback"
            if backends_seen == {"openai_fallback"}
            else ("mixed" if len(backends_seen) > 1 else next(iter(backends_seen), "jev"))
        )

        # Fast-silence short-circuit check (Section 3 Stage 1.2)
        latest_item = annotations[messages[-1].msg_id]
        window_msgs = messages[-self.cfg.triage.active_window_turns :]
        max_window_need = max(
            float(annotations[m.msg_id].stance_probs.get("need_signal", 0.0))
            for m in window_msgs
        )
        short_circuited = (
            self.cfg.triage.theta_skip > 0.0
            and latest_item.p_decision_change < self.cfg.triage.theta_skip
            and max_window_need < self.cfg.triage.theta_need
        )

        balanced_indices = self._select_balanced_indices(messages, annotations)
        slate = self._build_slate(
            conversation_id=conversation_id,
            channel_name=channel_name,
            env_mode=env_mode,
            messages=messages,
            selected_indices=balanced_indices,
            annotations=annotations,
            skill_store=skill_store,
        )

        # Invariant #1: Advocate and Skeptic get the EXACT SAME slate unless Ablation A3 is active!
        if self.cfg.ablation.asymmetric_evidence:
            adv_idx = self._select_asymmetric_indices(messages, annotations, side="advocate")
            skep_idx = self._select_asymmetric_indices(messages, annotations, side="skeptic")
            adv_slate = self._build_slate(
                conversation_id=conversation_id,
                channel_name=channel_name,
                env_mode=env_mode,
                messages=messages,
                selected_indices=adv_idx,
                annotations=annotations,
                skill_store=skill_store,
            )
            skep_slate = self._build_slate(
                conversation_id=conversation_id,
                channel_name=channel_name,
                env_mode=env_mode,
                messages=messages,
                selected_indices=skep_idx,
                annotations=annotations,
                skill_store=skill_store,
            )
        else:
            adv_slate = slate
            skep_slate = slate

        return TriageOutput(
            short_circuited=short_circuited,
            slate=slate,
            advocate_slate=adv_slate,
            skeptic_slate=skep_slate,
            all_triage_items=annotations,
            prompt_tokens_raw=prompt_tokens_raw,
            prompt_tokens_slate=estimate_tokens(slate.canonical_text),
            triage_backend_used=triage_backend_used,
        )
