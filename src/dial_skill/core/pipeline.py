"""Unified DialSkillAgent orchestrating Stages 1, 1.5, 2, and 3 across When2Speak & PageMantle."""

from __future__ import annotations

from typing import Literal

from dial_skill.config import DialSkillConfig, load_dial_skill_config
from dial_skill.core.executor import ReconResult, SkillExecutor, ToolEnvironment
from dial_skill.core.gate import DialecticGate
from dial_skill.core.skill_store import SkillStore
from dial_skill.core.triage import TriageStage, rebuild_slate_with_recon
from dial_skill.llm.jev_client import TriageClient
from dial_skill.llm.openai_client import OpenAIClient
from dial_skill.schemas import (
    ChatMessage,
    EpisodeResult,
    GateDecision,
    InterventionTier,
    SharedEvidenceSlate,
)


class DialSkillAgent:
    """End-to-end DIAL-Skill proactive agent (Zero weight updates, full SSES & budget enforcement)."""

    def __init__(
        self,
        *,
        triage_client: TriageClient,
        openai_client: OpenAIClient,
        gate_model: str,
        executor_model: str,
        skill_store: SkillStore | None = None,
        cfg: DialSkillConfig | None = None,
        method_name: str = "dial_skill",
        seed: int = 42,
    ) -> None:
        self.cfg = cfg or load_dial_skill_config()
        self.skill_store = skill_store or SkillStore.from_json_file()
        self.triage_client = triage_client
        self.openai_client = openai_client
        self.gate_model = gate_model
        self.executor_model = executor_model
        self.method_name = method_name
        self.seed = seed

        self.triage_stage = TriageStage(triage_client=triage_client, cfg=self.cfg)
        self.gate_stage = DialecticGate(
            openai_client=openai_client,
            gate_model=gate_model,
            cfg=self.cfg,
            seed=seed,
        )
        self.executor_stage = SkillExecutor(
            openai_client=openai_client,
            executor_model=executor_model,
            cfg=self.cfg,
            seed=seed,
        )

    async def run_episode(
        self,
        *,
        sample_id: str,
        conversation_id: str,
        channel_name: str,
        messages: list[ChatMessage],
        benchmark: Literal["when2speak", "pagemantle_official", "pagemantle_ext"],
        tool_env: ToolEnvironment | None = None,
        generate_text: bool = True,
        include_slate_in_metadata: bool = False,
    ) -> EpisodeResult:
        env_mode: Literal["when2speak", "pagemantle"] = (
            "when2speak" if benchmark == "when2speak" else "pagemantle"
        )
        triage_cost_before = self.triage_client.total_cost_usd

        # Stage 1: System-1 Decision-Change Triage & Balanced SSES
        triage_out = await self.triage_stage.run(
            conversation_id=conversation_id,
            channel_name=channel_name,
            env_mode=env_mode,
            messages=messages,
            skill_store=self.skill_store,
        )
        triage_cost = max(0.0, self.triage_client.total_cost_usd - triage_cost_before)

        # Fast-silence short-circuit
        if triage_out.short_circuited:
            gate = GateDecision(
                short_circuited=True,
                slate_hash=triage_out.slate.slate_hash,
                selected_skill="none",
                advocate_score=0.0,
                skeptic_score=1.0,
                margin_delta=-1.0,
                disagreement_u=0.0,
                raw_tier=InterventionTier.SILENT,
                final_tier=InterventionTier.SILENT,
                binary_speak=False,
                deescalation_reasons=["stage1_fast_silence_short_circuit"],
            )
            meta = {"recon_calls": 0, "aborted_to_silent": False}
            if include_slate_in_metadata:
                meta["slate"] = triage_out.slate.model_dump(mode="json")
            return EpisodeResult(
                sample_id=sample_id,
                benchmark=benchmark,
                method=self.method_name,
                model_id=self.gate_model,
                gate=gate,
                response_text=None,
                tool_calls_used=0,
                model_elapsed_sec=0.0,
                timed_out=False,
                prompt_tokens_raw=triage_out.prompt_tokens_raw,
                prompt_tokens_slate=triage_out.prompt_tokens_slate,
                total_cost_usd=round(triage_cost, 8),
                triage_backend_used=triage_out.triage_backend_used,
                metadata=meta,
            )

        slate = triage_out.slate
        adv_slate = triage_out.advocate_slate
        skep_slate = triage_out.skeptic_slate

        # Stage 1.5: Bounded Read-Only Silent Recon (PageMantle mode only)
        recon_res = ReconResult()
        if env_mode == "pagemantle" and tool_env is not None:
            recon_res = await self.executor_stage.run_silent_recon(
                slate=slate,
                tool_env=tool_env,
            )
            if recon_res.snippets:
                slate = rebuild_slate_with_recon(slate, recon_res.snippets)
                if self.cfg.ablation.asymmetric_evidence:
                    adv_slate = rebuild_slate_with_recon(adv_slate, recon_res.snippets)
                    skep_slate = rebuild_slate_with_recon(skep_slate, recon_res.snippets)
                else:
                    adv_slate = slate
                    skep_slate = slate

        # Stage 2: Parallel Symmetric-Evidence Dialectic Gating
        gate_decision, gate_cost, gate_elapsed = await self.gate_stage.evaluate(
            advocate_slate=adv_slate,
            skeptic_slate=skep_slate,
            skill_store=self.skill_store,
        )

        # Stage 3: Applicability-Routed Skill Execution
        skill_card = (
            None
            if self.cfg.ablation.disable_skill_routing
            else self.skill_store.get(gate_decision.selected_skill)
        )
        prior_model_elapsed = recon_res.model_elapsed_sec + gate_elapsed
        exec_out = await self.executor_stage.run_execution(
            slate=slate,
            gate=gate_decision,
            skill_card=skill_card,
            tool_env=tool_env,
            prior_recon=recon_res,
            prior_model_elapsed_sec=prior_model_elapsed,
            generate_text=generate_text,
        )

        if exec_out.aborted_to_silent:
            gate_decision = gate_decision.model_copy(
                update={
                    "final_tier": InterventionTier.SILENT,
                    "binary_speak": False,
                    "deescalation_reasons": [
                        *gate_decision.deescalation_reasons,
                        f"stage3_abort_to_silent:{exec_out.abort_reason or 'false_alarm'}",
                    ],
                }
            )

        total_cost = triage_cost + recon_res.cost_usd + gate_cost + exec_out.cost_usd
        total_tools = recon_res.tool_calls_used + exec_out.tool_calls_used

        meta = {
            "recon_calls": recon_res.tool_calls_used,
            "recon_snippets": [r.model_dump() for r in recon_res.snippets],
            "aborted_to_silent": exec_out.aborted_to_silent,
            "tool_trace": exec_out.tool_trace,
            "skill_store_version": self.skill_store.version,
            "triage_fallback_count": self.triage_client.fallback_count,
        }
        if include_slate_in_metadata:
            meta["slate"] = slate.model_dump(mode="json")

        return EpisodeResult(
            sample_id=sample_id,
            benchmark=benchmark,
            method=self.method_name,
            model_id=f"{self.gate_model}+{self.executor_model}" if generate_text else self.gate_model,
            gate=gate_decision,
            response_text=exec_out.response_text,
            target_channel=exec_out.target_channel,
            target_user=exec_out.target_user,
            tool_calls_used=total_tools,
            model_elapsed_sec=round(exec_out.model_elapsed_sec, 4),
            timed_out=exec_out.timed_out,
            prompt_tokens_raw=triage_out.prompt_tokens_raw,
            prompt_tokens_slate=triage_out.prompt_tokens_slate,
            total_cost_usd=round(total_cost, 8),
            triage_backend_used=triage_out.triage_backend_used,
            metadata=meta,
        )

    async def rescore_slate_with_store(
        self,
        *,
        slate: SharedEvidenceSlate,
        candidate_store: SkillStore,
    ) -> GateDecision:
        """Re-run Stage 2 on a cached SharedEvidenceSlate with a candidate SkillStore (for DB-CSE EscrowGate)."""
        rebuilt_skills = [
            candidate_store.require(card.name)
            for card in slate.candidate_skills
            if candidate_store.get(card.name) is not None
        ]
        from dial_skill.prompts.templates import render_shared_evidence_slate

        canonical_text, slate_hash = render_shared_evidence_slate(
            channel_name=slate.channel_name,
            env_mode=slate.env_mode,
            latest_msg_id=slate.latest_msg_id,
            selected_messages=slate.selected_messages,
            triage_annotations=slate.triage_annotations,
            recon_items=slate.recon_items,
            skill_priors=slate.skill_priors,
            candidate_skills=rebuilt_skills,
        )
        updated_slate = slate.model_copy(
            update={
                "candidate_skills": rebuilt_skills,
                "canonical_text": canonical_text,
                "slate_hash": slate_hash,
            }
        )
        decision, _, _ = await self.gate_stage.evaluate(
            advocate_slate=updated_slate,
            skeptic_slate=updated_slate,
            skill_store=candidate_store,
        )
        return decision
