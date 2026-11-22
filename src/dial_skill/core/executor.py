"""Stage 1.5 Bounded Silent Recon & Stage 3 Budgeted Tool + Channel-Shaped Execution (Invariant #7)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from dial_skill.config import DialSkillConfig
from dial_skill.llm.openai_client import OpenAIClient
from dial_skill.prompts.templates import build_execution_system_prompt
from dial_skill.schemas import (
    GateDecision,
    InterventionTier,
    ReconSnippet,
    SharedEvidenceSlate,
    SkillCard,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class ToolEnvironment(Protocol):
    """Lightweight protocol implemented by PageMantleEnv and offline test doubles."""

    def get_openai_tool_schemas(self, read_only: bool = False) -> list[dict[str, Any]]: ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


@dataclass
class ReconResult:
    snippets: list[ReconSnippet] = field(default_factory=list)
    tool_calls_used: int = 0
    model_elapsed_sec: float = 0.0
    cost_usd: float = 0.0
    executed_calls: dict[str, str] = field(default_factory=dict)


@dataclass
class ExecutionOutput:
    response_text: str | None = None
    target_channel: str | None = None
    target_user: str | None = None
    aborted_to_silent: bool = False
    abort_reason: str | None = None
    tool_calls_used: int = 0
    model_elapsed_sec: float = 0.0
    timed_out: bool = False
    cost_usd: float = 0.0
    tool_trace: list[dict[str, Any]] = field(default_factory=list)


def _canonical_call_key(name: str, arguments: dict[str, Any]) -> str:
    return f"{name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"


def _safe_parse_tool_args(raw_args: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    try:
        parsed = json.loads(raw_args or "{}")
        return parsed if isinstance(parsed, dict) else {"input": parsed}
    except Exception:
        return {"raw": str(raw_args)}


def _truncate_output(raw: Any, max_chars: int) -> str:
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 24)] + "\n...[truncated]..."


def _check_abort_to_silent(text: str) -> tuple[bool, str | None]:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False, None
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and parsed.get("abort_to_silent") is True:
            return True, str(parsed.get("reason", "verified false alarm"))
    except Exception:
        return False, None
    return False, None


class SkillExecutor:
    """Executes Stage-1.5 read-only Silent Recon and Stage-3 budgeted enterprise tool execution."""

    def __init__(
        self,
        openai_client: OpenAIClient,
        executor_model: str,
        cfg: DialSkillConfig,
        seed: int = 42,
    ) -> None:
        self.openai_client = openai_client
        self.executor_model = executor_model
        self.cfg = cfg
        self.seed = seed

    async def run_silent_recon(
        self,
        *,
        slate: SharedEvidenceSlate,
        tool_env: ToolEnvironment | None,
    ) -> ReconResult:
        """Stage 1.5: Execute at most B_recon=3 read-only enterprise lookups before Stage 2 gating."""
        result = ReconResult()
        if (
            slate.env_mode != "pagemantle"
            or not self.cfg.recon.enabled_on_pagemantle
            or tool_env is None
            or self.cfg.recon.max_calls <= 0
        ):
            return result

        max_p_chg = max(
            (ann.p_decision_change for ann in slate.triage_annotations.values()),
            default=1.0 if self.cfg.ablation.disable_triage else 0.0,
        )
        if max_p_chg < self.cfg.recon.theta_recon:
            return result

        top_skill = slate.candidate_skills[0] if slate.candidate_skills else None
        if top_skill is None or top_skill.name not in self.cfg.recon.eligible_skills:
            return result

        tools = tool_env.get_openai_tool_schemas(read_only=True)
        if not tools:
            return result

        plan_text = "\n".join(f"- {step}" for step in top_skill.investigation_plan)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    f"You are running Stage-1.5 Silent Recon (read-only) for candidate skill `{top_skill.name}`.\n"
                    f"Execute up to {self.cfg.recon.max_calls} focused read-only tool calls to verify key facts "
                    "discussed in the chat before deciding whether to intervene.\n"
                    f"Investigation Plan:\n{plan_text}\n"
                    "Do NOT send any chat messages. Stop calling tools once key facts are verified."
                ),
            },
            {"role": "user", "content": slate.canonical_text},
        ]

        recon_budget = self.cfg.recon.max_calls
        while (
            result.tool_calls_used < recon_budget
            and result.model_elapsed_sec < self.cfg.recon.soft_cap_sec
        ):
            llm_res = await self.openai_client.complete_text(
                model=self.executor_model,
                messages=messages,
                temperature=0.0,
                seed=self.seed,
                tools=tools,
                tool_choice="auto",
            )
            result.model_elapsed_sec += llm_res.elapsed_sec
            result.cost_usd += llm_res.cost_usd

            if not llm_res.tool_calls:
                break

            assistant_tool_calls = [
                {
                    "id": tc.get("id") or f"recon_tc_{i}",
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": (
                            tc["arguments"]
                            if isinstance(tc["arguments"], str)
                            else json.dumps(tc["arguments"])
                        ),
                    },
                }
                for i, tc in enumerate(llm_res.tool_calls)
            ]
            messages.append(
                {
                    "role": "assistant",
                    "content": llm_res.content or "",
                    "tool_calls": assistant_tool_calls,
                }
            )

            for idx, tc in enumerate(llm_res.tool_calls):
                if result.tool_calls_used >= recon_budget:
                    break
                tc_id = tc.get("id") or f"recon_tc_{idx}"
                name = tc["name"]
                args = _safe_parse_tool_args(tc.get("arguments", "{}"))
                call_key = _canonical_call_key(name, args)

                if call_key in result.executed_calls:
                    obs_str = result.executed_calls[call_key]
                else:
                    try:
                        # Tool wall-clock time is excluded from model_elapsed_sec (Invariant #7)
                        raw_obs = tool_env.call_tool(name, args)
                        obs_str = _truncate_output(raw_obs, self.cfg.executor.tool_output_max_chars)
                    except Exception as exc:
                        obs_str = f"TOOL_ERROR: {type(exc).__name__}: {exc}"
                    result.executed_calls[call_key] = obs_str
                    result.tool_calls_used += 1
                    recon_id = f"recon_{result.tool_calls_used}"
                    service = name.split("_")[0] if "_" in name else name
                    result.snippets.append(
                        ReconSnippet(
                            recon_id=recon_id,
                            service=service,
                            query=f"{name}({json.dumps(args, sort_keys=True)})",
                            summary_fact=obs_str[:400],
                        )
                    )

                messages.append({"role": "tool", "tool_call_id": tc_id, "content": obs_str})

        return result

    async def run_execution(
        self,
        *,
        slate: SharedEvidenceSlate,
        gate: GateDecision,
        skill_card: SkillCard | None,
        tool_env: ToolEnvironment | None = None,
        prior_recon: ReconResult | None = None,
        prior_model_elapsed_sec: float = 0.0,
        generate_text: bool = True,
    ) -> ExecutionOutput:
        """Stage 3: Budgeted enterprise tool investigation + channel-shaped response synthesis."""
        out = ExecutionOutput(model_elapsed_sec=prior_model_elapsed_sec)
        if gate.final_tier == InterventionTier.SILENT or not generate_text:
            return out

        latest_msg = slate.selected_messages[-1] if slate.selected_messages else None
        latest_speaker = latest_msg.speaker if latest_msg else "teammate"
        latest_thread_ts = latest_msg.thread_ts if latest_msg else None

        if gate.final_tier == InterventionTier.DM_NUDGE:
            out.target_channel = "DM"
            out.target_user = latest_speaker
        elif gate.final_tier == InterventionTier.THREAD_BRIEF:
            thread_ref = latest_thread_ts or slate.latest_msg_id
            out.target_channel = f"{slate.channel_name}#thread:{thread_ref}"
            out.target_user = latest_speaker
        else:
            out.target_channel = slate.channel_name
            out.target_user = None

        hard_cap = float(self.cfg.executor.hard_time_cap_sec)
        soft_cap = float(self.cfg.executor.soft_time_cap_sec)
        if out.model_elapsed_sec >= hard_cap:
            out.timed_out = True
            return out

        recon_calls = prior_recon.tool_calls_used if prior_recon else 0
        executed_calls: dict[str, str] = dict(prior_recon.executed_calls) if prior_recon else {}
        skill_cap = (
            skill_card.max_tool_calls
            if skill_card is not None
            else self.cfg.executor.default_max_tool_calls
        )
        max_total_tools = min(skill_cap, self.cfg.executor.default_max_tool_calls)
        remaining_tool_budget = max(0, max_total_tools - recon_calls)

        sys_prompt = build_execution_system_prompt(skill_card, gate.final_tier)
        if skill_card is not None and skill_card.investigation_plan and tool_env is not None:
            plan_lines = "\n".join(f"- {s}" for s in skill_card.investigation_plan)
            sys_prompt += f"\n\nInvestigation Plan (max {remaining_tool_budget} more tool calls):\n{plan_lines}"

        user_payload = (
            f"{slate.canonical_text}\n\n"
            f"Gate Rationale:\n"
            f"- Selected Skill: {gate.selected_skill}\n"
            f"- Final Tier: {gate.final_tier.value}\n"
            f"- Margin Delta: {gate.margin_delta:.3f} | Disagreement U: {gate.disagreement_u:.3f}\n"
            f"- Advocate Loss Assessment: {gate.advocate.unmitigated_loss if gate.advocate else 'N/A'}\n"
            f"- Skeptic Caution: {gate.skeptic.disruption_risk if gate.skeptic else 'N/A'}\n"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_payload},
        ]

        tools = (
            tool_env.get_openai_tool_schemas(read_only=False)
            if (tool_env is not None and slate.env_mode == "pagemantle" and remaining_tool_budget > 0)
            else None
        )

        # Iterative budgeted tool loop
        final_content = ""
        while True:
            if out.model_elapsed_sec >= hard_cap:
                out.timed_out = True
                break

            allow_tools = (
                tools is not None
                and out.tool_calls_used < remaining_tool_budget
                and out.model_elapsed_sec < soft_cap
            )
            has_tool_history = any(m.get("role") == "tool" for m in messages)
            if not allow_tools and has_tool_history:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Tool investigation budget is complete. Write your final concise, "
                            "evidence-grounded response now following the channel format (do not return an empty reply)."
                        ),
                    }
                )
            call_tools = tools if (allow_tools or (tools is not None and has_tool_history)) else None
            call_tool_choice = "auto" if allow_tools else ("none" if call_tools is not None else None)
            llm_res = await self.openai_client.complete_text(
                model=self.executor_model,
                messages=messages,
                temperature=self.cfg.executor.temperature,
                seed=self.seed,
                tools=call_tools,
                tool_choice=call_tool_choice,
            )
            out.model_elapsed_sec += llm_res.elapsed_sec
            out.cost_usd += llm_res.cost_usd

            if out.model_elapsed_sec > hard_cap:
                out.timed_out = True
                final_content = llm_res.content or ""
                break

            if not llm_res.tool_calls or not allow_tools or tool_env is None:
                final_content = llm_res.content or ""
                break

            assistant_tool_calls = [
                {
                    "id": tc.get("id") or f"exec_tc_{i}",
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": (
                            tc["arguments"]
                            if isinstance(tc["arguments"], str)
                            else json.dumps(tc["arguments"])
                        ),
                    },
                }
                for i, tc in enumerate(llm_res.tool_calls)
            ]
            messages.append(
                {
                    "role": "assistant",
                    "content": llm_res.content or "",
                    "tool_calls": assistant_tool_calls,
                }
            )

            for idx, tc in enumerate(llm_res.tool_calls):
                tc_id = tc.get("id") or f"exec_tc_{idx}"
                name = tc["name"]
                args = _safe_parse_tool_args(tc.get("arguments", "{}"))
                call_key = _canonical_call_key(name, args)

                if call_key in executed_calls:
                    obs_str = f"[Reused from Silent Recon]\n{executed_calls[call_key]}"
                elif out.tool_calls_used >= remaining_tool_budget:
                    obs_str = "BUDGET_REACHED: Synthesize final response now using gathered evidence."
                else:
                    try:
                        # Tool execution wall time excluded from model_elapsed_sec (Invariant #7)
                        raw_obs = tool_env.call_tool(name, args)
                        obs_str = _truncate_output(raw_obs, self.cfg.executor.tool_output_max_chars)
                    except Exception as exc:
                        obs_str = f"TOOL_ERROR: {type(exc).__name__}: {exc}"
                    executed_calls[call_key] = obs_str
                    out.tool_calls_used += 1
                    out.tool_trace.append({"tool": name, "arguments": args, "output_preview": obs_str[:200]})

                messages.append({"role": "tool", "tool_call_id": tc_id, "content": obs_str})

        # Safeguard: if a model returned empty text after tool calls (e.g., GPT-5.6 stop-without-text),
        # run one clean plain-text synthesis pass grounded on all collected evidence so we never emit "(empty)".
        if not final_content.strip() and not out.timed_out and out.model_elapsed_sec < hard_cap:
            evidence_blocks = []
            for call_k, obs_v in executed_calls.items():
                evidence_blocks.append(f"- {call_k} => {obs_v[:600]}")
            evidence_text = "\n".join(evidence_blocks) if evidence_blocks else "(no additional tool observations)"
            fallback_msgs = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{user_payload}\n\n"
                        f"Collected Tool Observations:\n{evidence_text}\n\n"
                        "Write your final concise, evidence-grounded response now (or exact abort_to_silent JSON if verified false alarm). "
                        "Do NOT return an empty message."
                    ),
                },
            ]
            fb_res = await self.openai_client.complete_text(
                model=self.executor_model,
                messages=fallback_msgs,
                temperature=self.cfg.executor.temperature,
                seed=self.seed,
                tools=None,
                tool_choice=None,
            )
            out.model_elapsed_sec += fb_res.elapsed_sec
            out.cost_usd += fb_res.cost_usd
            final_content = fb_res.content or ""

        aborted, reason = _check_abort_to_silent(final_content)
        if aborted:
            out.aborted_to_silent = True
            out.abort_reason = reason
            out.response_text = None
            out.target_channel = None
            out.target_user = None
        else:
            out.response_text = final_content.strip() or None
        return out
