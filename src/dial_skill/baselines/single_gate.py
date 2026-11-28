"""B1: Single-Model Confidence Gate baseline (LLM single score OR Jev-only score)."""

from __future__ import annotations

import json
from typing import Any, Literal

from dial_skill.baselines.direct_chat import _format_raw_history
from dial_skill.core.executor import ToolEnvironment, _safe_parse_tool_args, _truncate_output
from dial_skill.core.triage import estimate_tokens
from dial_skill.llm.jev_client import TriageClient
from dial_skill.llm.openai_client import OpenAIClient
from dial_skill.prompts.templates import (
    SINGLE_GATE_JSON_SCHEMA,
    SINGLE_GATE_SYSTEM_PROMPT,
    compute_slate_hash,
)
from dial_skill.schemas import (
    ChatMessage,
    EpisodeResult,
    GateDecision,
    InterventionTier,
)


class SingleGateBaseline:
    """B1: Single-perspective confidence threshold gate (binary speak vs. silent)."""

    def __init__(
        self,
        *,
        openai_client: OpenAIClient,
        gate_model: str,
        executor_model: str | None = None,
        triage_client: TriageClient | None = None,
        gate_source: Literal["llm", "jev"] = "llm",
        threshold: float = 0.50,
        max_tool_calls: int = 7,
        hard_time_cap_sec: float = 180.0,
        soft_time_cap_sec: float = 90.0,
        tool_output_max_chars: int = 3000,
        seed: int = 42,
        method_name: str | None = None,
    ) -> None:
        self.openai_client = openai_client
        self.gate_model = gate_model
        self.executor_model = executor_model or gate_model
        self.triage_client = triage_client
        self.gate_source = gate_source
        self.threshold = threshold
        self.max_tool_calls = max_tool_calls
        self.hard_time_cap_sec = hard_time_cap_sec
        self.soft_time_cap_sec = soft_time_cap_sec
        self.tool_output_max_chars = tool_output_max_chars
        self.seed = seed
        self.method_name = method_name or f"B1_single_gate_{gate_source}"

    async def run_episode(
        self,
        *,
        sample_id: str,
        conversation_id: str,
        channel_name: str,
        messages: list[ChatMessage],
        benchmark: Literal[
            "when2speak",
            "pagemantle_official",
            "pagemantle_ext",
            "pagemantle_synthetic_sanity",
        ],
        tool_env: ToolEnvironment | None = None,
        generate_text: bool = True,
    ) -> EpisodeResult:
        raw_text = _format_raw_history(channel_name, messages)
        raw_tokens = estimate_tokens(raw_text)
        raw_hash = compute_slate_hash(raw_text)

        cost = 0.0
        uncached_cost = 0.0
        model_elapsed = 0.0
        rationale = ""
        triage_used = "none"

        if self.gate_source == "jev":
            if self.triage_client is None:
                raise ValueError("SingleGateBaseline(gate_source='jev') requires triage_client.")
            if hasattr(self.triage_client, "score_message_with_cost"):
                item, call_cost, call_uncached = await self.triage_client.score_message_with_cost(
                    channel_name=channel_name,
                    prior_messages=messages[:-1],
                    target_message=messages[-1],
                )
            else:
                item = await self.triage_client.score_message(
                    channel_name=channel_name,
                    prior_messages=messages[:-1],
                    target_message=messages[-1],
                )
                call_cost = 0.0
                call_uncached = 0.0
            cost += call_cost
            uncached_cost += max(call_cost, call_uncached)
            speak_prob = float(item.p_decision_change * item.stance_probs.get("need_signal", 0.0))
            rationale = f"jev_stance={item.stance}, p_chg={item.p_decision_change:.2f}"
            triage_used = item.backend
        else:
            res = await self.openai_client.complete_json(
                model=self.gate_model,
                messages=[
                    {"role": "system", "content": SINGLE_GATE_SYSTEM_PROMPT},
                    {"role": "user", "content": raw_text},
                ],
                json_schema=SINGLE_GATE_JSON_SCHEMA,
                schema_name="b1_single_gate",
                temperature=0.0,
                seed=self.seed,
            )
            cost += res.cost_usd
            uncached_cost += res.uncached_cost_usd
            model_elapsed += res.elapsed_sec
            data = res.parsed_json or {}
            speak_prob = max(0.0, min(1.0, float(data.get("speak_probability", 0.0))))
            rationale = str(data.get("rationale", ""))

        speak = speak_prob >= self.threshold
        tier = InterventionTier.CHANNEL_INTERVENE if speak else InterventionTier.SILENT
        gate = GateDecision(
            short_circuited=False,
            slate_hash=raw_hash,
            selected_skill="none",
            advocate_score=round(speak_prob, 6),
            skeptic_score=round(1.0 - speak_prob, 6),
            margin_delta=round(speak_prob - self.threshold, 6),
            disagreement_u=0.0,
            raw_tier=tier,
            final_tier=tier,
            binary_speak=speak,
        )

        response_text: str | None = None
        tool_calls_used = 0
        timed_out = False

        if speak and generate_text:
            exec_msgs: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You are a proactive enterprise assistant. Use available tools if needed "
                        "and write a concise, accurate message helping the team."
                    ),
                },
                {"role": "user", "content": raw_text},
            ]
            tools = (
                tool_env.get_openai_tool_schemas(read_only=False)
                if (tool_env is not None and benchmark != "when2speak")
                else None
            )
            while True:
                if model_elapsed >= self.hard_time_cap_sec:
                    timed_out = True
                    break
                allow_tools = (
                    tools is not None
                    and tool_calls_used < self.max_tool_calls
                    and model_elapsed < self.soft_time_cap_sec
                )
                llm_res = await self.openai_client.complete_text(
                    model=self.executor_model,
                    messages=exec_msgs,
                    temperature=0.2,
                    seed=self.seed,
                    tools=tools if allow_tools else None,
                    tool_choice="auto" if allow_tools else None,
                )
                model_elapsed += llm_res.elapsed_sec
                cost += llm_res.cost_usd
                uncached_cost += llm_res.uncached_cost_usd
                if model_elapsed > self.hard_time_cap_sec:
                    timed_out = True
                    response_text = llm_res.content.strip() or None
                    break
                if not llm_res.tool_calls or not allow_tools or tool_env is None:
                    response_text = llm_res.content.strip() or None
                    break

                exec_msgs.append(
                    {
                        "role": "assistant",
                        "content": llm_res.content or "",
                        "tool_calls": [
                            {
                                "id": tc.get("id") or f"b1_tc_{i}",
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
                        ],
                    }
                )
                for i, tc in enumerate(llm_res.tool_calls):
                    tc_id = tc.get("id") or f"b1_tc_{i}"
                    name = tc["name"]
                    args = _safe_parse_tool_args(tc.get("arguments", "{}"))
                    if tool_calls_used >= self.max_tool_calls:
                        obs = "BUDGET_REACHED: Respond now."
                    else:
                        try:
                            obs = _truncate_output(tool_env.call_tool(name, args), self.tool_output_max_chars)
                        except Exception as exc:
                            obs = f"TOOL_ERROR: {exc}"
                        tool_calls_used += 1
                    exec_msgs.append({"role": "tool", "tool_call_id": tc_id, "content": obs})

        return EpisodeResult(
            sample_id=sample_id,
            benchmark=benchmark,
            method=self.method_name,
            model_id=self.gate_model,
            gate=gate,
            response_text=response_text,
            target_channel=channel_name if speak else None,
            tool_calls_used=tool_calls_used,
            model_elapsed_sec=round(model_elapsed, 4),
            timed_out=timed_out,
            prompt_tokens_raw=raw_tokens,
            prompt_tokens_slate=raw_tokens,
            total_cost_usd=round(cost, 8),
            uncached_cost_usd=round(max(cost, uncached_cost), 8),
            triage_backend_used=triage_used,
            metadata={"rationale": rationale, "speak_probability": speak_prob},
        )
