"""B2: ProACT-Reimpl baseline (breakdown-aware single-perspective gating + static skill routing, binary policy)."""

from __future__ import annotations

import json
from typing import Any, Literal

from dial_skill.baselines.direct_chat import _format_raw_history
from dial_skill.core.executor import ToolEnvironment, _safe_parse_tool_args, _truncate_output
from dial_skill.core.skill_store import SkillStore
from dial_skill.core.triage import estimate_tokens
from dial_skill.llm.openai_client import OpenAIClient
from dial_skill.prompts.templates import (
    PROACT_JSON_SCHEMA,
    PROACT_SYSTEM_PROMPT,
    compute_slate_hash,
)
from dial_skill.schemas import (
    ChatMessage,
    EpisodeResult,
    GateDecision,
    InterventionTier,
    SkillName,
)


class ProACTBaseline:
    """Faithful re-implementation of ProACT (Yang et al.) adapted to When2Speak & PageMantle.

    Key differences vs. DIAL-Skill:
    - Feeds full raw chat history (no System-1 Jev VoI triage or SSES)
    - Single-perspective breakdown detector (no Advocate/Skeptic adversarial margin)
    - Binary intervention policy in main channel (no 4-tier dynamic de-escalation)
    - Static v0 skill routing (no online DB-CSE boundary evolution)
    """

    def __init__(
        self,
        *,
        openai_client: OpenAIClient,
        gate_model: str,
        executor_model: str | None = None,
        static_skill_store: SkillStore | None = None,
        max_tool_calls: int = 7,
        hard_time_cap_sec: float = 180.0,
        soft_time_cap_sec: float = 90.0,
        tool_output_max_chars: int = 3000,
        seed: int = 42,
        method_name: str = "B2_proact",
    ) -> None:
        self.openai_client = openai_client
        self.gate_model = gate_model
        self.executor_model = executor_model or gate_model
        self.static_skill_store = static_skill_store or SkillStore.from_json_file()
        self.max_tool_calls = max_tool_calls
        self.hard_time_cap_sec = hard_time_cap_sec
        self.soft_time_cap_sec = soft_time_cap_sec
        self.tool_output_max_chars = tool_output_max_chars
        self.seed = seed
        self.method_name = method_name

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
    ) -> EpisodeResult:
        raw_text = _format_raw_history(channel_name, messages)
        raw_tokens = estimate_tokens(raw_text)
        raw_hash = compute_slate_hash(raw_text)

        res = await self.openai_client.complete_json(
            model=self.gate_model,
            messages=[
                {"role": "system", "content": PROACT_SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            json_schema=PROACT_JSON_SCHEMA,
            schema_name="b2_proact_breakdown_gate",
            temperature=0.0,
            seed=self.seed,
        )
        cost = res.cost_usd
        model_elapsed = res.elapsed_sec
        data = res.parsed_json or {}

        breakdown_detected = bool(data.get("breakdown_detected", False))
        breakdown_type = str(data.get("breakdown_type", "none"))
        should_intervene = bool(data.get("should_intervene", False))
        target_skill: SkillName = data.get("target_skill", "none")
        rationale = str(data.get("rationale", ""))

        # ProACT uses a binary policy (SILENT vs CHANNEL_INTERVENE, no dynamic de-escalation)
        final_tier = InterventionTier.CHANNEL_INTERVENE if should_intervene else InterventionTier.SILENT
        gate = GateDecision(
            short_circuited=False,
            slate_hash=raw_hash,
            selected_skill=target_skill,
            advocate_score=1.0 if should_intervene else 0.0,
            skeptic_score=0.0 if should_intervene else 1.0,
            margin_delta=0.5 if should_intervene else -0.5,
            disagreement_u=0.0,
            raw_tier=final_tier,
            final_tier=final_tier,
            binary_speak=should_intervene,
        )

        response_text: str | None = None
        tool_calls_used = 0
        timed_out = False

        if should_intervene and generate_text:
            static_card = self.static_skill_store.get(target_skill)
            skill_instructions = (
                f"Static Skill `{static_card.name}`: {static_card.description}\n"
                f"Response style: {static_card.response_contract}"
                if static_card is not None
                else "Provide concise collaborative assistance to resolve the detected breakdown."
            )
            exec_msgs: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        f"You are ProACT intervening in main channel `{channel_name}` to resolve "
                        f"a `{breakdown_type}` collaboration breakdown.\n{skill_instructions}"
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
                                "id": tc.get("id") or f"b2_tc_{i}",
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
                    tc_id = tc.get("id") or f"b2_tc_{i}"
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
            target_channel=channel_name if should_intervene else None,
            tool_calls_used=tool_calls_used,
            model_elapsed_sec=round(model_elapsed, 4),
            timed_out=timed_out,
            prompt_tokens_raw=raw_tokens,
            prompt_tokens_slate=raw_tokens,
            total_cost_usd=round(cost, 8),
            triage_backend_used="none",
            metadata={
                "breakdown_detected": breakdown_detected,
                "breakdown_type": breakdown_type,
                "rationale": rationale,
            },
        )
