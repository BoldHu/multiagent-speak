"""B0: Direct Chat proactive baseline (full raw chat history, optional PageMantle tool loop)."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Literal

from dial_skill.core.executor import ToolEnvironment, _safe_parse_tool_args, _truncate_output
from dial_skill.core.triage import estimate_tokens
from dial_skill.llm.openai_client import OpenAIClient
from dial_skill.prompts.templates import DIRECT_CHAT_SYSTEM_PROMPT, compute_slate_hash
from dial_skill.schemas import (
    ChatMessage,
    EpisodeResult,
    GateDecision,
    InterventionTier,
)


def _format_raw_history(channel_name: str, messages: list[ChatMessage]) -> str:
    lines = [f"Channel: {channel_name}", "Conversation History:"]
    for m in messages:
        lines.append(f"[{m.msg_id}] {m.speaker}: {m.text}")
    return "\n".join(lines)


def _infer_tier_from_direct_reply(text: str) -> InterventionTier:
    stripped = text.strip()
    if not stripped or stripped.upper() == "SILENT" or re.fullmatch(r"[\*\`\"\']?SILENT[\*\`\"\']?\.?", stripped, re.IGNORECASE):
        return InterventionTier.SILENT
    head = stripped[:80].lower()
    if any(tok in head for tok in ("[dm]", "dm:", "direct message", "dm_nudge")):
        return InterventionTier.DM_NUDGE
    if any(tok in head for tok in ("[thread]", "thread:", "thread_brief")):
        return InterventionTier.THREAD_BRIEF
    return InterventionTier.CHANNEL_INTERVENE


class DirectChatBaseline:
    """B0: Direct Chat baseline feeding raw conversation history directly to the LLM."""

    def __init__(
        self,
        *,
        openai_client: OpenAIClient,
        model: str,
        max_tool_calls: int = 7,
        hard_time_cap_sec: float = 180.0,
        soft_time_cap_sec: float = 90.0,
        tool_output_max_chars: int = 3000,
        temperature: float = 0.0,
        seed: int = 42,
        method_name: str = "B0_direct_chat",
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.max_tool_calls = max_tool_calls
        self.hard_time_cap_sec = hard_time_cap_sec
        self.soft_time_cap_sec = soft_time_cap_sec
        self.tool_output_max_chars = tool_output_max_chars
        self.temperature = temperature
        self.seed = seed
        self.method_name = method_name

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
        extra_memory_block: str = "",
    ) -> EpisodeResult:
        raw_text = _format_raw_history(channel_name, messages)
        raw_tokens = estimate_tokens(raw_text)
        sys_prompt = DIRECT_CHAT_SYSTEM_PROMPT
        if extra_memory_block:
            sys_prompt = f"{sys_prompt}\n\n{extra_memory_block}"

        chat_msgs: list[dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": raw_text},
        ]
        tools = (
            tool_env.get_openai_tool_schemas(read_only=False)
            if (tool_env is not None and benchmark != "when2speak")
            else None
        )

        model_elapsed = 0.0
        cost = 0.0
        uncached_cost = 0.0
        tool_calls_used = 0
        timed_out = False
        tool_trace: list[dict[str, Any]] = []
        final_content = ""

        while True:
            if model_elapsed >= self.hard_time_cap_sec:
                timed_out = True
                break
            allow_tools = (
                tools is not None
                and tool_calls_used < self.max_tool_calls
                and model_elapsed < self.soft_time_cap_sec
            )
            res = await self.openai_client.complete_text(
                model=self.model,
                messages=chat_msgs,
                temperature=self.temperature,
                seed=self.seed,
                tools=tools if allow_tools else None,
                tool_choice="auto" if allow_tools else None,
            )
            model_elapsed += res.elapsed_sec
            cost += res.cost_usd
            uncached_cost += res.uncached_cost_usd
            if model_elapsed > self.hard_time_cap_sec:
                timed_out = True
                final_content = res.content or ""
                break
            if not res.tool_calls or not allow_tools or tool_env is None:
                final_content = res.content or ""
                break

            chat_msgs.append(
                {
                    "role": "assistant",
                    "content": res.content or "",
                    "tool_calls": [
                        {
                            "id": tc.get("id") or f"b0_tc_{i}",
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
                        for i, tc in enumerate(res.tool_calls)
                    ],
                }
            )
            for i, tc in enumerate(res.tool_calls):
                tc_id = tc.get("id") or f"b0_tc_{i}"
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
                    tool_trace.append({"tool": name, "arguments": args})
                chat_msgs.append({"role": "tool", "tool_call_id": tc_id, "content": obs})

        tier = _infer_tier_from_direct_reply(final_content)
        speak = tier != InterventionTier.SILENT
        raw_hash = compute_slate_hash(raw_text)
        gate = GateDecision(
            short_circuited=False,
            slate_hash=raw_hash,
            selected_skill="none",
            advocate_score=1.0 if speak else 0.0,
            skeptic_score=0.0 if speak else 1.0,
            margin_delta=1.0 if speak else -1.0,
            disagreement_u=0.0,
            raw_tier=tier,
            final_tier=tier,
            binary_speak=speak,
        )

        return EpisodeResult(
            sample_id=sample_id,
            benchmark=benchmark,
            method=self.method_name,
            model_id=self.model,
            gate=gate,
            response_text=final_content.strip() if speak else None,
            target_channel=channel_name if speak else None,
            target_user=messages[-1].speaker if (speak and tier == InterventionTier.DM_NUDGE) else None,
            tool_calls_used=tool_calls_used,
            model_elapsed_sec=round(model_elapsed, 4),
            timed_out=timed_out,
            prompt_tokens_raw=raw_tokens,
            prompt_tokens_slate=raw_tokens,
            total_cost_usd=round(cost, 8),
            uncached_cost_usd=round(max(cost, uncached_cost), 8),
            triage_backend_used="none",
            metadata={"tool_trace": tool_trace, "extra_memory_tokens": estimate_tokens(extra_memory_block)},
        )


class DirectSelfConsistencyBaseline:
    """Compute-matched Direct Self-Consistency baseline (`Direct-SC@K`, default K=3).

    Samples K independent direct trajectories at `temperature > 0` with distinct seeds,
    uses continuous speak vote fraction `k_speak / K` as calibrated confidence for AUROC
    and validation threshold calibration, and takes majority vote for intervention.
    """

    def __init__(
        self,
        *,
        openai_client: OpenAIClient,
        model: str,
        n_samples: int = 3,
        temperature: float = 0.7,
        vote_threshold: float = 0.50,
        seed: int = 42,
        method_name: str | None = None,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.n_samples = max(2, int(n_samples))
        self.temperature = temperature
        self.vote_threshold = vote_threshold
        self.seed = seed
        self.method_name = method_name or f"B0_direct_sc{self.n_samples}"

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
        del generate_text, tool_env
        raw_text = _format_raw_history(channel_name, messages)
        raw_tokens = estimate_tokens(raw_text)
        raw_hash = compute_slate_hash(raw_text)
        chat_msgs = [
            {"role": "system", "content": DIRECT_CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": raw_text},
        ]

        results = await asyncio.gather(
            *(
                self.openai_client.complete_text(
                    model=self.model,
                    messages=chat_msgs,
                    temperature=self.temperature,
                    seed=self.seed + idx * 101,
                )
                for idx in range(self.n_samples)
            )
        )

        speak_candidates: list[tuple[InterventionTier, str]] = []
        cost = 0.0
        uncached_cost = 0.0
        elapsed = 0.0
        for res in results:
            cost += res.cost_usd
            uncached_cost += res.uncached_cost_usd
            elapsed = max(elapsed, res.elapsed_sec)
            tier_k = _infer_tier_from_direct_reply(res.content or "")
            if tier_k != InterventionTier.SILENT:
                speak_candidates.append((tier_k, (res.content or "").strip()))

        vote_prob = len(speak_candidates) / float(self.n_samples)
        speak = vote_prob >= self.vote_threshold
        if speak and speak_candidates:
            # Pick median-length speaking sample for representative conciseness
            speak_candidates.sort(key=lambda pair: len(pair[1]))
            chosen_tier, chosen_text = speak_candidates[len(speak_candidates) // 2]
        else:
            chosen_tier = InterventionTier.CHANNEL_INTERVENE if speak else InterventionTier.SILENT
            chosen_text = None

        gate = GateDecision(
            short_circuited=False,
            slate_hash=raw_hash,
            selected_skill="none",
            advocate_score=round(vote_prob, 6),
            skeptic_score=round(1.0 - vote_prob, 6),
            margin_delta=round(vote_prob - self.vote_threshold, 6),
            disagreement_u=round(min(vote_prob, 1.0 - vote_prob), 6),
            raw_tier=chosen_tier,
            final_tier=chosen_tier,
            binary_speak=speak,
        )
        return EpisodeResult(
            sample_id=sample_id,
            benchmark=benchmark,
            method=self.method_name,
            model_id=self.model,
            gate=gate,
            response_text=chosen_text if speak else None,
            target_channel=channel_name if speak else None,
            target_user=messages[-1].speaker if (speak and chosen_tier == InterventionTier.DM_NUDGE) else None,
            tool_calls_used=0,
            model_elapsed_sec=round(elapsed, 4),
            timed_out=False,
            prompt_tokens_raw=raw_tokens,
            prompt_tokens_slate=raw_tokens,
            total_cost_usd=round(cost, 8),
            uncached_cost_usd=round(max(cost, uncached_cost), 8),
            triage_backend_used="none",
            metadata={"n_samples": self.n_samples, "speak_votes": len(speak_candidates), "vote_prob": round(vote_prob, 4)},
        )
