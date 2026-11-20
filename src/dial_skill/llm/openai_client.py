"""Async OpenAI client with Structured Outputs, retries, USD cost tracking, and SQLite cache."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError
from pydantic import BaseModel
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from dial_skill.config import ModelsConfig, load_models_config
from dial_skill.utils.cache import SQLiteAPICache, compute_prompt_hash
from dial_skill.utils.secrets import load_openai_api_key

logger = logging.getLogger(__name__)

_TRANSIENT_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError, TimeoutError)


class LLMCallResult(BaseModel):
    content: str = ""
    parsed_json: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = []
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    elapsed_sec: float = 0.0
    from_cache: bool = False


@dataclass
class UsageTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    api_calls: int = 0
    cache_hits: int = 0
    model_elapsed_sec: float = 0.0
    per_model_calls: dict[str, int] = field(default_factory=dict)

    def record(self, res: LLMCallResult) -> None:
        if res.from_cache:
            self.cache_hits += 1
        else:
            self.api_calls += 1
            self.prompt_tokens += res.prompt_tokens
            self.completion_tokens += res.completion_tokens
            self.total_cost_usd += res.cost_usd
            self.model_elapsed_sec += res.elapsed_sec
            self.per_model_calls[res.model] = self.per_model_calls.get(res.model, 0) + 1


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse strict JSON text, stripping optional markdown code fences if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"Could not parse JSON object from model response: {text[:200]!r}")


class OpenAIClient:
    """Async OpenAI wrapper enforcing caching, retries, concurrency limits, and USD tracking."""

    def __init__(
        self,
        api_key: str | None = None,
        models_cfg: ModelsConfig | None = None,
        cache: SQLiteAPICache | None = None,
        async_client: Any | None = None,
        max_retries: int = 3,
    ) -> None:
        self.models_cfg = models_cfg or load_models_config()
        self.cache = cache if cache is not None else SQLiteAPICache()
        self._api_key = api_key if api_key is not None else load_openai_api_key(required=False)
        self._client = async_client
        self._max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max(1, self.models_cfg.openai.max_concurrency))
        self.usage = UsageTracker()

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "OpenAI API key not found. Set OPENAI_API_KEY or provide OPENAI_KEY.txt."
                )
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                timeout=float(self.models_cfg.openai.timeout_sec),
            )
        return self._client

    def estimate_cost_usd(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = self.models_cfg.openai.pricing_per_1m_tokens
        rates = pricing.get(model) or pricing.get("default") or {"input": 1.0, "output": 4.0}
        in_cost = (prompt_tokens / 1_000_000.0) * float(rates["input"])
        out_cost = (completion_tokens / 1_000_000.0) * float(rates["output"])
        return round(in_cost + out_cost, 8)

    async def list_models(self) -> list[str]:
        client = self._get_client()
        resp = await client.models.list()
        data = getattr(resp, "data", resp)
        return [getattr(m, "id", str(m)) for m in data]

    async def _call_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        seed: int | None,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any | None,
        use_cache: bool,
    ) -> LLMCallResult:
        cache_payload = {
            "messages": messages,
            "response_format": response_format,
            "tools": tools,
            "tool_choice": tool_choice,
            "max_tokens": max_tokens,
        }
        prompt_hash = compute_prompt_hash(cache_payload)

        if use_cache:
            cached = self.cache.get(
                provider="openai",
                model=model,
                prompt_hash=prompt_hash,
                temperature=temperature,
                seed=seed,
            )
            if cached is not None:
                res = LLMCallResult(
                    content=cached.get("content", ""),
                    parsed_json=cached.get("parsed_json"),
                    tool_calls=cached.get("tool_calls", []),
                    model=model,
                    prompt_tokens=int(cached.get("prompt_tokens", 0)),
                    completion_tokens=int(cached.get("completion_tokens", 0)),
                    cost_usd=0.0,
                    elapsed_sec=0.0,
                    from_cache=True,
                )
                self.usage.record(res)
                return res

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        client = self._get_client()

        async def _invoke() -> Any:
            async for attempt in AsyncRetrying(
                reraise=True,
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
                retry=retry_if_exception_type(_TRANSIENT_ERRORS),
            ):
                with attempt:
                    return await client.chat.completions.create(**kwargs)

        t0 = time.perf_counter()
        async with self._semaphore:
            response = await _invoke()
        elapsed = time.perf_counter() - t0

        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""

        extracted_tool_calls: list[dict[str, Any]] = []
        raw_tcs = getattr(msg, "tool_calls", None) or []
        for tc in raw_tcs:
            fn = getattr(tc, "function", None)
            if fn is not None:
                extracted_tool_calls.append(
                    {
                        "id": getattr(tc, "id", ""),
                        "name": getattr(fn, "name", ""),
                        "arguments": getattr(fn, "arguments", "{}"),
                    }
                )
            elif isinstance(tc, dict):
                extracted_tool_calls.append(tc)

        usage_obj = getattr(response, "usage", None)
        p_tok = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
        c_tok = int(getattr(usage_obj, "completion_tokens", 0) or 0)
        cost = self.estimate_cost_usd(model, p_tok, c_tok)

        parsed_json: dict[str, Any] | None = None
        if response_format is not None and content:
            parsed_json = _extract_json_object(content)

        if use_cache:
            self.cache.set(
                provider="openai",
                model=model,
                prompt_hash=prompt_hash,
                temperature=temperature,
                seed=seed,
                response={
                    "content": content,
                    "parsed_json": parsed_json,
                    "tool_calls": extracted_tool_calls,
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                },
            )

        res = LLMCallResult(
            content=content,
            parsed_json=parsed_json,
            tool_calls=extracted_tool_calls,
            model=model,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            cost_usd=cost,
            elapsed_sec=elapsed,
            from_cache=False,
        )
        self.usage.record(res)
        return res

    async def complete_json(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "structured_output",
        temperature: float = 0.0,
        seed: int | None = 42,
        max_tokens: int | None = None,
        use_cache: bool = True,
    ) -> LLMCallResult:
        """Execute a structured JSON chat completion with strict JSON schema when provided."""
        if json_schema is not None:
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": json_schema,
                },
            }
        else:
            response_format = {"type": "json_object"}
        res = await self._call_chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            response_format=response_format,
            tools=None,
            tool_choice=None,
            use_cache=use_cache,
        )
        if res.parsed_json is None and res.content:
            res.parsed_json = _extract_json_object(res.content)
        return res

    async def complete_text(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        seed: int | None = 42,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        use_cache: bool = True,
    ) -> LLMCallResult:
        """Execute a standard (optionally tool-enabled) chat completion."""
        return await self._call_chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            response_format=None,
            tools=tools,
            tool_choice=tool_choice,
            use_cache=use_cache,
        )
