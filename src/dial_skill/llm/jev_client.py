"""TypeSafe Jev System-1 triage client with graceful fallback to OpenAI Luna (Invariant #5)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

import httpx

from dial_skill.config import ModelsConfig, load_models_config
from dial_skill.llm.openai_client import OpenAIClient
from dial_skill.schemas import (
    ACTIVE_SKILLS,
    STANCE_LABELS,
    ChatMessage,
    SkillName,
    StanceLabel,
    TriageItem,
)
from dial_skill.utils.cache import SQLiteAPICache, compute_prompt_hash
from dial_skill.utils.secrets import load_jev_api_key

logger = logging.getLogger(__name__)

ALL_SKILL_NAMES: tuple[SkillName, ...] = (*ACTIVE_SKILLS, "none")

OPENAI_TRIAGE_SYSTEM_PROMPT = (
    "You are a fast, calibrated System-1 decision-change triage classifier monitoring an enterprise group chat.\n"
    "Do NOT generate conversational replies. Evaluate ONLY whether the target message could change an AI "
    "teammate's intervention decision (speak vs. wait vs. de-escalate scope) or skill choice.\n"
    "Be calibrated: routine chatter should get p_decision_change < 0.20; clear resolutions/deferrals should "
    "get HIGH p_decision_change with stance='defer_or_resolved_signal' so downstream judges know NOT to interrupt!\n"
    "Return strictly valid JSON matching the schema."
)

OPENAI_TRIAGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "p_decision_change",
        "stance",
        "stance_probs",
        "skill_hint",
        "skill_probs",
        "urgency_score",
    ],
    "properties": {
        "p_decision_change": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "stance": {"type": "string", "enum": list(STANCE_LABELS)},
        "stance_probs": {
            "type": "object",
            "additionalProperties": False,
            "required": list(STANCE_LABELS),
            "properties": {
                k: {"type": "number", "minimum": 0.0, "maximum": 1.0} for k in STANCE_LABELS
            },
        },
        "skill_hint": {"type": "string", "enum": list(ALL_SKILL_NAMES)},
        "skill_probs": {
            "type": "object",
            "additionalProperties": False,
            "required": list(ALL_SKILL_NAMES),
            "properties": {
                k: {"type": "number", "minimum": 0.0, "maximum": 1.0} for k in ALL_SKILL_NAMES
            },
        },
        "urgency_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

_JEV_QUESTION_SPEC: dict[str, dict[str, Any]] = {
    "decision_change": {
        "type": "noul",
        "instructions": (
            "Does the latest target message introduce new information, an unresolved question, "
            "a factual discrepancy, a technical/business risk, a coordination bottleneck, OR "
            "an explicit resolution/deferral/social boundary that could change whether an AI "
            "teammate should intervene or which skill it should use?"
        ),
    },
    "stance": {
        "type": "choice",
        "instructions": (
            "Classify how the latest target message shifts the AI's intervention decision."
        ),
        "criteria": {
            "need_signal": (
                "Signals an unresolved problem, missing fact, conflicting data, failure, "
                "stalled decision, coordination gap, or policy/access risk."
            ),
            "defer_or_resolved_signal": (
                "Signals that a teammate already answered accurately, took clear ownership, "
                "explicitly deferred the topic, or is smoothly resolving it without AI help."
            ),
            "scope_risk_signal": (
                "Signals sensitivity, privacy/permission concern, off-topic channel risk, "
                "or high disruption risk if an AI speaks publicly in the main channel."
            ),
            "background_chatter": (
                "Routine acknowledgment, greeting, filler, or normal progress that does not "
                "meaningfully alter whether the AI should intervene."
            ),
        },
    },
    "skill_hint": {
        "type": "choice",
        "instructions": (
            "If an AI teammate were to help based on this message, which single enterprise "
            "skill best matches the underlying need? Choose 'none' if no skill applies."
        ),
        "criteria": {
            "find": "Locate a specific fact, document, owner, past decision, or status.",
            "explain": "Clarify how a process works, or define a term, rule, or internal policy.",
            "analyze": "Combine or compare cross-source data to surface trends, shifts, or metrics.",
            "check": "Verify whether a stated claim, number, or assumption matches source evidence.",
            "diagnose": "Isolate the root cause of a technical, billing, or operational failure.",
            "decide": "Weigh trade-offs, prioritize options, and recommend a reasoned choice.",
            "coordinate": "Clarify ownership, unblock handoffs, or align colleagues across teams.",
            "guard": "Flag data access, permission, security, or compliance policy violations.",
            "none": "No enterprise assistance skill is warranted by this message.",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is potential AI intervention based on this message?",
        "criteria": [
            "none: no intervention warranted",
            "low_async: minor issue suitable for a quiet private DM or async follow-up",
            "medium_thread: concrete issue suitable for a brief threaded reply",
            "high_immediate: severe conflict, costly mistake, or urgent blocker requiring main-channel alert",
        ],
    },
}


def build_jev_triage_questions() -> dict[str, Any]:
    """Build TypeSafe SDK question objects (Choice/Noul/Score) when SDK is installed, else dict spec."""
    try:
        from typesafe_sdk import Choice, Noul, Score  # type: ignore[import-not-found]

        return {
            "decision_change": Noul(instructions=_JEV_QUESTION_SPEC["decision_change"]["instructions"]),
            "stance": Choice(
                instructions=_JEV_QUESTION_SPEC["stance"]["instructions"],
                criteria=_JEV_QUESTION_SPEC["stance"]["criteria"],
            ),
            "skill_hint": Choice(
                instructions=_JEV_QUESTION_SPEC["skill_hint"]["instructions"],
                criteria=_JEV_QUESTION_SPEC["skill_hint"]["criteria"],
            ),
            "urgency": Score(
                instructions=_JEV_QUESTION_SPEC["urgency"]["instructions"],
                criteria=_JEV_QUESTION_SPEC["urgency"]["criteria"],
            ),
        }
    except Exception:
        return dict(_JEV_QUESTION_SPEC)


def build_jev_state_payload(
    channel_name: str,
    prior_messages: list[ChatMessage],
    target_message: ChatMessage,
) -> dict[str, Any]:
    """Construct the exact Jev state payload specified in docs/PROMPT_SYSTEM.md P1-A."""
    window = prior_messages[-2:]
    return {
        "channel": channel_name,
        "context_window": [
            {"msg_id": m.msg_id, "speaker": m.speaker, "text": m.text} for m in window
        ],
        "target_message": {
            "msg_id": target_message.msg_id,
            "speaker": target_message.speaker,
            "timestamp": target_message.timestamp,
            "text": target_message.text,
        },
    }


def build_jev_cache_key_payload(
    channel_name: str,
    prior_messages: list[ChatMessage],
    target_message: ChatMessage,
) -> dict[str, Any]:
    """Content-only cache payload stripping synthetic msg_id so sliding windows hit cache across turns."""
    window = prior_messages[-2:]
    return {
        "channel": channel_name,
        "context_window": [{"speaker": m.speaker, "text": m.text} for m in window],
        "target_message": {
            "speaker": target_message.speaker,
            "timestamp": target_message.timestamp,
            "text": target_message.text,
        },
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_distribution(
    raw_probs: dict[str, Any] | None,
    allowed_keys: tuple[str, ...],
    fallback_top: str,
) -> dict[str, float]:
    out = {k: 0.0 for k in allowed_keys}
    if isinstance(raw_probs, dict):
        for k in allowed_keys:
            if k in raw_probs and raw_probs[k] is not None:
                out[k] = max(0.0, float(raw_probs[k]))
    total = sum(out.values())
    if total <= 0.0:
        key = fallback_top if fallback_top in out else allowed_keys[-1]
        out[key] = 1.0
        return out
    return {k: round(v / total, 6) for k, v in out.items()}


def _extract_attr_or_key(obj: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            return obj[key]
        if hasattr(obj, key):
            return getattr(obj, key)
    return default


def parse_jev_response_to_triage_item(
    raw_resp: Any,
    msg_id: str,
    backend: Literal["jev", "openai_fallback"] = "jev",
) -> TriageItem:
    """Parse TypeSafe SDK response (nouls/choices/scores), REST answers dict, or OpenAI fallback JSON."""
    nouls = _extract_attr_or_key(raw_resp, "nouls", default=None)
    choices = _extract_attr_or_key(raw_resp, "choices", default=None)
    scores = _extract_attr_or_key(raw_resp, "scores", default=None)
    answers = _extract_attr_or_key(raw_resp, "answers", "results", "output", default=raw_resp)

    # 1. decision_change (Noul)
    dc_field = _extract_attr_or_key(nouls, "decision_change", default=None) if nouls is not None else None
    if dc_field is None:
        dc_field = _extract_attr_or_key(answers, "decision_change", default=answers)
    p_chg = _extract_attr_or_key(
        dc_field, "noul", "probability", "prob", "score", "value", "p_decision_change", default=0.5
    )
    if isinstance(p_chg, bool):
        p_chg = 1.0 if p_chg else 0.0
    p_chg_f = _clamp01(float(p_chg))

    # 2. stance (Choice)
    stance_field = _extract_attr_or_key(choices, "stance", default=None) if choices is not None else None
    if stance_field is None:
        stance_field = _extract_attr_or_key(answers, "stance", default={})
    raw_stance_probs = _extract_attr_or_key(
        stance_field, "probabilities", "probs", "distribution", "scores", default=None
    )
    stance_label = _extract_attr_or_key(stance_field, "choice", "label", "value", "answer", default=None)
    if isinstance(stance_field, str):
        stance_label = stance_field
    if raw_stance_probs is None and isinstance(answers, dict) and "stance_probs" in answers:
        raw_stance_probs = answers["stance_probs"]
    stance_probs = _normalize_distribution(
        raw_stance_probs,
        STANCE_LABELS,
        fallback_top=str(stance_label or "background_chatter"),
    )
    if stance_label not in STANCE_LABELS:
        stance_label = max(stance_probs.items(), key=lambda kv: kv[1])[0]

    # 3. skill_hint (Choice)
    skill_field = _extract_attr_or_key(choices, "skill_hint", default=None) if choices is not None else None
    if skill_field is None:
        skill_field = _extract_attr_or_key(answers, "skill_hint", default={})
    raw_skill_probs = _extract_attr_or_key(
        skill_field, "probabilities", "probs", "distribution", "scores", default=None
    )
    skill_label = _extract_attr_or_key(skill_field, "choice", "label", "value", "answer", default=None)
    if isinstance(skill_field, str):
        skill_label = skill_field
    if raw_skill_probs is None and isinstance(answers, dict) and "skill_probs" in answers:
        raw_skill_probs = answers["skill_probs"]
    skill_probs = _normalize_distribution(
        raw_skill_probs,
        ALL_SKILL_NAMES,
        fallback_top=str(skill_label or "none"),
    )
    if skill_label not in ALL_SKILL_NAMES:
        skill_label = max(skill_probs.items(), key=lambda kv: kv[1])[0]

    # 4. urgency (Score)
    urg_field = _extract_attr_or_key(scores, "urgency", default=None) if scores is not None else None
    if urg_field is None:
        urg_field = _extract_attr_or_key(answers, "urgency", "urgency_score", default=0.0)
    urg_val = (
        urg_field
        if isinstance(urg_field, (int, float))
        else _extract_attr_or_key(urg_field, "score", "value", "probability", default=0.0)
    )
    urg_num = float(urg_val or 0.0)
    if urg_num > 1.0:
        # Normalize rubric index scales (e.g. 0..3 for 4 urgency criteria) onto [0, 1]
        legend = _extract_attr_or_key(urg_field, "legend", default=None)
        max_level = max(1, (len(legend) - 1) if isinstance(legend, (list, dict)) and len(legend) > 1 else 3)
        urg_num = urg_num / float(max_level)
    urg_f = _clamp01(urg_num)

    # 5. confidence (optional calibrated confidence)
    conf_val = _extract_attr_or_key(
        stance_field, "confidence", default=_extract_attr_or_key(dc_field, "confidence", default=None)
    )
    confidence = _clamp01(float(conf_val)) if conf_val is not None else max(stance_probs.values())

    return TriageItem(
        msg_id=msg_id,
        backend=backend,
        p_decision_change=p_chg_f,
        stance=stance_label,  # type: ignore[arg-type]
        stance_probs=stance_probs,  # type: ignore[arg-type]
        skill_hint=skill_label,  # type: ignore[arg-type]
        skill_probs=skill_probs,  # type: ignore[arg-type]
        urgency_score=urg_f,
        confidence=confidence,
    )


class OpenAILunaTriageFallback:
    """OpenAI Luna structured triage fallback returning the exact same TriageItem schema."""

    def __init__(
        self,
        openai_client: OpenAIClient,
        model: str,
        seed: int = 42,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.seed = seed
        self.last_uncached_cost_usd: float = 0.0

    async def score_message(
        self,
        channel_name: str,
        prior_messages: list[ChatMessage],
        target_message: ChatMessage,
    ) -> tuple[TriageItem, float]:
        prior_window = prior_messages[-2:]
        if prior_window:
            prior_fmt = "\n".join(f"[{m.msg_id}] {m.speaker}: {m.text}" for m in prior_window)
        else:
            prior_fmt = "(none)"

        user_prompt = (
            f"Channel: {channel_name}\n"
            f"Prior 2 messages:\n{prior_fmt}\n\n"
            f"Target message to score:\n"
            f"[{target_message.msg_id}] {target_message.speaker} ({target_message.timestamp}): {target_message.text}"
        )
        res = await self.openai_client.complete_json(
            model=self.model,
            messages=[
                {"role": "system", "content": OPENAI_TRIAGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            json_schema=OPENAI_TRIAGE_JSON_SCHEMA,
            schema_name="dial_skill_triage_item",
            temperature=0.0,
            seed=self.seed,
        )
        data = res.parsed_json or {}
        item = parse_jev_response_to_triage_item(
            data, msg_id=target_message.msg_id, backend="openai_fallback"
        )
        self.last_uncached_cost_usd = res.uncached_cost_usd
        return item, res.cost_usd


class TriageClient:
    """Unified Stage-1 Triage Client: TypeSafe Jev primary + graceful OpenAI Luna fallback."""

    def __init__(
        self,
        models_cfg: ModelsConfig | None = None,
        cache: SQLiteAPICache | None = None,
        openai_client: OpenAIClient | None = None,
        luna_model: str | None = None,
        jev_api_key: str | None = None,
        backend: Literal["jev", "openai_fallback"] | None = None,
        sdk_client: Any | None = None,
        http_client: httpx.AsyncClient | None = None,
        seed: int = 42,
    ) -> None:
        self.models_cfg = models_cfg or load_models_config()
        self.cache = cache if cache is not None else SQLiteAPICache()
        self.backend: Literal["jev", "openai_fallback"] = backend or self.models_cfg.roles.triage_backend
        self.jev_api_key = (
            jev_api_key if jev_api_key is not None else load_jev_api_key(required=False)
        )
        self.seed = seed
        self._sdk_client = sdk_client
        self._http_client = http_client

        resolved_luna = luna_model or self.models_cfg.openai.luna_candidates[0]
        self._openai_client = openai_client
        self._luna_model = resolved_luna
        self._fallback_client = (
            OpenAILunaTriageFallback(openai_client, resolved_luna, seed=seed)
            if openai_client is not None
            else None
        )

        # Runtime telemetry (Invariant #5)
        self.jev_calls: int = 0
        self.fallback_count: int = 0
        self.cache_hits: int = 0
        self.total_cost_usd: float = 0.0
        self.uncached_cost_usd: float = 0.0
        self.fallback_reasons: list[str] = []
        self._jev_circuit_open_reason: str | None = None
        self._consecutive_jev_failures: int = 0

    def _get_fallback(self) -> OpenAILunaTriageFallback:
        if self._fallback_client is None:
            client = self._openai_client or OpenAIClient(models_cfg=self.models_cfg, cache=self.cache)
            self._openai_client = client
            self._fallback_client = OpenAILunaTriageFallback(client, self._luna_model, seed=self.seed)
        return self._fallback_client

    def _build_sdk_client(self) -> Any | None:
        """Instantiate AsyncTypeSafeClient (preferred) or TypeSafeClient with explicit timeout & short retry."""
        if not self.jev_api_key:
            return None
        try:
            import inspect
            import typesafe_sdk  # type: ignore[import-not-found]
        except Exception:
            return None

        client_cls = getattr(typesafe_sdk, "AsyncTypeSafeClient", None) or getattr(
            typesafe_sdk, "TypeSafeClient", None
        )
        if client_cls is None:
            return None

        timeout_sec = float(self.models_cfg.jev.timeout_sec)
        kwargs: dict[str, Any] = {"api_key": self.jev_api_key}
        try:
            sig = inspect.signature(client_cls.__init__)
            params = sig.parameters
            if "base_url" in params:
                kwargs["base_url"] = self.models_cfg.jev.base_url
            if "timeout" in params:
                kwargs["timeout"] = timeout_sec
            retry_cls = getattr(typesafe_sdk, "RetryPolicy", None)
            if "retry" in params and retry_cls is not None:
                try:
                    kwargs["retry"] = retry_cls(max_retries=1, backoff_max=2.0)
                except Exception:
                    pass
        except Exception:
            kwargs["base_url"] = self.models_cfg.jev.base_url

        try:
            return client_cls(**kwargs)
        except TypeError:
            return client_cls(api_key=self.jev_api_key)

    async def _invoke_jev_backend(
        self,
        model: str,
        state_payload: dict[str, Any],
    ) -> Any:
        """Invoke Jev via injected/installed typesafe_sdk OR direct POST /v1/systemone with hard timeout."""
        questions = build_jev_triage_questions()
        timeout_sec = float(self.models_cfg.jev.timeout_sec)

        # Path A: injected or installed TypeSafe SDK client
        sdk = self._sdk_client
        if sdk is None and self._http_client is None:
            sdk = self._build_sdk_client()
            self._sdk_client = sdk

        if sdk is not None:
            fn = getattr(sdk, "system_one", None) or getattr(sdk, "systemone", None)
            if fn is None:
                raise RuntimeError("TypeSafe SDK client does not expose system_one method")

            async def _call_sdk(include_model: bool) -> Any:
                call_kw: dict[str, Any] = {"state": state_payload, "questions": questions}
                if include_model:
                    call_kw["model"] = model
                if asyncio.iscoroutinefunction(fn):
                    return await fn(**call_kw)
                return await asyncio.to_thread(fn, **call_kw)

            try:
                raw = await asyncio.wait_for(_call_sdk(True), timeout=timeout_sec + 2.0)
            except TypeError:
                raw = await asyncio.wait_for(_call_sdk(False), timeout=timeout_sec + 2.0)

            if hasattr(raw, "model_dump"):
                return raw.model_dump()
            return raw

        # Path B: Direct REST call to POST https://api.typesafe.ai/v1/systemone
        if not self.jev_api_key:
            raise PermissionError("Missing JEV_API_KEY / TYPESAFE_API_KEY")

        url = f"{self.models_cfg.jev.base_url.rstrip('/')}/v1/systemone"
        request_body = {
            "model": model,
            "state": state_payload,
            "questions": _JEV_QUESTION_SPEC,
        }
        headers = {
            "Authorization": f"Bearer {self.jev_api_key}",
            "Content-Type": "application/json",
        }
        if self._http_client is not None:
            resp = await asyncio.wait_for(
                self._http_client.post(url, json=request_body, headers=headers),
                timeout=timeout_sec + 2.0,
            )
        else:
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                resp = await asyncio.wait_for(
                    client.post(url, json=request_body, headers=headers),
                    timeout=timeout_sec + 2.0,
                )

        if resp.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"Jev HTTP {resp.status_code}: {resp.text[:200]}",
                request=resp.request,
                response=resp,
            )
        return resp.json()

    async def score_message(
        self,
        channel_name: str,
        prior_messages: list[ChatMessage],
        target_message: ChatMessage,
        backend_override: Literal["jev", "openai_fallback"] | None = None,
    ) -> TriageItem:
        """Score a single message with caching and graceful Jev -> OpenAI Luna fallback."""
        active_backend = backend_override or self.backend

        if active_backend == "openai_fallback":
            fb = self._get_fallback()
            item, cost = await fb.score_message(
                channel_name, prior_messages, target_message
            )
            self.total_cost_usd += cost
            self.uncached_cost_usd += max(cost, fb.last_uncached_cost_usd)
            return item

        state_payload = build_jev_state_payload(channel_name, prior_messages, target_message)
        cache_payload = build_jev_cache_key_payload(channel_name, prior_messages, target_message)
        jev_model = self.models_cfg.jev.model
        prompt_hash = compute_prompt_hash({"state": cache_payload, "questions": _JEV_QUESTION_SPEC})
        legacy_hash = compute_prompt_hash({"state": state_payload, "questions": _JEV_QUESTION_SPEC})

        # 1. Check persistent SQLite cache first (content-based key, then legacy msg_id key for backward reuse)
        cached = self.cache.get(
            provider="jev",
            model=jev_model,
            prompt_hash=prompt_hash,
            temperature=0.0,
            seed=self.seed,
        )
        if cached is None:
            cached = self.cache.get(
                provider="jev",
                model=jev_model,
                prompt_hash=legacy_hash,
                temperature=0.0,
                seed=self.seed,
            )
            if cached is not None:
                # Backfill content-based key so future sliding-window turns hit directly
                self.cache.set(
                    provider="jev",
                    model=jev_model,
                    prompt_hash=prompt_hash,
                    temperature=0.0,
                    seed=self.seed,
                    response=cached,
                )
        if cached is not None:
            self.cache_hits += 1
            self.uncached_cost_usd += float(self.models_cfg.jev.cost_per_call_usd)
            return TriageItem.model_validate(cached).model_copy(update={"msg_id": target_message.msg_id})

        # 2. If circuit breaker tripped (e.g., quota exhausted / auth error / missing key), fall back cleanly
        if self._jev_circuit_open_reason is not None:
            return await self._execute_fallback(
                channel_name, prior_messages, target_message, self._jev_circuit_open_reason
            )

        # 3. Try Jev primary model (and pinned fallback if 404/model-not-found)
        last_error: Exception | None = None
        models_to_try = [jev_model]
        if self.models_cfg.jev.pinned_fallback and self.models_cfg.jev.pinned_fallback != jev_model:
            models_to_try.append(self.models_cfg.jev.pinned_fallback)

        for candidate_model in models_to_try:
            try:
                raw = await self._invoke_jev_backend(candidate_model, state_payload)
                item = parse_jev_response_to_triage_item(
                    raw, msg_id=target_message.msg_id, backend="jev"
                )
                self.cache.set(
                    provider="jev",
                    model=jev_model,
                    prompt_hash=prompt_hash,
                    temperature=0.0,
                    seed=self.seed,
                    response=item.model_dump(),
                )
                self.jev_calls += 1
                self._consecutive_jev_failures = 0
                call_cost = float(self.models_cfg.jev.cost_per_call_usd)
                self.total_cost_usd += call_cost
                self.uncached_cost_usd += call_cost
                return item
            except Exception as exc:
                last_error = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                # Only try pinned_fallback on 404 / model-not-found; auth/quota/timeout skip straight to fallback
                if (
                    status in (401, 402, 403, 429)
                    or isinstance(exc, (PermissionError, TimeoutError, asyncio.TimeoutError))
                ):
                    break

        # 4. Graceful fallback to OpenAI Luna (Invariant #5)
        reason = f"{type(last_error).__name__}: {last_error}"
        status = getattr(getattr(last_error, "response", None), "status_code", None)
        self._consecutive_jev_failures += 1
        if (
            status in (401, 402, 403)
            or isinstance(last_error, PermissionError)
            or self._consecutive_jev_failures >= 3
        ):
            self._jev_circuit_open_reason = reason

        if not self.models_cfg.jev.auto_fallback_to_openai:
            raise RuntimeError(f"Jev triage failed and auto_fallback_to_openai=False: {reason}") from last_error

        return await self._execute_fallback(channel_name, prior_messages, target_message, reason)

    async def _execute_fallback(
        self,
        channel_name: str,
        prior_messages: list[ChatMessage],
        target_message: ChatMessage,
        reason: str,
    ) -> TriageItem:
        self.fallback_count += 1
        if len(self.fallback_reasons) < 20:
            self.fallback_reasons.append(reason)
        logger.warning(
            "Jev triage unavailable (%s); falling back to OpenAI Luna (%s) [fallback_count=%d]",
            reason,
            self._luna_model,
            self.fallback_count,
        )
        fb = self._get_fallback()
        item, cost = await fb.score_message(
            channel_name, prior_messages, target_message
        )
        self.total_cost_usd += cost
        self.uncached_cost_usd += max(cost, fb.last_uncached_cost_usd)
        return item

    def get_run_metadata(self) -> dict[str, Any]:
        return {
            "configured_triage_backend": self.backend,
            "jev_calls": self.jev_calls,
            "jev_cache_hits": self.cache_hits,
            "fallback_count": self.fallback_count,
            "jev_circuit_open_reason": self._jev_circuit_open_reason,
            "fallback_reasons_sample": self.fallback_reasons[:5],
            "triage_cost_usd": round(self.total_cost_usd, 6),
            "triage_uncached_cost_usd": round(self.uncached_cost_usd, 6),
        }
