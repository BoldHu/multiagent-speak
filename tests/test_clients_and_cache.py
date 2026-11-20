"""Milestone 1 tests: configs, schemas, secrets, SQLite cache, checkpoints, OpenAI & Jev clients, preflight."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from dial_skill.config import (
    find_repo_root,
    load_dial_skill_config,
    load_experiments_config,
    load_models_config,
)
from dial_skill.llm.jev_client import TriageClient
from dial_skill.llm.openai_client import OpenAIClient
from dial_skill.llm.preflight import load_resolved_models, resolve_model_for_role, run_preflight
from dial_skill.schemas import ACTIVE_SKILLS, ChatMessage, SkillCard
from dial_skill.utils.cache import SQLiteAPICache, compute_prompt_hash
from dial_skill.utils.checkpoint import JsonlCheckpointWriter, load_completed_ids, load_jsonl, save_run_metadata
from dial_skill.utils.secrets import load_jev_api_key, load_openai_api_key, mask_secret


class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        idx = min(self.calls - 1, len(self._responses) - 1)
        content = self._responses[idx]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=200, completion_tokens=50),
        )


class _FakeModelsAPI:
    def __init__(self, model_ids: list[str]) -> None:
        self._model_ids = model_ids

    async def list(self) -> Any:
        return SimpleNamespace(data=[SimpleNamespace(id=m) for m in self._model_ids])


class _FakeAsyncOpenAI:
    def __init__(self, responses: list[str], model_ids: list[str] | None = None) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))
        self.models = _FakeModelsAPI(model_ids or ["gpt-5.6-luna", "gpt-6.1-terra", "gpt-6.1-sol"])


def test_configs_and_seed_skills_validate() -> None:
    models_cfg = load_models_config()
    dial_cfg = load_dial_skill_config()
    exp_cfg = load_experiments_config()

    assert models_cfg.jev.model == "jev-latest"
    assert "gpt-6-luna" in models_cfg.openai.luna_candidates
    assert dial_cfg.gate.lambda_disruption == 1.0
    assert exp_cfg["seed"] == 42

    seed_path = find_repo_root() / "configs" / "seed_skills_v0.json"
    raw = json.loads(seed_path.read_text(encoding="utf-8"))
    assert raw["version"] == 0
    assert set(raw["skills"].keys()) == set(ACTIVE_SKILLS)
    for name, card_dict in raw["skills"].items():
        card = SkillCard.model_validate(card_dict)
        assert card.name == name
        assert len(card.trigger_conditions) <= 5
        assert len(card.veto_conditions) <= 5
        assert len(card.investigation_plan) <= 4


def test_secret_lookup_priority_and_masking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    drive_root = tmp_path / "drive"
    (drive_root / "secrets").mkdir(parents=True)
    ws_root = tmp_path / "ws"
    ws_root.mkdir()

    # Local fallback (tier 4)
    (ws_root / "OPENAI_KEY.txt").write_text("OPENAI_API_KEY=sk-local-12345678\n", encoding="utf-8")
    (ws_root / "JEV_API_KEY.txt").write_text("jev-local-12345678\n", encoding="utf-8")
    assert load_openai_api_key(drive_root=drive_root, workspace_root=ws_root) == "sk-local-12345678"
    assert load_jev_api_key(drive_root=drive_root, workspace_root=ws_root) == "jev-local-12345678"

    # Drive secrets override local (tier 2 > tier 4)
    (drive_root / "secrets" / "OPENAI_KEY.txt").write_text("sk-drive-abcdef99\n", encoding="utf-8")
    assert load_openai_api_key(drive_root=drive_root, workspace_root=ws_root) == "sk-drive-abcdef99"

    # Env overrides Drive (tier 1 > tier 2)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-override99")
    assert load_openai_api_key(drive_root=drive_root, workspace_root=ws_root) == "sk-env-override99"
    assert mask_secret("sk-env-override99") == "sk-e...de99"
    assert mask_secret(None) == "<missing>"


def test_sqlite_cache_five_tuple_key_and_persistence(tmp_path: Path) -> None:
    db_path = tmp_path / "api_cache.sqlite"
    cache = SQLiteAPICache(db_path)
    ph = compute_prompt_hash({"msg": "hello"})

    assert cache.get("openai", "gpt-6-luna", ph, temperature=0.0, seed=42) is None
    cache.set("openai", "gpt-6-luna", ph, {"ok": True}, temperature=0.0, seed=42)

    assert cache.get("openai", "gpt-6-luna", ph, temperature=0.0, seed=42) == {"ok": True}
    # Changing ANY element of the 5-tuple must miss
    assert cache.get("jev", "gpt-6-luna", ph, temperature=0.0, seed=42) is None
    assert cache.get("openai", "gpt-6.1-terra", ph, temperature=0.0, seed=42) is None
    assert cache.get("openai", "gpt-6-luna", ph, temperature=0.2, seed=42) is None
    assert cache.get("openai", "gpt-6-luna", ph, temperature=0.0, seed=99) is None
    cache.close()

    # Reopen and verify persistence
    cache2 = SQLiteAPICache(db_path)
    assert cache2.get("openai", "gpt-6-luna", ph, temperature=0.0, seed=42) == {"ok": True}
    cache2.close()


def test_atomic_jsonl_checkpoint_resume_and_corrupt_tail(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    writer = JsonlCheckpointWriter(path)
    writer.append({"sample_id": "s1", "val": 1})
    writer.append({"sample_id": "s2", "val": 2})

    # Simulate abrupt Colab disconnect leaving a half-written trailing line
    with path.open("a", encoding="utf-8") as f:
        f.write('{"sample_id": "s3", "val": ')

    completed = load_completed_ids(path)
    assert completed == {"s1", "s2"}
    rows = load_jsonl(path)
    assert len(rows) == 2

    meta_path = save_run_metadata(tmp_path / "run_01", {"mode": "pilot"})
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["config"]["mode"] == "pilot"
    assert "timestamp_utc" in meta


def test_openai_client_structured_json_and_cache_reuse(tmp_path: Path) -> None:
    fake_oai = _FakeAsyncOpenAI(['{"advocate_score": 0.82, "note": "ok"}'])
    cache = SQLiteAPICache(tmp_path / "cache.sqlite")
    client = OpenAIClient(api_key="test-key", cache=cache, async_client=fake_oai)

    async def _run() -> tuple[Any, Any]:
        msgs = [{"role": "user", "content": "score this"}]
        r1 = await client.complete_json(model="gpt-6-luna", messages=msgs, temperature=0.0, seed=42)
        r2 = await client.complete_json(model="gpt-6-luna", messages=msgs, temperature=0.0, seed=42)
        return r1, r2

    r1, r2 = asyncio.run(_run())
    assert r1.parsed_json == {"advocate_score": 0.82, "note": "ok"}
    assert r1.from_cache is False
    assert r1.cost_usd > 0.0
    assert r2.from_cache is True
    assert r2.cost_usd == 0.0
    assert fake_oai.chat.completions.calls == 1
    assert client.usage.api_calls == 1
    assert client.usage.cache_hits == 1
    cache.close()


def test_jev_client_success_caching_and_graceful_fallback(tmp_path: Path) -> None:
    msg = ChatMessage(
        msg_id="m_1",
        speaker="alice",
        text="Wait, Stripe shows invoice INV-42 failed—did anyone check why?",
        timestamp="2026-01-01T10:00:00Z",
        channel="#billing",
    )

    # 1. Mocked HTTP Jev success + cache hit
    jev_calls = {"n": 0}

    def _jev_ok_handler(request: httpx.Request) -> httpx.Response:
        jev_calls["n"] += 1
        return httpx.Response(
            200,
            json={
                "answers": {
                    "decision_change": {"probability": 0.88},
                    "stance": {
                        "choice": "need_signal",
                        "probabilities": {
                            "need_signal": 0.85,
                            "defer_or_resolved_signal": 0.05,
                            "scope_risk_signal": 0.05,
                            "background_chatter": 0.05,
                        },
                        "confidence": 0.91,
                    },
                    "skill_hint": {
                        "choice": "diagnose",
                        "probabilities": {"diagnose": 0.7, "check": 0.2, "none": 0.1},
                    },
                    "urgency": {"score": 0.72},
                }
            },
        )

    cache = SQLiteAPICache(tmp_path / "jev_cache.sqlite")
    http_ok = httpx.AsyncClient(transport=httpx.MockTransport(_jev_ok_handler))
    tc_ok = TriageClient(cache=cache, jev_api_key="fake-jev", http_client=http_ok)

    async def _run_ok() -> tuple[Any, Any]:
        t1 = await tc_ok.score_message("#billing", [], msg)
        t2 = await tc_ok.score_message("#billing", [], msg)
        await http_ok.aclose()
        return t1, t2

    t1, t2 = asyncio.run(_run_ok())
    assert t1.backend == "jev"
    assert t1.stance == "need_signal"
    assert t1.skill_hint == "diagnose"
    assert abs(t1.p_decision_change - 0.88) < 1e-6
    assert t2 == t1
    assert jev_calls["n"] == 1
    assert tc_ok.cache_hits == 1

    # 2. Mocked Jev HTTP 402 quota exhaustion -> graceful OpenAI Luna fallback (Invariant #5)
    fallback_json = json.dumps(
        {
            "p_decision_change": 0.81,
            "stance": "need_signal",
            "stance_probs": {
                "need_signal": 0.80,
                "defer_or_resolved_signal": 0.05,
                "scope_risk_signal": 0.05,
                "background_chatter": 0.10,
            },
            "skill_hint": "diagnose",
            "skill_probs": {
                "find": 0.05,
                "explain": 0.0,
                "analyze": 0.05,
                "check": 0.15,
                "diagnose": 0.70,
                "decide": 0.0,
                "coordinate": 0.0,
                "guard": 0.0,
                "none": 0.05,
            },
            "urgency_score": 0.65,
        }
    )
    fake_oai = _FakeAsyncOpenAI([fallback_json])
    oai_client = OpenAIClient(api_key="test-key", cache=cache, async_client=fake_oai)
    http_402 = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(402, text="quota exceeded"))
    )
    tc_fb = TriageClient(
        cache=SQLiteAPICache(tmp_path / "jev_fb_cache.sqlite"),
        openai_client=oai_client,
        luna_model="gpt-6-luna",
        jev_api_key="fake-jev",
        http_client=http_402,
    )

    async def _run_fb() -> Any:
        item = await tc_fb.score_message("#billing", [], msg)
        await http_402.aclose()
        return item

    fb_item = asyncio.run(_run_fb())
    assert fb_item.backend == "openai_fallback"
    assert fb_item.skill_hint == "diagnose"
    assert fb_item.stance == "need_signal"
    assert tc_fb.fallback_count == 1
    meta = tc_fb.get_run_metadata()
    assert meta["fallback_count"] == 1
    assert meta["jev_circuit_open_reason"] is not None

    # 3. Official typesafe_sdk nouls/choices/scores object schema & timeout fallback
    from dial_skill.llm.jev_client import parse_jev_response_to_triage_item

    sdk_obj = SimpleNamespace(
        nouls={"decision_change": SimpleNamespace(noul=0.91)},
        choices={
            "stance": SimpleNamespace(
                choice="defer_or_resolved_signal",
                probabilities={"need_signal": 0.05, "defer_or_resolved_signal": 0.90, "scope_risk_signal": 0.02, "background_chatter": 0.03},
                confidence=0.94,
            ),
            "skill_hint": SimpleNamespace(
                choice="check",
                probabilities={"check": 0.85, "none": 0.15},
                confidence=0.88,
            ),
        },
        scores={"urgency": SimpleNamespace(score=2.25, legend=["none", "low_async", "medium_thread", "high_immediate"])},
    )
    parsed_sdk = parse_jev_response_to_triage_item(sdk_obj, msg_id="m_sdk", backend="jev")
    assert abs(parsed_sdk.p_decision_change - 0.91) < 1e-6
    assert parsed_sdk.stance == "defer_or_resolved_signal"
    assert parsed_sdk.skill_hint == "check"
    assert abs(parsed_sdk.urgency_score - 0.75) < 1e-6  # 2.25 / 3.0 = 0.75
    assert abs(parsed_sdk.confidence - 0.94) < 1e-6

    # Hanging SDK call MUST time out via asyncio.wait_for and fall back gracefully to OpenAI Luna
    class _HangingSDK:
        async def system_one(self, **kwargs: Any) -> Any:
            await asyncio.sleep(5.0)

    fast_models_cfg = load_models_config()
    tc_timeout = TriageClient(
        models_cfg=fast_models_cfg,
        cache=SQLiteAPICache(tmp_path / "jev_to_cache.sqlite"),
        openai_client=oai_client,
        luna_model="gpt-6-luna",
        jev_api_key="fake-jev",
        sdk_client=_HangingSDK(),
    )
    tc_timeout.models_cfg.jev.timeout_sec = -1.9  # effective wait_for timeout = 0.1s
    to_item = asyncio.run(tc_timeout.score_message("#billing", [], msg))
    assert to_item.backend == "openai_fallback"
    assert tc_timeout.fallback_count == 1
    cache.close()


def test_preflight_resolves_candidates_and_saves_json(tmp_path: Path) -> None:
    fake_oai = _FakeAsyncOpenAI(
        responses=["OK", "OK"],
        model_ids=["gpt-5.6-luna", "gpt-6.1-terra", "gpt-6-astra"],
    )
    cache = SQLiteAPICache(tmp_path / "pf_cache.sqlite")
    oai = OpenAIClient(api_key="test-key", cache=cache, async_client=fake_oai)

    http_ok = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "answers": {
                        "decision_change": {"probability": 0.9},
                        "stance": {"choice": "need_signal", "probabilities": {"need_signal": 0.9, "background_chatter": 0.1}},
                        "skill_hint": {"choice": "diagnose", "probabilities": {"diagnose": 0.8, "none": 0.2}},
                        "urgency": {"score": 0.7},
                    }
                },
            )
        )
    )
    tc = TriageClient(cache=cache, openai_client=oai, jev_api_key="fake-jev", http_client=http_ok)

    async def _run() -> Any:
        res = await run_preflight(
            drive_root=tmp_path,
            openai_client=oai,
            triage_client=tc,
            probe_chat_completion=True,
        )
        await http_ok.aclose()
        return res

    resolved = asyncio.run(_run())
    assert resolved.luna == "gpt-5.6-luna"
    assert resolved.terra == "gpt-6.1-terra"
    assert resolved.sol == "gpt-6-astra"
    assert resolved.jev_healthy is True
    assert resolved.triage_backend == "jev"

    loaded = load_resolved_models(tmp_path)
    assert loaded is not None
    assert loaded.luna == "gpt-5.6-luna"
    assert resolve_model_for_role("gate", resolved=loaded) == "gpt-5.6-luna"
    assert resolve_model_for_role("executor", resolved=loaded) == "gpt-6.1-terra"
    cache.close()
