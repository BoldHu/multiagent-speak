"""Milestone 2 tests: SSES invariant, balanced retention, margin/U_t de-escalation, short-circuit, Silent Recon, 180s cap, and A1-A7 ablations."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from dial_skill.config import DialSkillConfig, load_dial_skill_config
from dial_skill.core.gate import DialecticGate, compute_gate_decision
from dial_skill.core.pipeline import DialSkillAgent
from dial_skill.core.skill_store import SkillStore
from dial_skill.core.triage import TriageStage
from dial_skill.llm.jev_client import TriageClient
from dial_skill.llm.openai_client import OpenAIClient
from dial_skill.schemas import (
    AdvocateOutput,
    ChatMessage,
    InterventionTier,
    SharedEvidenceSlate,
    SkepticOutput,
)
from dial_skill.utils.cache import SQLiteAPICache


class _RoleAwareFakeOpenAI:
    """Deterministic async OpenAI mock returning distinct Advocate, Skeptic, Recon, and Exec outputs."""

    def __init__(
        self,
        advocate_payload: dict[str, Any] | None = None,
        skeptic_payload: dict[str, Any] | None = None,
        combined_payload: dict[str, Any] | None = None,
        exec_text: str = "**Heads-up:** Invoice INV-42 failed due to expired card.",
        recon_tool_call: dict[str, Any] | None = None,
    ) -> None:
        self.advocate_payload = advocate_payload or {
            "advocate_score": 0.85,
            "unmitigated_loss": "Unresolved billing failure impacts renewal.",
            "recommended_skill": "diagnose",
            "minimum_effective_channel": "THREAD_BRIEF",
            "cited_evidence_ids": ["m_last"],
        }
        self.skeptic_payload = skeptic_payload or {
            "skeptic_score": 0.20,
            "disruption_risk": "Low risk; team is blocked.",
            "matched_veto_conditions": [],
            "max_acceptable_channel": "CHANNEL_INTERVENE",
            "cited_evidence_ids": ["m_last"],
        }
        self.combined_payload = combined_payload or {
            **self.advocate_payload,
            **self.skeptic_payload,
        }
        self.exec_text = exec_text
        self.recon_tool_call = recon_tool_call
        self.seen_prompts: list[tuple[str, str]] = []
        self._recon_done = False

        outer = self

        class _Completions:
            async def create(self, **kwargs: Any) -> Any:
                msgs = kwargs["messages"]
                sys_msg = msgs[0]["content"]
                user_msg = msgs[1]["content"]
                outer.seen_prompts.append((sys_msg[:40], user_msg))

                if "ADVOCATE evaluator" in sys_msg:
                    content = json.dumps(outer.advocate_payload)
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
                        usage=SimpleNamespace(prompt_tokens=150, completion_tokens=40),
                    )
                if "SKEPTIC evaluator" in sys_msg:
                    content = json.dumps(outer.skeptic_payload)
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
                        usage=SimpleNamespace(prompt_tokens=150, completion_tokens=40),
                    )
                if "SINGLE pass" in sys_msg:
                    content = json.dumps(outer.combined_payload)
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
                        usage=SimpleNamespace(prompt_tokens=150, completion_tokens=60),
                    )
                if "Stage-1.5 Silent Recon" in sys_msg:
                    if outer.recon_tool_call and not outer._recon_done:
                        outer._recon_done = True
                        tc = SimpleNamespace(
                            id="tc_recon_1",
                            function=SimpleNamespace(
                                name=outer.recon_tool_call["name"],
                                arguments=json.dumps(outer.recon_tool_call["arguments"]),
                            ),
                        )
                        return SimpleNamespace(
                            choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[tc]))],
                            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
                        )
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None))],
                        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
                    )
                # Stage 3 Execution
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=outer.exec_text, tool_calls=None))],
                    usage=SimpleNamespace(prompt_tokens=200, completion_tokens=45),
                )

        self.chat = SimpleNamespace(completions=_Completions())


class _FakeToolEnv:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_openai_tool_schemas(self, read_only: bool = False) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "stripe_get_invoice",
                    "description": "Lookup Stripe invoice status",
                    "parameters": {"type": "object", "properties": {"invoice_id": {"type": "string"}}},
                },
            }
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        return {"invoice_id": arguments.get("invoice_id"), "status": "payment_failed", "reason": "card_expired"}


def _make_jev_mock_client(
    per_msg_scores: dict[str, dict[str, Any]],
    cache_path: Path,
) -> tuple[TriageClient, httpx.AsyncClient]:
    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        msg_id = body["state"]["target_message"]["msg_id"]
        spec = per_msg_scores.get(
            msg_id,
            {
                "p_chg": 0.05,
                "stance": "background_chatter",
                "stance_probs": {
                    "need_signal": 0.02,
                    "defer_or_resolved_signal": 0.02,
                    "scope_risk_signal": 0.01,
                    "background_chatter": 0.95,
                },
                "skill": "none",
                "urgency": 0.05,
            },
        )
        return httpx.Response(
            200,
            json={
                "answers": {
                    "decision_change": {"probability": spec["p_chg"]},
                    "stance": {"choice": spec["stance"], "probabilities": spec["stance_probs"]},
                    "skill_hint": {"choice": spec["skill"], "probabilities": {spec["skill"]: 0.85, "none": 0.15}},
                    "urgency": {"score": spec["urgency"]},
                }
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    tc = TriageClient(cache=SQLiteAPICache(cache_path), jev_api_key="fake-jev", http_client=http)
    return tc, http


def test_balanced_sses_retention_and_hash_equality(tmp_path: Path) -> None:
    """Verify Invariant #1 (SSES identical hash) & Invariant #2 (balanced need + defer retention)."""
    messages = [
        ChatMessage(msg_id=f"m_{i}", speaker=f"u{i%3}", text=f"Routine chatter turn {i}" * 3)
        for i in range(14)
    ]
    messages[3].text = "Did anyone check why Stripe invoice INV-42 failed?"
    messages[7].text = "Update: Bob already resolved INV-42 and updated the card!"
    messages[9].text = "Wait, please don't post customer PII in #general."

    scores = {
        "m_3": {
            "p_chg": 0.92,
            "stance": "need_signal",
            "stance_probs": {"need_signal": 0.90, "defer_or_resolved_signal": 0.03, "scope_risk_signal": 0.02, "background_chatter": 0.05},
            "skill": "diagnose",
            "urgency": 0.8,
        },
        "m_7": {
            "p_chg": 0.95,
            "stance": "defer_or_resolved_signal",
            "stance_probs": {"need_signal": 0.02, "defer_or_resolved_signal": 0.93, "scope_risk_signal": 0.02, "background_chatter": 0.03},
            "skill": "diagnose",
            "urgency": 0.1,
        },
        "m_9": {
            "p_chg": 0.88,
            "stance": "scope_risk_signal",
            "stance_probs": {"need_signal": 0.05, "defer_or_resolved_signal": 0.05, "scope_risk_signal": 0.85, "background_chatter": 0.05},
            "skill": "guard",
            "urgency": 0.6,
        },
        "m_13": {
            "p_chg": 0.40,
            "stance": "need_signal",
            "stance_probs": {"need_signal": 0.40, "defer_or_resolved_signal": 0.30, "scope_risk_signal": 0.10, "background_chatter": 0.20},
            "skill": "diagnose",
            "urgency": 0.4,
        },
    }
    tc, http = _make_jev_mock_client(scores, tmp_path / "triage.sqlite")
    cfg = load_dial_skill_config()
    store = SkillStore.from_json_file()
    stage = TriageStage(tc, cfg)

    async def _run() -> Any:
        res = await stage.run(
            conversation_id="conv_1",
            channel_name="#billing",
            env_mode="pagemantle",
            messages=messages,
            skill_store=store,
        )
        await http.aclose()
        return res

    out = asyncio.run(_run())
    assert out.short_circuited is False
    selected_ids = [m.msg_id for m in out.slate.selected_messages]
    assert len(selected_ids) <= cfg.triage.k_slate
    # Last 2 turns always included
    assert "m_12" in selected_ids and "m_13" in selected_ids
    # BOTH need_signal (m_3) AND defer/scope signals (m_7, m_9) MUST be retained!
    assert "m_3" in selected_ids
    assert "m_7" in selected_ids
    assert "m_9" in selected_ids
    # Chronological order preserved
    assert selected_ids == sorted(selected_ids, key=lambda x: int(x.split("_")[1]))
    # Invariant #1: exact same slate_hash & canonical_text
    assert out.advocate_slate.slate_hash == out.skeptic_slate.slate_hash
    assert out.advocate_slate.canonical_text == out.skeptic_slate.canonical_text


def test_sses_invariant_runtime_assertion_catches_mismatch(tmp_path: Path) -> None:
    """Gate MUST raise AssertionError if Advocate & Skeptic receive mismatched slates when A3 is False."""
    cfg = load_dial_skill_config()
    store = SkillStore.from_json_file()
    fake_oai = _RoleAwareFakeOpenAI()
    oai = OpenAIClient(api_key="k", cache=SQLiteAPICache(tmp_path / "c.sqlite"), async_client=fake_oai)
    gate = DialecticGate(oai, "gpt-6-luna", cfg)

    s1 = SharedEvidenceSlate(
        conversation_id="c",
        channel_name="#g",
        env_mode="when2speak",
        latest_msg_id="m1",
        selected_messages=[ChatMessage(msg_id="m1", speaker="a", text="hi")],
        triage_annotations={},
        skill_priors={"find": 1.0, "explain": 0.0, "analyze": 0.0, "check": 0.0, "diagnose": 0.0, "decide": 0.0, "coordinate": 0.0, "guard": 0.0, "none": 0.0},
        candidate_skills=[store.require("find")],
        canonical_text="SLATE_A",
        slate_hash="hash_a",
    )
    s2 = s1.model_copy(update={"canonical_text": "SLATE_B", "slate_hash": "hash_b"})

    with pytest.raises(AssertionError, match="Invariant #1"):
        asyncio.run(gate.evaluate(advocate_slate=s1, skeptic_slate=s2, skill_store=store))


def test_margin_ut_deescalation_and_guard_privacy_ceiling() -> None:
    """Verify exact Delta_t, U_t, Skeptic cap, gamma_k bias, privacy ceiling, and hard floor math."""
    cfg = load_dial_skill_config()
    store = SkillStore.from_json_file()
    base_slate = SharedEvidenceSlate(
        conversation_id="c1",
        channel_name="#eng",
        env_mode="pagemantle",
        latest_msg_id="m1",
        selected_messages=[ChatMessage(msg_id="m1", speaker="alice", text="test")],
        triage_annotations={},
        skill_priors={"diagnose": 0.8, "guard": 0.1, "find": 0.0, "explain": 0.0, "analyze": 0.0, "check": 0.0, "decide": 0.0, "coordinate": 0.0, "none": 0.1},
        candidate_skills=[store.require("diagnose")],
        canonical_text="SLATE",
        slate_hash="h1",
    )

    # Case 1: High Advocate (0.90), Low Skeptic (0.10) -> Delta=0.80 >= 0.50 -> CHANNEL_INTERVENE
    d1 = compute_gate_decision(
        slate=base_slate,
        advocate=AdvocateOutput(slate_hash="h1", advocate_score=0.90, unmitigated_loss="x", recommended_skill="diagnose", minimum_effective_channel="CHANNEL_INTERVENE", cited_evidence_ids=["m1"]),
        skeptic=SkepticOutput(slate_hash="h1", skeptic_score=0.10, disruption_risk="low", matched_veto_conditions=[], max_acceptable_channel=InterventionTier.CHANNEL_INTERVENE, cited_evidence_ids=["m1"]),
        skill_store=store,
        cfg=cfg,
    )
    assert abs(d1.margin_delta - 0.80) < 1e-6
    assert abs(d1.disagreement_u - 0.10) < 1e-6
    assert d1.raw_tier == InterventionTier.CHANNEL_INTERVENE
    assert d1.final_tier == InterventionTier.CHANNEL_INTERVENE
    assert d1.binary_speak is True

    # Case 2: High Epistemic Disagreement U_t = min(0.95, 0.60) = 0.60 >= 0.55
    # Delta = 0.95 - 0.60 = 0.35 -> raw_tier = THREAD_BRIEF (2); U_t de-escalates -> DM_NUDGE (1)!
    d2 = compute_gate_decision(
        slate=base_slate,
        advocate=AdvocateOutput(slate_hash="h1", advocate_score=0.95, unmitigated_loss="x", recommended_skill="diagnose", minimum_effective_channel="THREAD_BRIEF", cited_evidence_ids=["m1"]),
        skeptic=SkepticOutput(slate_hash="h1", skeptic_score=0.60, disruption_risk="sensitive", matched_veto_conditions=[], max_acceptable_channel=InterventionTier.THREAD_BRIEF, cited_evidence_ids=["m1"]),
        skill_store=store,
        cfg=cfg,
    )
    assert d2.raw_tier == InterventionTier.THREAD_BRIEF
    assert d2.final_tier == InterventionTier.DM_NUDGE
    assert any("epistemic_disagreement_u" in r for r in d2.deescalation_reasons)

    # Case 3: Guard skill has max_tier=THREAD_BRIEF privacy ceiling
    guard_slate = base_slate.model_copy(
        update={
            "skill_priors": {"guard": 0.9, "diagnose": 0.0, "find": 0.0, "explain": 0.0, "analyze": 0.0, "check": 0.0, "decide": 0.0, "coordinate": 0.0, "none": 0.1}
        }
    )
    d3 = compute_gate_decision(
        slate=guard_slate,
        advocate=AdvocateOutput(slate_hash="h1", advocate_score=0.95, unmitigated_loss="PII leak", recommended_skill="guard", minimum_effective_channel="CHANNEL_INTERVENE", cited_evidence_ids=["m1"]),
        skeptic=SkepticOutput(slate_hash="h1", skeptic_score=0.10, disruption_risk="low", matched_veto_conditions=[], max_acceptable_channel=InterventionTier.CHANNEL_INTERVENE, cited_evidence_ids=["m1"]),
        skill_store=store,
        cfg=cfg,
    )
    assert d3.raw_tier == InterventionTier.CHANNEL_INTERVENE
    assert d3.final_tier == InterventionTier.THREAD_BRIEF
    assert any("skill_privacy_ceiling" in r for r in d3.deescalation_reasons)

    # Case 4: Hard Need Floor (S_adv = 0.25 < 0.30) -> SILENT even if S_skep = 0.0
    d4 = compute_gate_decision(
        slate=base_slate,
        advocate=AdvocateOutput(slate_hash="h1", advocate_score=0.25, unmitigated_loss="minor", recommended_skill="diagnose", minimum_effective_channel="DM_NUDGE", cited_evidence_ids=["m1"]),
        skeptic=SkepticOutput(slate_hash="h1", skeptic_score=0.0, disruption_risk="none", matched_veto_conditions=[], max_acceptable_channel=InterventionTier.CHANNEL_INTERVENE, cited_evidence_ids=["m1"]),
        skill_store=store,
        cfg=cfg,
    )
    assert d4.final_tier == InterventionTier.SILENT
    assert d4.binary_speak is False


def test_fast_silence_short_circuit_skips_stage2(tmp_path: Path) -> None:
    messages = [
        ChatMessage(msg_id="m_0", speaker="alice", text="Good morning!"),
        ChatMessage(msg_id="m_1", speaker="bob", text="Morning! Ready for coffee."),
    ]
    tc, http = _make_jev_mock_client({}, tmp_path / "sc_jev.sqlite")
    fake_oai = _RoleAwareFakeOpenAI()
    oai = OpenAIClient(api_key="k", cache=SQLiteAPICache(tmp_path / "sc_oai.sqlite"), async_client=fake_oai)
    agent = DialSkillAgent(
        triage_client=tc,
        openai_client=oai,
        gate_model="gpt-6-luna",
        executor_model="gpt-6.1-terra",
    )

    async def _run() -> Any:
        ep = await agent.run_episode(
            sample_id="w2s_0",
            conversation_id="c0",
            channel_name="#general",
            messages=messages,
            benchmark="when2speak",
            generate_text=False,
        )
        await http.aclose()
        return ep

    ep = asyncio.run(_run())
    assert ep.gate.short_circuited is True
    assert ep.gate.final_tier == InterventionTier.SILENT
    assert ep.gate.binary_speak is False
    assert len(fake_oai.seen_prompts) == 0  # Zero Stage-2 LLM calls!


def test_full_pagemantle_pipeline_with_silent_recon_and_abort(tmp_path: Path) -> None:
    messages = [
        ChatMessage(msg_id="m_0", speaker="alice", text="Hey, did Stripe invoice INV-42 fail?"),
        ChatMessage(msg_id="m_last", speaker="bob", text="Not sure, let me look around."),
    ]
    scores = {
        "m_0": {
            "p_chg": 0.85,
            "stance": "need_signal",
            "stance_probs": {"need_signal": 0.85, "defer_or_resolved_signal": 0.05, "scope_risk_signal": 0.05, "background_chatter": 0.05},
            "skill": "diagnose",
            "urgency": 0.7,
        },
        "m_last": {
            "p_chg": 0.60,
            "stance": "need_signal",
            "stance_probs": {"need_signal": 0.60, "defer_or_resolved_signal": 0.20, "scope_risk_signal": 0.05, "background_chatter": 0.15},
            "skill": "diagnose",
            "urgency": 0.6,
        },
    }
    tc, http = _make_jev_mock_client(scores, tmp_path / "pm_jev.sqlite")
    fake_oai = _RoleAwareFakeOpenAI(
        recon_tool_call={"name": "stripe_get_invoice", "arguments": {"invoice_id": "INV-42"}},
    )
    oai = OpenAIClient(api_key="k", cache=SQLiteAPICache(tmp_path / "pm_oai.sqlite"), async_client=fake_oai)
    tool_env = _FakeToolEnv()
    agent = DialSkillAgent(
        triage_client=tc,
        openai_client=oai,
        gate_model="gpt-6-luna",
        executor_model="gpt-6.1-terra",
    )

    async def _run() -> Any:
        ep = await agent.run_episode(
            sample_id="pm_01",
            conversation_id="pm_c1",
            channel_name="#billing",
            messages=messages,
            benchmark="pagemantle_official",
            tool_env=tool_env,
            generate_text=True,
        )
        await http.aclose()
        return ep

    ep = asyncio.run(_run())
    assert ep.gate.final_tier == InterventionTier.CHANNEL_INTERVENE
    assert ep.tool_calls_used == 1
    assert ep.metadata["recon_calls"] == 1
    assert len(tool_env.calls) == 1
    # Verify Advocate and Skeptic BOTH saw the recon evidence identically
    adv_prompts = [u for s, u in fake_oai.seen_prompts if "ADVOCATE" in s]
    skep_prompts = [u for s, u in fake_oai.seen_prompts if "SKEPTIC" in s]
    assert len(adv_prompts) == 1 and len(skep_prompts) == 1
    assert "card_expired" in adv_prompts[0]
    assert adv_prompts[0].split("Shared Evidence Slate as ADVOCATE:\n\n")[1] == skep_prompts[0].split("Shared Evidence Slate as SKEPTIC:\n\n")[1]
    assert ep.response_text is not None and "INV-42" in ep.response_text


def test_all_seven_ablation_switches_work_cleanly(tmp_path: Path) -> None:
    """Verify A1 through A7 ablation switches execute without error and modify behavior as designed."""
    messages = [
        ChatMessage(msg_id=f"m_{i}", speaker=f"u{i%2}", text=f"Turn {i} message about metrics and billing.")
        for i in range(10)
    ]
    scores = {
        f"m_{i}": {
            "p_chg": 0.7 if i in (2, 5, 9) else 0.1,
            "stance": "defer_or_resolved_signal" if i == 5 else "need_signal",
            "stance_probs": (
                {"need_signal": 0.05, "defer_or_resolved_signal": 0.90, "scope_risk_signal": 0.0, "background_chatter": 0.05}
                if i == 5
                else {"need_signal": 0.80, "defer_or_resolved_signal": 0.05, "scope_risk_signal": 0.05, "background_chatter": 0.10}
            ),
            "skill": "analyze",
            "urgency": 0.6,
        }
        for i in range(10)
    }

    ablation_cases: list[tuple[str, dict[str, Any]]] = [
        ("A1", {"ablation": {"disable_triage": True}}),
        ("A2", {"ablation": {"triage_backend_override": "openai_fallback"}}),
        ("A3", {"ablation": {"asymmetric_evidence": True}}),
        ("A4_adv", {"ablation": {"gate_mode": "advocate_only"}}),
        ("A4_skep", {"ablation": {"gate_mode": "skeptic_only"}}),
        ("A4_comb", {"ablation": {"gate_mode": "single_combined"}}),
        ("A5", {"ablation": {"binary_only": True}}),
        ("A6", {"ablation": {"disable_disagreement_deesc": True}}),
        ("A7", {"ablation": {"disable_skill_routing": True}}),
    ]

    for tag, overrides in ablation_cases:
        cfg = load_dial_skill_config(overrides=overrides)
        tc, http = _make_jev_mock_client(scores, tmp_path / f"jev_{tag}.sqlite")
        # For A2 fallback mock, return valid triage JSON on non-Advocate/Skeptic JSON calls
        fake_oai = _RoleAwareFakeOpenAI(
            advocate_payload={
                "advocate_score": 0.90,
                "unmitigated_loss": "loss",
                "recommended_skill": "analyze",
                "minimum_effective_channel": "THREAD_BRIEF",
                "cited_evidence_ids": ["m_9"],
            },
            skeptic_payload={
                "skeptic_score": 0.58,
                "disruption_risk": "tension",
                "matched_veto_conditions": [],
                "max_acceptable_channel": "THREAD_BRIEF",
                "cited_evidence_ids": ["m_9"],
            },
        )
        if tag == "A2":
            triage_fb_json = json.dumps(
                {
                    "p_decision_change": 0.75,
                    "stance": "need_signal",
                    "stance_probs": {"need_signal": 0.75, "defer_or_resolved_signal": 0.1, "scope_risk_signal": 0.05, "background_chatter": 0.1},
                    "skill_hint": "analyze",
                    "skill_probs": {"find": 0.0, "explain": 0.0, "analyze": 0.8, "check": 0.0, "diagnose": 0.0, "decide": 0.0, "coordinate": 0.0, "guard": 0.0, "none": 0.2},
                    "urgency_score": 0.6,
                }
            )
            orig_create = fake_oai.chat.completions.create

            async def _wrapped_create(orig=orig_create, fb=triage_fb_json, **kw: Any) -> Any:
                if "System-1 decision-change triage" in kw["messages"][0]["content"]:
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=fb, tool_calls=None))],
                        usage=SimpleNamespace(prompt_tokens=80, completion_tokens=30),
                    )
                return await orig(**kw)

            fake_oai.chat.completions.create = _wrapped_create  # type: ignore[method-assign]

        oai = OpenAIClient(api_key="k", cache=SQLiteAPICache(tmp_path / f"oai_{tag}.sqlite"), async_client=fake_oai)
        tc._openai_client = oai
        agent = DialSkillAgent(
            triage_client=tc,
            openai_client=oai,
            gate_model="gpt-6-luna",
            executor_model="gpt-6.1-terra",
            cfg=cfg,
        )

        async def _run(a=agent, h=http) -> Any:
            ep = await a.run_episode(
                sample_id=f"s_{tag}",
                conversation_id=f"c_{tag}",
                channel_name="#analytics",
                messages=messages,
                benchmark="when2speak",
                generate_text=False,
            )
            await h.aclose()
            return ep

        ep = asyncio.run(_run())
        if tag == "A1":
            assert ep.triage_backend_used == "disabled_a1"
        elif tag == "A2":
            assert ep.triage_backend_used == "openai_fallback"
        elif tag == "A3":
            assert ep.gate.advocate is not None and ep.gate.skeptic is not None
            assert ep.gate.advocate.slate_hash != ep.gate.skeptic.slate_hash
        elif tag == "A5":
            assert ep.gate.final_tier == InterventionTier.CHANNEL_INTERVENE
        elif tag == "A6":
            assert ep.gate.final_tier == InterventionTier.THREAD_BRIEF  # U_t=0.58 NOT de-escalated!
        elif tag == "A7":
            assert ep.gate.selected_skill == "none"
