"""Preflight probe verifying OpenAI models and Jev health -> config_cache/resolved_models.json."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from dial_skill.config import ModelsConfig, load_models_config
from dial_skill.llm.jev_client import TriageClient
from dial_skill.llm.openai_client import OpenAIClient
from dial_skill.schemas import ChatMessage
from dial_skill.utils.cache import SQLiteAPICache
from dial_skill.utils.drive import ensure_drive_layout, get_drive_root
from dial_skill.utils.secrets import describe_secret_sources

logger = logging.getLogger(__name__)


class ResolvedModels(BaseModel):
    timestamp_utc: str
    luna: str
    terra: str
    sol: str | None = None
    jev_model: str
    jev_healthy: bool
    triage_backend: Literal["jev", "openai_fallback"]
    available_openai_models_count: int = 0
    notes: list[str] = []


def _select_first_available(candidates: list[str], available: set[str]) -> str:
    for cand in candidates:
        if cand in available:
            return cand
    return candidates[0]


def resolve_model_for_role(
    role: Literal["gate", "executor", "critic", "judge", "luna", "terra", "sol"],
    models_cfg: ModelsConfig | None = None,
    resolved: ResolvedModels | None = None,
) -> str:
    """Resolve an exact OpenAI model ID for a logical role from config + resolved_models.json."""
    cfg = models_cfg or load_models_config()
    role_to_tier = {
        "gate": cfg.roles.gate_model_tier,
        "executor": cfg.roles.executor_model_tier,
        "critic": cfg.roles.critic_model_tier,
        "judge": cfg.roles.judge_model_tier,
        "luna": "luna",
        "terra": "terra",
        "sol": "sol",
    }
    tier = role_to_tier[role]
    if resolved is not None:
        if tier == "luna":
            return resolved.luna
        if tier == "terra":
            return resolved.terra
        if tier == "sol" and resolved.sol:
            return resolved.sol
    if tier == "luna":
        return cfg.openai.luna_candidates[0]
    if tier == "terra":
        return cfg.openai.terra_candidates[0]
    return cfg.openai.sol_candidates[0]


def load_resolved_models(drive_root: str | Path | None = None) -> ResolvedModels | None:
    """Load saved resolved_models.json from Drive config_cache if it exists."""
    path = get_drive_root(drive_root) / "config_cache" / "resolved_models.json"
    if not path.is_file():
        return None
    return ResolvedModels.model_validate_json(path.read_text(encoding="utf-8"))


async def run_preflight(
    models_cfg: ModelsConfig | None = None,
    drive_root: str | Path | None = None,
    openai_client: OpenAIClient | None = None,
    triage_client: TriageClient | None = None,
    probe_chat_completion: bool = True,
) -> ResolvedModels:
    """Probe OpenAI /v1/models and run a 1-item Jev health check, persisting resolved_models.json."""
    cfg = models_cfg or load_models_config()
    root = ensure_drive_layout(drive_root)
    cache = SQLiteAPICache(root / "caches" / "api_cache.sqlite")
    oai = openai_client or OpenAIClient(models_cfg=cfg, cache=cache)

    notes: list[str] = []
    available_models: set[str] = set()
    try:
        model_ids = await oai.list_models()
        available_models = set(model_ids)
    except Exception as exc:
        notes.append(f"OpenAI /v1/models listing failed ({type(exc).__name__}: {exc}); using configured defaults.")

    luna = _select_first_available(cfg.openai.luna_candidates, available_models)
    terra = _select_first_available(cfg.openai.terra_candidates, available_models)
    sol = (
        _select_first_available(cfg.openai.sol_candidates, available_models)
        if cfg.openai.sol_candidates
        else None
    )

    if probe_chat_completion:
        for tier_name, candidates, current in [
            ("luna", cfg.openai.luna_candidates, luna),
            ("terra", cfg.openai.terra_candidates, terra),
        ]:
            order = [current, *[c for c in candidates if c != current]]
            chosen = current
            for cand in order:
                try:
                    await oai.complete_text(
                        model=cand,
                        messages=[{"role": "user", "content": "Reply with OK."}],
                        temperature=0.0,
                        max_tokens=4,
                        use_cache=False,
                    )
                    chosen = cand
                    break
                except Exception as exc:
                    notes.append(f"Probe failed for {tier_name} candidate '{cand}': {type(exc).__name__}")
            if tier_name == "luna":
                luna = chosen
            else:
                terra = chosen

    # 1-item Jev health check
    tc = triage_client or TriageClient(
        models_cfg=cfg,
        cache=cache,
        openai_client=oai,
        luna_model=luna,
        backend="jev",
    )
    probe_msg = ChatMessage(
        msg_id="preflight_0",
        speaker="alice",
        text="Can someone check why Stripe invoice INV-100 failed?",
        timestamp="2026-01-01T00:00:00Z",
        channel="#ops",
    )
    jev_healthy = False
    try:
        item = await tc.score_message("#ops", [], probe_msg, backend_override="jev")
        jev_healthy = item.backend == "jev"
        if not jev_healthy:
            reason_note = "; ".join(tc.fallback_reasons[:2]) if tc.fallback_reasons else "unknown"
            notes.append(
                f"Jev probe fell back to OpenAI Luna ({reason_note}); setting triage_backend='openai_fallback'."
            )
    except Exception as exc:
        notes.append(f"Jev health check failed: {type(exc).__name__}: {exc}")

    triage_backend: Literal["jev", "openai_fallback"] = (
        "jev" if (jev_healthy and cfg.roles.triage_backend == "jev") else "openai_fallback"
    )
    resolved = ResolvedModels(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        luna=luna,
        terra=terra,
        sol=sol,
        jev_model=cfg.jev.model,
        jev_healthy=jev_healthy,
        triage_backend=triage_backend,
        available_openai_models_count=len(available_models),
        notes=notes,
    )

    out_path = root / "config_cache" / "resolved_models.json"
    out_path.write_text(resolved.model_dump_json(indent=2), encoding="utf-8")
    return resolved


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("Secret lookup status:", json.dumps(describe_secret_sources(), indent=2))
    resolved = asyncio.run(run_preflight())
    print("Resolved models saved:", resolved.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
