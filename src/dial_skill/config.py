"""Typed YAML configuration loaders for DIAL-Skill."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class JevSettings(BaseModel):
    base_url: str = "https://api.typesafe.ai"
    model: str = "jev-latest"
    pinned_fallback: str = "jev-1.14.0"
    timeout_sec: int = 20
    auto_fallback_to_openai: bool = True
    cost_per_call_usd: float = 0.0005


class OpenAISettings(BaseModel):
    luna_candidates: list[str] = Field(
        default_factory=lambda: ["gpt-6-luna", "gpt-5.6-luna", "gpt-4.1-mini"]
    )
    terra_candidates: list[str] = Field(
        default_factory=lambda: ["gpt-6.1-terra", "gpt-5.6-terra", "gpt-4.1"]
    )
    sol_candidates: list[str] = Field(default_factory=lambda: ["gpt-6.1-sol", "gpt-6-astra"])
    max_concurrency: int = 8
    timeout_sec: int = 60
    pricing_per_1m_tokens: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "gpt-6-luna": {"input": 0.25, "output": 1.00},
            "gpt-5.6-luna": {"input": 0.25, "output": 1.00},
            "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
            "gpt-6.1-terra": {"input": 2.00, "output": 8.00},
            "gpt-5.6-terra": {"input": 2.00, "output": 8.00},
            "gpt-4.1": {"input": 2.00, "output": 8.00},
            "gpt-6.1-sol": {"input": 5.00, "output": 20.00},
            "gpt-6-astra": {"input": 5.00, "output": 20.00},
            "default": {"input": 1.00, "output": 4.00},
        }
    )


class RoleModelSettings(BaseModel):
    triage_backend: Literal["jev", "openai_fallback"] = "jev"
    gate_model_tier: Literal["luna", "terra", "sol"] = "luna"
    executor_model_tier: Literal["luna", "terra", "sol"] = "terra"
    critic_model_tier: Literal["luna", "terra", "sol"] = "terra"
    judge_model_tier: Literal["luna", "terra", "sol"] = "terra"


class ModelsConfig(BaseModel):
    jev: JevSettings = Field(default_factory=JevSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    roles: RoleModelSettings = Field(default_factory=RoleModelSettings)


class TriageConfig(BaseModel):
    theta_skip: float = 0.15
    theta_need: float = 0.25
    k_slate: int = 8
    recency_beta: float = 0.85
    active_window_turns: int = 10


class ReconConfig(BaseModel):
    enabled_on_pagemantle: bool = True
    theta_recon: float = 0.45
    max_calls: int = 3
    soft_cap_sec: float = 30.0
    eligible_skills: list[str] = Field(
        default_factory=lambda: ["find", "check", "analyze", "diagnose", "guard"]
    )


class GateConfig(BaseModel):
    lambda_disruption: float = 1.0
    theta_floor: float = 0.30
    tau_dm: float = 0.05
    tau_thread: float = 0.25
    tau_channel: float = 0.50
    tau_disagreement: float = 0.55
    skeptic_channel_cap_threshold: float = 0.60
    skill_prior_override_floor: float = 0.10
    temperature: float = 0.0


class ExecutorConfig(BaseModel):
    hard_time_cap_sec: float = 180.0
    soft_time_cap_sec: float = 90.0
    default_max_tool_calls: int = 7
    tool_output_max_chars: int = 3000
    temperature: float = 0.2


class EvolutionConfig(BaseModel):
    eta_fp: float = 0.04
    eta_fn: float = 0.06
    eta_scope: float = 0.35
    delta_clip: float = 0.35
    gamma_clip: float = 2.0
    tp_decay: float = 0.98
    anchor_buffer_max: int = 8
    escrow_anchors_m: int = 3
    escrow_min_delta_shift: float = 0.10
    max_rules_per_side: int = 5
    max_investigation_steps: int = 4
    enable_escrow_gate: bool = True


class AblationConfig(BaseModel):
    disable_triage: bool = False  # A1
    triage_backend_override: Literal["jev", "openai_fallback"] | None = None  # A2
    asymmetric_evidence: bool = False  # A3
    gate_mode: Literal[
        "dual_margin", "advocate_only", "skeptic_only", "single_combined"
    ] = "dual_margin"  # A4
    binary_only: bool = False  # A5
    disable_disagreement_deesc: bool = False  # A6
    disable_skill_routing: bool = False  # A7


class DialSkillConfig(BaseModel):
    triage: TriageConfig = Field(default_factory=TriageConfig)
    recon: ReconConfig = Field(default_factory=ReconConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    ablation: AblationConfig = Field(default_factory=AblationConfig)


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from start (or this module) to find repository root containing configs/."""
    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent
    for parent in [cur, *cur.parents]:
        if (parent / "configs").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd().resolve()


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = deepcopy(val)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def load_models_config(path: str | Path | None = None) -> ModelsConfig:
    cfg_path = Path(path) if path else find_repo_root() / "configs" / "models.yaml"
    return ModelsConfig.model_validate(load_yaml(cfg_path))


def load_dial_skill_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> DialSkillConfig:
    cfg_path = Path(path) if path else find_repo_root() / "configs" / "dial_skill.yaml"
    data = load_yaml(cfg_path)
    if overrides:
        data = _deep_merge(data, overrides)
    return DialSkillConfig.model_validate(data)


def load_experiments_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else find_repo_root() / "configs" / "experiments.yaml"
    return load_yaml(cfg_path)
