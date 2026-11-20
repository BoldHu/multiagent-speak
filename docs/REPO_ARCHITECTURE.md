# Repository Architecture & Colab Blueprint: `when2speak` (`dial_skill`)

---

## 1. Directory Structure

```text
when2speak/                                    # GitHub Repository Root
├── .gitignore                                 # Blocks *_KEY.txt, .env, data/, runs/, *.sqlite
├── AGENT.md                                   # Master implementer blueprint for Gemini 4 Neon
├── README.md                                  # Quickstart (Local VS Code + GitHub + Colab + Drive)
├── pyproject.toml                             # PEP 621 package metadata & dependencies
│
├── configs/
│   ├── models.yaml                            # Jev + OpenAI Luna/Terra aliases, pricing, fallback order
│   ├── dial_skill.yaml                        # All DIAL-Skill & DB-CSE hyperparameters
│   ├── experiments.yaml                       # Pilot vs Full split sizes, ablations, seeds
│   └── seed_skills_v0.json                    # Initial 8 Deepnote-derived Dual-Boundary Skill Cards
│
├── docs/
│   ├── TECHNICAL_DESIGN.md                    # Full mathematical method, benchmarks, criteria
│   ├── PROMPT_SYSTEM.md                       # Complete Jev / LLM / DB-CSE / Judge prompt library
│   └── REPO_ARCHITECTURE.md                   # This document
│
├── src/dial_skill/
│   ├── __init__.py
│   ├── schemas.py                             # All Pydantic v2 data contracts
│   │
│   ├── utils/
│   │   ├── secrets.py                         # Multi-source loader: env -> Drive -> Colab -> local txt
│   │   ├── cache.py                           # Persistent SQLite API response cache (on Drive)
│   │   ├── checkpoint.py                      # Atomic JSONL writer + idempotent --resume loader
│   │   └── drive.py                           # Colab/local path resolver ($DRIVE_ROOT vs ./runs)
│   │
│   ├── llm/
│   │   ├── preflight.py                       # Probes OpenAI /v1/models & Jev health -> resolved_models.json
│   │   ├── openai_client.py                   # Async OpenAI client (Structured Outputs, retries, USD tracking)
│   │   └── jev_client.py                      # TypeSafe Jev system_one client + auto Luna fallback
│   │
│   ├── prompts/
│   │   └── templates.py                       # Strict prompt & JSON schema builders
│   │
│   ├── core/
│   │   ├── triage.py                          # Stage 1: Jev incremental triage, short-circuit, SSES builder
│   │   ├── gate.py                            # Stage 2: Parallel Advocate/Skeptic, Delta/U_t, 4-tier ladder
│   │   ├── skill_store.py                     # Versioned JSON SkillStore (v0..vN) + O(1) cap validation
│   │   ├── executor.py                        # Stage 1.5 Silent Recon + Stage 3 Budgeted Tool & Channel Exec
│   │   └── pipeline.py                        # Unified DialSkillAgent (When2Speak + PageMantle modes)
│   │
│   ├── evolution/
│   │   ├── feedback.py                        # 5-class typed social-operational feedback extractor
│   │   ├── db_cse.py                          # Attribution Critic + Numeric + Symbolic + EscrowGate + Pruner
│   │   └── baselines.py                       # ReflexionAppendUpdater & AWMWorkflowUpdater
│   │
│   ├── benchmarks/
│   │   ├── when2speak.py                      # HF loader + conversation-level stratified sampler
│   │   ├── pagemantle_env.py                  # 11-service PageMantle disk-backed harness + 180s timer
│   │   └── build_pagemantle_ext.py            # DB-grounded 56-task builder + existence & judge verifier
│   │
│   ├── baselines/
│   │   ├── direct_chat.py                     # B0: Direct Chat baseline
│   │   ├── single_gate.py                     # B1: Single-Model Confidence Gate baseline
│   │   └── proact.py                          # B2: ProACT-Reimpl (breakdown-aware + static skill routing)
│   │
│   └── eval/
│       ├── metrics.py                         # Macro F1, FIR, MIR, AUROC, ECE, McNemar, Bootstrap CIs
│       ├── judge.py                           # Blinded Dual-Judge (Deepnote 4-axis + Dialogue + Kappa)
│       ├── runners.py                         # Typer CLI & Python runners for Pilot/Main/Online/Ablations
│       └── report.py                          # Publication LaTeX table & matplotlib vector figure builders
│
├── notebooks/                                 # All 7 Colab-native notebooks (Drive mount + git pull!)
│   ├── 00_setup_and_preflight.ipynb
│   ├── 01_build_pagemantle_ext.ipynb
│   ├── 02_pilot_verification.ipynb
│   ├── 03_exp_when2speak_main.ipynb
│   ├── 04_exp_pagemantle_and_online_dbcse.ipynb
│   ├── 05_exp_ablations_and_scaling.ipynb
│   └── 06_analysis_and_paper_artifacts.ipynb
│
├── scripts/
│   └── validate_notebooks.py                  # Validates JSON & Python AST of all notebooks
│
└── tests/
    ├── test_clients_and_cache.py
    ├── test_core_pipeline.py
    ├── test_evolution_dbcse.py
    ├── test_benchmarks.py
    └── test_metrics_and_baselines.py
```

### Persistent Google Drive Layout (Never Lost on Colab Disconnect!)

```text
/content/drive/MyDrive/when2speak_research/
├── secrets/
│   ├── JEV_API_KEY.txt                        # Uploaded manually by user (NOT in GitHub)
│   └── OPENAI_KEY.txt                         # Uploaded manually by user (NOT in GitHub)
├── config_cache/
│   └── resolved_models.json                   # Verified Luna / Terra / Jev IDs from Notebook 00
├── datasets/
│   ├── pagemantle/                            # ~10 GB PageMantle snapshot (downloaded ONCE)
│   ├── when2speak/                            # Cached HF When2Speak splits
│   └── pagemantle_ext/
│       ├── pm_official_8.jsonl
│       ├── pm_online_stream_24.jsonl
│       ├── pm_eval_32.jsonl
│       └── verification_audit.json
├── caches/
│   └── api_cache.sqlite                       # Shared persistent response cache across runs
├── runs/
│   ├── pilot_YYYYMMDD/
│   │   ├── run_config.json
│   │   ├── predictions_when2speak.jsonl       # Atomic checkpointed predictions
│   │   ├── predictions_pagemantle.jsonl
│   │   ├── skill_store_v*.json
│   │   └── pilot_scorecard.json               # Automated GO / NO-GO report
│   ├── main_when2speak/
│   ├── main_pagemantle/
│   ├── online_dbcse_waves/
│   ├── ablations_a1_a7/
│   └── scaling_pareto/
└── paper_artifacts/
    ├── tables/*.tex
    ├── figures/*.pdf
    └── human_calibration_30.csv
```

---

## 2. Core Pydantic v2 Data Contracts (`src/dial_skill/schemas.py`)

```python
from __future__ import annotations
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field

class InterventionTier(str, Enum):
    SILENT = "SILENT"                    # Tier 0
    DM_NUDGE = "DM_NUDGE"                # Tier 1
    THREAD_BRIEF = "THREAD_BRIEF"        # Tier 2
    CHANNEL_INTERVENE = "CHANNEL_INTERVENE"  # Tier 3

TIER_TO_INT = {
    InterventionTier.SILENT: 0,
    InterventionTier.DM_NUDGE: 1,
    InterventionTier.THREAD_BRIEF: 2,
    InterventionTier.CHANNEL_INTERVENE: 3,
}
INT_TO_TIER = {v: k for k, v in TIER_TO_INT.items()}

SkillName = Literal[
    "find", "explain", "analyze", "check",
    "diagnose", "decide", "coordinate", "guard", "none"
]
StanceLabel = Literal[
    "need_signal", "defer_or_resolved_signal",
    "scope_risk_signal", "background_chatter"
]
FeedbackType = Literal[
    "TP_VALIDATED", "TN_VALIDATED",
    "FP_INTRUSION", "FP_SCOPE_OVERREACH",
    "FN_OMISSION", "CONTENT_EVIDENCE_GAP"
]

class ChatMessage(BaseModel):
    msg_id: str
    speaker: str
    text: str
    timestamp: str = ""
    channel: str = "#general"
    thread_ts: str | None = None

class TriageItem(BaseModel):
    msg_id: str
    backend: Literal["jev", "openai_fallback"]
    p_decision_change: float = Field(ge=0.0, le=1.0)
    stance: StanceLabel
    stance_probs: dict[StanceLabel, float]
    skill_hint: SkillName
    skill_probs: dict[SkillName, float]
    urgency_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class ReconSnippet(BaseModel):
    recon_id: str
    service: str
    query: str
    summary_fact: str

class ChannelPolicy(BaseModel):
    default_tier: InterventionTier
    max_tier: InterventionTier
    deescalation_rule: str

class SkillCard(BaseModel):
    name: SkillName
    description: str
    trigger_conditions: list[str] = Field(max_length=5)
    veto_conditions: list[str] = Field(max_length=5)
    channel_policy: ChannelPolicy
    investigation_plan: list[str] = Field(max_length=4)
    max_tool_calls: int = 6
    response_contract: str
    margin_offset: float = Field(default=0.0, ge=-0.35, le=0.35)
    deescalation_bias: float = Field(default=0.0, ge=0.0, le=2.0)
    rule_stats: dict[str, dict[str, int]] = Field(default_factory=dict)

class SharedEvidenceSlate(BaseModel):
    conversation_id: str
    channel_name: str
    env_mode: Literal["when2speak", "pagemantle"]
    latest_msg_id: str
    selected_messages: list[ChatMessage]
    triage_annotations: dict[str, TriageItem]
    recon_items: list[ReconSnippet] = Field(default_factory=list)
    skill_priors: dict[SkillName, float]
    candidate_skills: list[SkillCard]
    canonical_text: str
    slate_hash: str   # sha256(canonical_text) — MUST match between Advocate & Skeptic!

class AdvocateOutput(BaseModel):
    slate_hash: str
    advocate_score: float = Field(ge=0.0, le=1.0)
    unmitigated_loss: str
    recommended_skill: SkillName
    minimum_effective_channel: Literal["DM_NUDGE", "THREAD_BRIEF", "CHANNEL_INTERVENE"]
    cited_evidence_ids: list[str]

class SkepticOutput(BaseModel):
    slate_hash: str
    skeptic_score: float = Field(ge=0.0, le=1.0)
    disruption_risk: str
    matched_veto_conditions: list[str]
    max_acceptable_channel: InterventionTier
    cited_evidence_ids: list[str]

class GateDecision(BaseModel):
    short_circuited: bool = False
    slate_hash: str
    selected_skill: SkillName
    advocate_score: float
    skeptic_score: float
    margin_delta: float          # S_adv - lambda * S_skep - delta_k
    disagreement_u: float        # min(S_adv, S_skep)
    raw_tier: InterventionTier
    final_tier: InterventionTier
    binary_speak: bool           # final_tier != SILENT
    deescalation_reasons: list[str] = Field(default_factory=list)
    advocate: AdvocateOutput | None = None
    skeptic: SkepticOutput | None = None

class EpisodeResult(BaseModel):
    sample_id: str
    benchmark: Literal["when2speak", "pagemantle_official", "pagemantle_ext"]
    method: str
    model_id: str
    gate: GateDecision
    response_text: str | None = None
    target_channel: str | None = None
    target_user: str | None = None
    tool_calls_used: int = 0
    model_elapsed_sec: float = 0.0
    timed_out: bool = False
    prompt_tokens_raw: int = 0
    prompt_tokens_slate: int = 0
    total_cost_usd: float = 0.0
    triage_backend_used: str = "jev"
    metadata: dict[str, Any] = Field(default_factory=dict)
```

---

## 3. Configuration YAML Schemas (`configs/`)

### `configs/models.yaml`
```yaml
jev:
  base_url: "https://api.typesafe.ai"
  model: "jev-latest"
  pinned_fallback: "jev-1.14.0"
  timeout_sec: 20
  auto_fallback_to_openai: true

openai:
  luna_candidates: ["gpt-6-luna", "gpt-5.6-luna", "gpt-4.1-mini"]
  terra_candidates: ["gpt-6.1-terra", "gpt-5.6-terra", "gpt-4.1"]
  sol_candidates: ["gpt-6.1-sol", "gpt-6-astra"]
  max_concurrency: 8
  timeout_sec: 60

roles:
  triage_backend: "jev"          # "jev" | "openai_fallback"
  gate_model_tier: "luna"        # Advocate + Skeptic (cheap, fast, parallel)
  executor_model_tier: "terra"   # PageMantle tool-calling & response synthesis (or "luna" for scaling exp)
  critic_model_tier: "terra"     # DB-CSE Attribution & Patch Proposer
  judge_model_tier: "terra"      # Blinded Dual-Judge
```

### `configs/dial_skill.yaml`
```yaml
triage:
  theta_skip: 0.15
  theta_need: 0.25
  k_slate: 8
  recency_beta: 0.85

recon:
  enabled_on_pagemantle: true
  theta_recon: 0.45
  max_calls: 3
  soft_cap_sec: 30.0

gate:
  lambda_disruption: 1.0
  theta_floor: 0.30
  tau_dm: 0.05
  tau_thread: 0.25
  tau_channel: 0.50
  tau_disagreement: 0.55
  temperature: 0.0

executor:
  hard_time_cap_sec: 180.0
  soft_time_cap_sec: 90.0
  default_max_tool_calls: 7
  tool_output_max_chars: 3000
  temperature: 0.2

evolution:
  eta_fp: 0.04
  eta_fn: 0.06
  eta_scope: 0.35
  delta_clip: 0.35
  gamma_clip: 2.0
  tp_decay: 0.98
  anchor_buffer_max: 8
  escrow_anchors_m: 3
  max_rules_per_side: 5
  enable_escrow_gate: true
```

---

## 4. Colab Notebook Suite (`notebooks/00` – `06`)

### Universal Cell 0 & Cell 1 (Required at the Top of EVERY Notebook)

```python
# === CELL 0: Mount Google Drive & Configure Persistent Workspace ===
import os
from pathlib import Path

IN_COLAB = "COLAB_RELEASE_TAG" in os.environ or Path("/content").exists()
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE_ROOT = Path("/content/drive/MyDrive/when2speak_research")
else:
    DRIVE_ROOT = Path("./local_drive_mirror").resolve()

for sub in ["secrets", "config_cache", "datasets", "caches", "runs", "paper_artifacts/tables", "paper_artifacts/figures"]:
    (DRIVE_ROOT / sub).mkdir(parents=True, exist_ok=True)

os.environ["DIAL_SKILL_DRIVE_ROOT"] = str(DRIVE_ROOT)
print("Persistent DRIVE_ROOT:", DRIVE_ROOT)
```

```python
# === CELL 1: Sync GitHub Repo (git clone / git pull) & Hot-Reload ===
REPO_URL = "https://github.com/<YOUR_USERNAME>/when2speak.git"  # Edit once
REPO_DIR = Path("/content/when2speak") if IN_COLAB else Path(".").resolve()

if IN_COLAB:
    if not REPO_DIR.exists():
        !git clone {REPO_URL} {REPO_DIR}
    else:
        !git -C {REPO_DIR} pull --ff-only
    %cd {REPO_DIR}
    !pip install -q -e . --no-deps

%load_ext autoreload
%autoreload 2
```

### Purpose of Each Notebook

| Notebook | Goal | Saves to Google Drive | Supports `MODE` |
| :--- | :--- | :--- | :---: |
| **`00_setup_and_preflight.ipynb`** | Install deps, clone `deepnote/agent-initiative-benchmark`, verify `JEV_API_KEY.txt` + `OPENAI_KEY.txt`, resolve working `Luna`/`Terra`/`Jev` IDs, cache PageMantle (10 GB) & When2Speak on Drive. | `config_cache/resolved_models.json`, `datasets/pagemantle/`, `datasets/when2speak/` | Setup |
| **`01_build_pagemantle_ext.ipynb`** | Mine verified cross-service facts from PageMantle's 11 services, synthesize 56 grounded tasks (40 Action + 16 Silence Traps), run deterministic DB-existence + Terra ambiguity verifier, write leak-free splits. | `datasets/pagemantle_ext/*.jsonl`, `verification_audit.json` | Build |
| **`02_pilot_verification.ipynb`** | Run Preliminary Verification (`When2Speak-Pilot` 400 turns + `pm_official_8` + 4 traps + 1 mini DB-CSE wave). Prints automated **GO / NO-GO Scorecard** & cost breakdown (\$5–\$15). | `runs/pilot_*/predictions_*.jsonl`, `pilot_scorecard.json` | `pilot` |
| **`03_exp_when2speak_main.ipynb`** | Run `When2Speak` calibration (200 val turns) + Main evaluation (`pilot`=400 / `main`=2,000 / `full`=21,687) comparing `Direct Chat`, `Single Gate`, `ProACT`, and `DIAL-Skill` (`Luna` & `Terra`) + Dialogue quality subset. | `runs/main_when2speak/*.jsonl`, `metrics_summary.json` | `pilot` / `main` / `full` |
| **`04_exp_pagemantle_and_online_dbcse.ipynb`** | Run `pm_official_8` + `pm_eval_32` pre-evolution baselines (`Direct Chat`, `Single Gate`, `ProACT-Enterprise`, `DIAL-Skill-v0`), then run 3-wave online stream (`pm_online_stream_24`) comparing `DB-CSE` vs `Reflexion-Append` vs `AWM-Workflow` vs `DB-CSE w/o Escrow`, then evaluate post-evolution generalization on held-out `pm_official_8` + `pm_eval_32`. | `runs/main_pagemantle/`, `runs/online_dbcse_waves/`, `skill_store_v*.json`, `skill_evolution_log.jsonl` | `pilot` / `full` |
| **`05_exp_ablations_and_scaling.ipynb`** | Run all 7 ablations (`A1`–`A7`) + $\lambda$-sweep FIR–MIR Pareto frontier + `Luna`/`Terra`/(optional `Sol`) cost-performance scaling. Reuses cached Stage-1 SQLite calls to save cost! | `runs/ablations_a1_a7/`, `runs/scaling_pareto/` | `pilot` / `full` |
| **`06_analysis_and_paper_artifacts.ipynb`** | Run Blinded Dual-Judge scoring, McNemar & Bootstrap 95% CIs, export 30-sample CSV for quick human $\kappa/\rho$ spot-check, generate all LaTeX tables & vector PDF figures, and extract 3 paper case studies. | `paper_artifacts/tables/*.tex`, `paper_artifacts/figures/*.pdf`, `human_calibration_30.csv` | Analysis |

---

## 5. Key Dependencies (`pyproject.toml`)

```toml
[project]
name = "dial-skill"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "openai>=1.60.0",
  "typesafe-sdk>=0.1.0",
  "pydantic>=2.7.0",
  "pyyaml>=6.0",
  "datasets>=2.20.0",
  "huggingface-hub>=0.24.0",
  "duckdb>=1.1.0",
  "pandas>=2.2.0",
  "numpy>=1.26.0",
  "scipy>=1.13.0",
  "scikit-learn>=1.5.0",
  "matplotlib>=3.9.0",
  "tenacity>=8.3.0",
  "typer>=0.12.0",
  "rich>=13.7.0",
  "tqdm>=4.66.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.2.0", "pytest-asyncio>=0.23.0", "nbformat>=5.10.0", "ruff>=0.5.0"]
```

---

## 6. Daily VS Code + GitHub + Colab Developer Workflow

1. **Edit in VS Code** (`d:\python_project\when2speak\src\dial_skill\...`) and verify offline with `pytest -v`.
2. **Push code**: `git add -A && git commit -m "..." && git push` (keys and data are blocked by `.gitignore`).
3. **Pull in Colab** (or VS Code connected to Colab kernel): re-run **Cell 1** (`!git pull --ff-only`).
4. **Resume experiments**: re-run the experiment cell — `checkpoint.py` automatically skips completed `sample_id`s on Google Drive, and `cache.py` reuses prior Jev/OpenAI responses from `api_cache.sqlite`!
