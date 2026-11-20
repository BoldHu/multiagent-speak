# AGENT.md — Engineering & Implementation Blueprint for DIAL-Skill

> **Target Implementer**: Gemini 4 Neon  
> **Repository**: `when2speak` (Package: `dial_skill`)  
> **Research Goal**: Implement, verify, and run publishable experiments for **DIAL-Skill** (*Decision-Informed Adversarial-Margin Gating with Online Dual-Boundary Skill Evolution*) across **When2Speak** (`duke-trust-lab/When2Speak`) and **Deepnote Agent Initiative Benchmark / PageMantle-Ext** (`deepnote/agent-initiative-benchmark` + `deepnote/pagemantle-company-dataset`), running natively on **Google Colab + Google Drive + GitHub + VS Code**.

---

## 1. Authoritative Companion Documents (Read First)

Before writing any code, read these three design specs in full — they contain every formula, schema, prompt, baseline, and file contract:

1. `docs/TECHNICAL_DESIGN.md` — Full algorithmic formulation, innovation matrix vs. ProACT/When2Speak/Reflexion/AWM, `PageMantle-Ext` 64-task construction spec, experiment matrix, Pilot Go/No-Go criteria, Paper-Ready thresholds, hyperparameters, and compute/cost budget.
2. `docs/PROMPT_SYSTEM.md` — Exact Jev `system_one` schemas, OpenAI Luna fallback schemas, Advocate/Skeptic prompts, 8 Deepnote-aligned initial Skill Cards (JSON), channel-shaped execution prompts, DB-CSE online evolution prompts, baseline prompts, and blinded dual-judge evaluation prompts.
3. `docs/REPO_ARCHITECTURE.md` — Exact directory tree, Pydantic data models, module interfaces, YAML config specs, and cell-by-cell design of Notebooks `00` through `06`.

---

## 2. Non-Negotiable Architectural Invariants

Violating any of these invariants invalidates the core scientific claims of the paper. Enforce them with unit tests and runtime assertions:

1. **Symmetric Shared Evidence Slate (SSES) Invariant**:
   - Stage 1 (Jev Triage or OpenAI Luna Fallback) produces a single canonical `SharedEvidenceSlate` object with a deterministic `slate_hash = sha256(canonical_json)`.
   - Stage 2 **Advocate** and **Skeptic** MUST receive the **exact same** `SharedEvidenceSlate` (`advocate_input.slate_hash == skeptic_input.slate_hash`), executed independently in parallel (neither sees the other's response).
   - Exception ONLY when explicitly running Ablation `A3_asymmetric_evidence` (`cfg.ablation.asymmetric_evidence = True`).
2. **Balanced Evidence Retention in Stage 1**:
   - Jev Triage MUST retain high-`p_decision_change` messages of **both** `need_signal` AND `defer_or_resolved_signal` / `scope_risk_signal` stances (plus the latest 2 turns) so Skeptic is never blinded to resolutions or deferrals.
3. **Zero Weight Updates**:
   - No PyTorch fine-tuning, LoRA, or RL weight updates anywhere in `DIAL-Skill`. Online learning (`DB-CSE`) ONLY mutates the structured JSON `SkillStore` (`margin_offset`, `deescalation_bias`, `trigger_conditions`, `veto_conditions`, `investigation_plan`, `channel_policy`).
4. **Counterfactual Replay Escrow & O(1) Prompt Cap**:
   - Online symbolic patches MUST pass the `EscrowGate` (fixes failed episode AND causes `0` regressions on sampled anchor episodes) before promotion from `CANDIDATE` to `ACTIVE`.
   - Every Skill Card enforces hard caps: `len(trigger_conditions) <= 5`, `len(veto_conditions) <= 5`, `len(investigation_plan) <= 4` via consolidation/pruning.
5. **Graceful Jev $\rightarrow$ OpenAI Luna Fallback**:
   - `TriageClient` tries TypeSafe Jev (`typesafe-sdk`, `POST https://api.typesafe.ai/v1/systemone`, `jev-latest`) first when `triage_backend == "jev"`.
   - On quota exhaustion (`402`/`429`), auth error, missing key, or repeated timeout, it logs a structured warning, flips a session fallback counter (recorded in run metadata!), and seamlessly invokes `OpenAILunaTriageFallback` returning the **exact same `TriageItem` schema**.
6. **Secret Safety & Zero Data Loss on Colab**:
   - NEVER commit `JEV_API_KEY.txt`, `OPENAI_KEY.txt`, or any `.env` file. Key loading lookup order:
     1. Environment variables (`TYPESAFE_API_KEY` / `JEV_API_KEY`, `OPENAI_API_KEY`)
     2. Google Drive secrets folder (`/content/drive/MyDrive/when2speak_research/secrets/`)
     3. Colab `userdata` secrets (`google.colab.userdata`)
     4. Local workspace root (`./JEV_API_KEY.txt`, `./OPENAI_KEY.txt`)
   - Every evaluation runner MUST write results atomically (`append + flush + os.fsync`) to `predictions.jsonl` on Google Drive and support idempotent `--resume` skipping already-completed `sample_id`s.
   - All Jev and OpenAI calls MUST pass through a persistent SQLite response cache (`api_cache.sqlite` on Drive) keyed by `(provider, model, prompt_hash, temperature, seed)` so re-runs and ablations sharing Stage 1 never pay twice.
7. **Deepnote 180-Second Budget Enforcement**:
   - On PageMantle tasks, track `model_elapsed_sec` (excluding tool execution time, matching Deepnote's official spec) with a hard ceiling of `180.0s`, plus per-skill `max_tool_calls` (default `6–8`) and soft cap `90.0s` to avoid timeouts.
8. **Zero Data Leakage Across Splits**:
   - `pm_official_8` (the 8 official Deepnote public tasks) and `pm_eval_32` MUST NEVER enter the online evolution stream `pm_online_stream_24`.
   - When2Speak threshold calibration (`lambda`, `tau_1`) MUST ONLY touch the `validation` split (`<= 200` turns), never `test`.

---

## 3. Model Configuration & Preflight Policy

- Read exact model IDs from `configs/models.yaml` (never hardcode model strings in Python logic).
- Default cheap-first OpenAI tiers:
  - `luna`: `gpt-6-luna` (auto-fallback probe order: `["gpt-6-luna", "gpt-5.6-luna", "gpt-4.1-mini"]`)
  - `terra`: `gpt-6.1-terra` (auto-fallback probe order: `["gpt-6.1-terra", "gpt-5.6-terra", "gpt-4.1"]`)
  - `jev`: `jev-latest` (pinned fallback: `jev-1.14.0`, automatic LLM fallback: `luna`)
- Implement `src/dial_skill/llm/preflight.py` to query `GET /v1/models` on OpenAI and execute a 1-item health check on Jev at startup (Notebook `00`), resolving and saving `resolved_models.json` to Google Drive so all downstream notebooks use verified working model IDs.

---

## 4. Phased Implementation Plan (Execute In Order)

Do NOT attempt to write everything in one giant opaque commit. Implement and verify sequentially across 7 milestones, keeping each milestone cleanly tested:

### Milestone 1: Project Scaffolding, Configs, Secrets & Cached LLM/Jev Clients
- Create `pyproject.toml`, `configs/models.yaml`, `configs/dial_skill.yaml`, `configs/experiments.yaml`.
- Implement `src/dial_skill/schemas.py` (all Pydantic v2 data structures defined in `docs/REPO_ARCHITECTURE.md`).
- Implement `src/dial_skill/utils/secrets.py`, `src/dial_skill/utils/cache.py` (SQLite persistent cache), `src/dial_skill/utils/checkpoint.py` (atomic JSONL resume writer + Drive path resolver).
- Implement `src/dial_skill/llm/openai_client.py` (async `openai` client with structured outputs `response_format`, retry/backoff, token & USD cost tracker, cache integration).
- Implement `src/dial_skill/llm/jev_client.py` (`typesafe-sdk` wrapper for `system_one` + automatic fallback to `OpenAILunaTriageFallback`).
- Implement `src/dial_skill/llm/preflight.py`.
- **Gate**: `pytest tests/test_clients_and_cache.py` passes (using mocked HTTP/SDK responses, zero live API cost).

### Milestone 2: Core DIAL-Skill Engine (Stage 1 Triage, Stage 2 Dialectic Gate, Stage 3 Skill Router & Executor)
- Implement `src/dial_skill/prompts/templates.py` loading exact prompts from `docs/PROMPT_SYSTEM.md` (and `configs/seed_skills_v0.json`).
- Implement `src/dial_skill/core/triage.py`: incremental message triage with cache, fast-silence short-circuit ($\theta_{\text{skip}}$), balanced `SharedEvidenceSlate` builder with SHA-256 `slate_hash`.
- Implement `src/dial_skill/core/gate.py`: parallel async Advocate + Skeptic evaluators, `slate_hash` assertion, differential margin $\Delta_t = S_{\text{adv}} - \lambda S_{\text{skep}} - \delta_{k^*}$, epistemic disagreement $U_t = \min(S_{\text{adv}}, S_{\text{skep}})$, 4-tier dynamic de-escalation (`SILENT` / `DM_NUDGE` / `THREAD_BRIEF` / `CHANNEL_INTERVENE`), and ablation hooks (`A1`–`A7`).
- Implement `src/dial_skill/core/skill_store.py`: versioned JSON store for the 8 Deepnote-derived Skill Cards (`find`, `explain`, `analyze`, `check`, `diagnose`, `decide`, `coordinate`, `guard`), atomic save/load, serialization for prompt injection.
- Implement `src/dial_skill/core/executor.py`: Stage-1.5 bounded Silent Recon (max 3 read-only tool calls) + Stage-3 budget-aware tool investigation + channel-shaped response generation (`DM_NUDGE`, `THREAD_BRIEF`, `CHANNEL_INTERVENE`).
- Implement `src/dial_skill/core/pipeline.py`: unified `DialSkillAgent` orchestrating Stages 1–3 for both When2Speak (text-only) and PageMantle (tool-grounded).
- **Gate**: `pytest tests/test_core_pipeline.py` passes (verifies `slate_hash` equality, margin & $U_t$ de-escalation math, fast short-circuit, binary projection, and all 7 ablation switches).

### Milestone 3: Online Non-Parametric Skill Evolution (`DB-CSE`)
- Implement `src/dial_skill/evolution/feedback.py`: typed feedback extractor (`TP_VALIDATED`, `FP_INTRUSION`, `FP_SCOPE_OVERREACH`, `FN_OMISSION`, `CONTENT_EVIDENCE_GAP`) from both PageMantle rubric scores and When2Speak validation labels.
- Implement `src/dial_skill/evolution/db_cse.py`:
  - Counterfactual Attribution Critic
  - Bounded numeric updates ($\delta_k \in [-0.35, 0.35]$, $\gamma_k \in [0.0, 2.0]$)
  - Generalized symbolic patch generator (rejects entity-specific overfitting via regex/guardrail check)
  - `EscrowGate`: counterfactual replay on failed episode + $M=3$ anchor episodes (requires fix + 0 regressions)
  - Hard $O(1)$ cap consolidation/pruning + immutable `skill_evolution_log.jsonl` audit trail.
- Implement online baseline comparators in `src/dial_skill/evolution/baselines.py`: `ReflexionAppendUpdater` and `AWMWorkflowUpdater`.
- **Gate**: `pytest tests/test_evolution_dbcse.py` passes (verifies zero regressions via EscrowGate, bounded numeric clips, $O(1)$ cap enforcement, and serialization rollback).

### Milestone 4: Benchmark Loaders, PageMantle 11-Service Harness & `PageMantle-Ext` Builder
- Implement `src/dial_skill/benchmarks/when2speak.py`: Hugging Face loader (`duke-trust-lab/When2Speak`), deterministic conversation-level stratified sampler for `pilot` (400 turns), `val_calib` (200 turns), `main` (2,000 turns), and `full` (all test turns).
- Implement `src/dial_skill/benchmarks/pagemantle_env.py`: wrapper around `deepnote/agent-initiative-benchmark` and `deepnote/pagemantle-company-dataset` exposing all **11 simulated enterprise services** (Slack, Notion, Linear, HubSpot, Stripe, PostHog, Google Analytics, Google Search Console, Google Calendar, BigQuery, Snowflake) via disk-backed DuckDB/SQLite/JSON connectors (safe for 12 GB RAM Colab!) with strict tool-call counter and 180s timer.
  - **Important**: Inspect the actual cloned `deepnote/agent-initiative-benchmark` repo structure at runtime (Notebook `00` clones it into `external/agent-initiative-benchmark` or installs it in editable mode) and wrap its native tool registry directly wherever possible rather than re-inventing its tool schemas!
- Implement `src/dial_skill/benchmarks/build_pagemantle_ext.py`: deterministic, DB-grounded synthesizer producing `pm_online_stream_24` (18 Action + 6 Silence Traps across 3 waves) and `pm_eval_32` (22 Action + 10 Silence Traps) from real PageMantle records, plus deterministic entity-existence & cross-model ambiguity verifier.
- **Gate**: `pytest tests/test_benchmarks.py` passes on mocked fixtures; CLI dry-run builds & verifies sample tasks cleanly.

### Milestone 5: Baselines & Evaluation Suite
- Implement `src/dial_skill/baselines/`:
  - `direct_chat.py` (`B0: Direct Chat`)
  - `single_gate.py` (`B1: Single-Model Confidence Gate`)
  - `proact.py` (`B2: ProACT-Reimpl` — breakdown-aware single-perspective gating + static skill routing, faithful to Yang et al.)
- Implement `src/dial_skill/eval/metrics.py`:
  - When2Speak: Macro F1, SPEAK F1/P/R, FIR, MIR, AUROC, AUPRC, ECE, McNemar's test, bootstrap 95% CIs.
  - PageMantle: 4-dimension rubric aggregation, Silence Accuracy, Action Recall, Channel Match / De-escalation Utility, Timeout Rate, Cost/Task, Token Reduction %.
- Implement `src/dial_skill/eval/judge.py`: Blinded Dual-Judge evaluator (anonymized outputs, deterministic temperature 0.0, dual rubric passes, Cohen's $\kappa$ / Spearman $\rho$ calculator for human spot-check export CSV).
- Implement `src/dial_skill/eval/runners.py`: unified CLI & Python entrypoints for Pilot, Main, Online Stream, Ablation Matrix, and Scaling/Pareto sweeps with full resume support.
- **Gate**: `pytest tests/test_metrics_and_baselines.py` passes.

### Milestone 6: Colab Notebooks (`00`–`06`) & Plotting / LaTeX Table Generators
- Implement `src/dial_skill/eval/report.py`: generates publication-ready LaTeX tables (`tab_when2speak.tex`, `tab_pagemantle.tex`, `tab_online_evolution.tex`, `tab_ablations.tex`) and matplotlib vector figures (`fig_fir_mir_pareto.pdf`, `fig_online_learning_curve.pdf`, `fig_cost_vs_score.pdf`, `fig_ablation_delta.pdf`, `fig_prompt_length_o1.pdf`).
- Generate all 7 clean, documented Jupyter notebooks in `notebooks/` (`00` through `06`) — EVERY notebook starts with Google Drive mount + `git pull` + editable install + `autoreload` + key bootstrap, and saves every artifact to `$DRIVE_ROOT/`.
- **Gate**: Execute `pytest` (all unit tests green) and `nbformat` validation on all 7 notebooks (valid JSON, zero syntax errors in code cells).

---

## 5. Pragmatic Engineering Guidelines (No Over-Engineering!)

- **Keep code clean, modular, and readable**: Prefer simple dataclasses/Pydantic models and explicit async functions over heavy abstractions or LangChain/AutoGen bloat. Use official lightweight SDKs (`openai`, `typesafe-sdk`, `datasets`, `duckdb`, `pydantic`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`, `tenacity`, `rich`, `typer`).
- **Avoid over-defensive boilerplate**: Focus on reliable API retry (exponential backoff via `tenacity`), atomic Drive JSONL checkpointing, strict JSON schema validation, and clean unit tests. Do not build unnecessary web dashboards, Docker orchestration, or distributed Ray clusters.
- **Cost consciousness**: Always default scripts and notebooks to `MODE = "pilot"` until the user explicitly flips `MODE = "full"`! Always print estimated API call count and USD cost before launching a full sweep.
- **Reproducibility**: Fix `seed = 42` for dataset sampling, temperature `0.0` for gating/judging (`0.2` max for natural-language response generation), and save a full `run_config.json` + `git rev-parse HEAD` alongside every run directory on Google Drive.

---

## 6. Quick Verification Commands (For Implementer)

```bash
# 1. Install in editable mode (local dev or Colab)
pip install -e ".[dev]"

# 2. Run full offline unit test suite (zero API cost)
pytest -v

# 3. Validate all Colab notebooks parse cleanly
python -m scripts.validate_notebooks

# 4. Run live API preflight (1 Jev call + 1 Luna call + 1 Terra call, < $0.01)
python -m dial_skill.llm.preflight

# 5. Run end-to-end Pilot Verification (When2Speak 400 turns + PageMantle official 8)
python -m dial_skill.eval.runners pilot --config configs/experiments.yaml
```
