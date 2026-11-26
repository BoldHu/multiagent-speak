# DIAL-Collab (`when2speak`)

**Closed-Loop Multi-Agent Collaboration for Proactive Turn-Taking ("When to Speak") and Verified Utterance Synthesis ("How to Speak")** across **When2Speak** (`duke-trust-lab/When2Speak`), **Deepnote Agent Initiative Benchmark (`Official-8`)**, and **`PageMantle-Hard`** (`deepnote/agent-initiative-benchmark` + `deepnote/pagemantle-company-dataset`).

## Architecture Overview

1. **Stage 1 — System-1 Decision-Change Triage & Thread-Isolated Shared Evidence Slate**:
   - TypeSafe Jev (`jev-latest`, with structured `OpenAILunaTriageFallback`) scores each turn's decision-change Value of Information (VoI), stance, skill prior, and urgency.
   - On multi-channel enterprise snapshots (`Official-8`, `PageMantle-Hard`), snapshots are decomposed by `(channel_id, root_ts)` so already-settled decoy threads do not contaminate active threads.
2. **Stage 1.5 — Bounded Read-Only Silent Recon (Enterprise Mode)**:
   - Executes bounded read-only cross-service lookups across company records before gating so downstream agents reason over verified facts rather than chat surface cues alone.
3. **Stage 2 — Closed-Loop Deliberative Gating (`deliberative_collab`)**:
   - **Advocate** proposes a concrete intervention thesis (`advocate_score`, `recommended_skill`, `minimum_effective_channel`, `unmitigated_loss`, `cited_evidence_ids`).
   - On non-trivial cases (`advocate_score >= deliberative_rebuttal_min_adv`), **Skeptic** directly **cross-examines** the Advocate's thesis against the shared evidence slate and candidate Skill `veto_conditions`.
   - Validation-calibrated margin thresholds (`calibrate_when2speak_thresholds` / `calibrate_1d_score_threshold`) and leak-free **Platt scaling** (`calibrate_platt_from_validation`) map scores to calibrated probabilities and intervention tiers (`SILENT`, `DM_NUDGE`, `THREAD_BRIEF`, `CHANNEL_INTERVENE`) with zero test-set leakage.
4. **Stage 3 — Applicability-Routed Execution + Pre-Delivery Skeptic-Auditor (`Draft -> Audit -> Refine`)**:
   - **Executor** synthesizes a response (or stages candidate Slack writes in a staging buffer after tool verification).
   - **Pre-Delivery Skeptic-Auditor** audits the draft against conversation history, retrieved tool observations, conciseness, and least-privilege disclosure policy (`APPROVE`, `REVISE` max 1 round, or `VETO_TO_SILENT`), consolidating duplicate writes to prevent multi-message spam.
5. **Secondary Extension — Bounded Online Skill Adaptation (`DB-CSE`)**:
   - Optional training-free online adaptation updating bounded per-skill offsets and escrow-verified symbolic boundary predicates (`max_rules_per_side = 5`).

## Self-Contained Google Colab Experiment Suite

All notebooks include self-contained environment setup cells (`Google Drive` mount, `git clone/pull`, `pip install`, secret bootstrap, SQLite API cache reuse, and atomic JSONL `--resume`):

- **Primary Publication Notebooks (Clean V3 Suite)**:
  - `notebooks/10_when2speak_collab_and_utterance_quality.ipynb`: Equal validation-calibrated When2Speak timing (`2,000` turns, Platt ECE, `Direct-SC@3`, `ProACT`, `A4` ablations, cross-family backbones, paired bootstrap + Holm-Bonferroni) **and** Utterance Quality (`100` stratified gold `SPEAK` turns graded by cross-family Dual-Judge).
  - `notebooks/11_official_deepnote_8_clean_multiseed.ipynb`: Zero-contamination programmatic check + 5-way controlled evaluation (`Direct`, `Prompt-Matched Direct`, `Flat v1`, `w/o Stage-3 Audit`, `DIAL-Collab Full`) × `3` seeds × `gpt-4.1-mini` / `gpt-4.1` / `gpt-5.6-terra` on Deepnote `Official-8`.
  - `notebooks/12_pagemantle_hard_v3_ablations_and_dbcse.ipynb`: 32 held-out multi-channel `PageMantle-Hard` tasks across 3 backbones + `Prompt-Matched Direct` + `Stage-3 Audit` ablation + secondary `DB-CSE` vs `Reflexion` held-out comparison.
- **Setup, Diagnostics & Legacy Audit Notebooks**: `00`–`09` (note: `PageMantle-Ext` in `01`/`04` serves strictly as a DB-grounded synthetic sanity check; `Official-8` and `PageMantle-Hard` are the primary enterprise benchmarks).

## Local Verification (Zero API Cost)

```bash
pip install -e ".[dev]"
pytest -q
python -m scripts.validate_notebooks
```

## Secret Safety

API keys (`JEV_API_KEY.txt`, `OPENAI_KEY.txt`, `AUTODL_KEY.txt`) and `local_drive_mirror/` are excluded by `.gitignore` and never printed in logs or notebooks.
