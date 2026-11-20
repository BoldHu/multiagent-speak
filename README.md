# DIAL-Skill (`when2speak`)

**Decision-Informed Adversarial-Margin Gating with Online Dual-Boundary Skill Evolution** for proactive enterprise agents across **When2Speak** (`duke-trust-lab/When2Speak`) and **Deepnote Agent Initiative Benchmark / PageMantle-Ext** (`deepnote/agent-initiative-benchmark` + `deepnote/pagemantle-company-dataset`).

## Architecture Highlights

1. **Stage 1 — System-1 Decision-Change Triage & Symmetric Shared Evidence Slate (SSES)**:
   - TypeSafe Jev (`jev-latest`, with automatic `OpenAILunaTriageFallback`) scores each turn's decision-change VoI, stance, skill hint, and urgency.
   - Balanced retention of both `need_signal` and `defer_or_resolved_signal` / `scope_risk_signal` turns compiles a canonical `SharedEvidenceSlate` verified by SHA-256 `slate_hash`.
2. **Stage 1.5 — Bounded Read-Only Silent Recon (PageMantle Mode)**:
   - Up to 3 read-only enterprise tool lookups inject verified source-of-truth facts into the shared slate before gating.
3. **Stage 2 — Symmetric-Evidence Dialectic Gating & 4-Tier Dynamic De-escalation**:
   - Parallel independent **Advocate** and **Skeptic** evaluators inspect the bit-for-bit identical slate (`advocate.slate_hash == skeptic.slate_hash`).
   - Differential margin $\Delta_t = S_{\text{adv}} - \lambda S_{\text{skep}} - \delta_{k^*}$ and epistemic disagreement $U_t = \min(S_{\text{adv}}, S_{\text{skep}})$ map onto `SILENT` / `DM_NUDGE` / `THREAD_BRIEF` / `CHANNEL_INTERVENE`.
4. **Stage 3 — Applicability-Routed Skill Execution**:
   - 8 Deepnote-derived Dual-Boundary Skill Cards (`find`, `explain`, `analyze`, `check`, `diagnose`, `decide`, `coordinate`, `guard`) with bounded tool budgets and Deepnote 180s model-time tracking.
5. **Stage 4 — Online Dual-Boundary Counterfactual Skill Evolution (`DB-CSE`)**:
   - Zero weight updates. Bounded numeric updates $(\delta_k, \gamma_k)$ + generalized atomic boundary patches verified by a Counterfactual Replay `EscrowGate` (0 anchor regressions) under a strict $O(1)$ prompt cap.

## Quickstart (Local & Google Colab)

```bash
# 1. Install in editable mode
pip install -e ".[dev]"

# 2. Run offline unit tests (zero API cost)
pytest -v

# 3. Validate all Colab notebooks
python -m scripts.validate_notebooks

# 4. Run live API preflight (< $0.01)
python -m dial_skill.llm.preflight

# 5. Run end-to-end Pilot Verification
python -m dial_skill.eval.runners pilot --config configs/experiments.yaml
```

## Secret Safety

API keys (`JEV_API_KEY.txt`, `OPENAI_KEY.txt`) are strictly excluded by `.gitignore`. Lookup order:
1. Environment variables (`TYPESAFE_API_KEY` / `JEV_API_KEY`, `OPENAI_API_KEY`)
2. Google Drive secrets folder (`/content/drive/MyDrive/when2speak_research/secrets/`)
3. Colab `userdata` secrets
4. Local workspace root (`./JEV_API_KEY.txt`, `./OPENAI_KEY.txt`)
