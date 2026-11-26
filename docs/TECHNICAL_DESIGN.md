# Technical Design: DIAL-Collab (v3 Main-Track Architecture)
**Closed-Loop Multi-Agent Collaboration for Proactive Turn-Taking ("When to Speak") and Verified Utterance Synthesis ("How to Speak")**

---

## 1. Problem Formulation & Dual Research Objectives

Proactive conversational and workplace AI agents embedded in multi-party discussions (e.g., open-domain group chats or enterprise Slack workspaces connected to issue trackers, billing, analytics, and policy stores) must jointly solve two coupled sub-problems **without task-specific weight fine-tuning**:
1. **Proactive Turn-Taking (`When & Where to Speak`)**: Overcoming LLMs' sycophantic *speak-bias* (over-interruption on active human momentum, resolved threads, or casual banter) without collapsing into over-conservative silence on genuine unresolved needs, while selecting the least-intrusive effective channel (`SILENT` $\prec$ `DM_NUDGE` $\prec$ `THREAD_BRIEF` $\prec$ `CHANNEL_INTERVENE`).
2. **Utterance Quality & Grounded Hygiene (`How & What to Speak`)**: Ensuring every delivered message is concise, socially calibrated, factually verified against authoritative source records as of the conversation timestamp, compliant with least-privilege disclosure policies on mixed-intent requests, and free of multi-message spam.

### Positioning Relative to Prior Literature

| Literature Thread | Representative Works | Limitation Addressed by `DIAL-Collab` |
| :--- | :--- | :--- |
| **Proactive Turn-Taking & Inner Monologue** | *When2Speak* (arXiv:2605.05626), *Inner Thoughts* (CHI 2025), *ProACT* (Yang et al.), *ProAgent* (Zhang et al., AAAI 2024) | Supervised SFT/RL requires 173k labeled turns and yields ~51% MIR on SFT; single-perspective zero-shot gates lack adversarial cross-examination and pre-delivery response auditing. |
| **Multi-Agent Debate & Critique** | *Multiagent Debate* (Du et al., ICML 2024), *ChatEval* (Chan et al., ICLR 2024), *MAD* (Liang et al., EMNLP 2024) | Standard symmetric debate assumes an output *must* be produced; we adapt asymmetric **Advocate $\rightarrow$ Skeptic Cross-Examination** to the asymmetric silence-vs-action decision AND **Executor $\rightarrow$ Skeptic-Auditor** to pre-delivery response verification. |
| **Non-Parametric Agent Memory & Evolution** | *Reflexion* (Shinn et al., NeurIPS 2023), *AWM* (Wang et al., 2024), *ExpeL* (Zhao et al., AAAI 2024), *ACE* (ICLR 2026) | Unbounded append-only reflection grows linearly ($O(T)$) and induces silence/action oscillation; our secondary `DB-CSE` extension bounds skill-card capacity ($K_{\max}=5$ predicates/side) behind counterfactual anchor replay. |

---

## 2. Core Closed-Loop Multi-Agent Architecture (`DIAL-Collab`)

Unlike naive parallel voting (`Advocate || Skeptic`) or single-prompt gating, `DIAL-Collab` establishes **two genuine closed-loop multi-agent interactions** under strict token and latency discipline:

1. **Stage 1 — System-1 Decision-Change Triage & Per-Thread Isolation (`core/triage.py`, `benchmarks/deepnote_official.py`)**:
   - Scores incremental messages with `jev-latest` (or `OpenAILunaTriageFallback`) for decision-change Value of Information (`p_decision_change`), stance (`need_signal`, `defer_or_resolved_signal`, `scope_risk_signal`, `background_chatter`), and skill priors.
   - **Per-Thread Snapshot Decomposition**: On multi-channel enterprise snapshots (`Official-8`, `PageMantle-Hard`), messages are partitioned by `(channel_id, root_ts)` into isolated `ThreadCandidate` slates so visibly settled decoy threads in one channel cannot poison gating on active unresolved threads in another channel.
2. **Stage 1.5 — Bounded Read-Only Silent Recon (`core/executor.py`, `benchmarks/deepnote_official.py`)**:
   - Executes bounded read-only cross-service lookups across company tools *before* Stage-2 gating so downstream agents evaluate verified enterprise records (e.g., discovering a traffic drop is already owned in Linear) rather than chat surface cues alone.
3. **Stage 2 — Closed-Loop Deliberative Gating (`core/gate.py`, `gate_mode="deliberative_collab"`)**:
   - **Step 2a (Advocate Thesis)**: The `Advocate` evaluates the `SharedEvidenceSlate` and proposes a structured intervention thesis (`advocate_score`, `recommended_skill`, `minimum_effective_channel`, `unmitigated_loss`, `cited_evidence_ids`).
   - **Step 2b (Conditional Skeptic Cross-Examination)**: If `advocate_score < deliberative_rebuttal_min_adv` (`0.35`), both roles agree no unmet need exists and Stage 2 exits immediately in **1 LLM call**. Otherwise, the `Skeptic` receives **both** the identical `SharedEvidenceSlate` and the `Advocate`'s exact thesis, performing targeted cross-examination (`skeptic_score`, `matched_veto_conditions`, `max_acceptable_channel`).
   - **Leak-Free Validation Calibration & Platt Scaling (`eval/runners.py`, `eval/metrics.py`)**:
     - Margin thresholds $(\lambda, \tau_{\text{dm}})$ for 2D gates — and scalar thresholds $\tau$ for **all** 1D baselines (`B1`, `ProACT`, `Direct-SC@3`, `Advocate-only`, `Skeptic-only`) — are calibrated **exclusively** on `val_calib` ($\le 200$ validation turns) maximizing Macro-F1.
     - 1D Platt scaling $P(y=1 \mid \Delta_t) = \sigma(a \Delta_t + b)$ (`LogisticRegression(C=50.0)`) is fit strictly on `val_calib` so `ECE` reflects genuine probability calibration on skewed (~10–13%) base rates without test leakage.
4. **Stage 3 — Applicability-Routed Execution + Pre-Delivery Skeptic-Auditor (`Draft -> Audit -> Refine`)**:
   - **When2Speak Mode (`core/executor.py`)**: `SkillExecutor` drafts a natural, non-robotic group-chat contribution (`WHEN2SPEAK_EXECUTION_SYSTEM_PROMPT`); `Pre-Delivery Skeptic-Auditor` audits the draft against conversation history (`APPROVE`, `REVISE` into a crisp $\le 65$-word reply, or `VETO_TO_SILENT` if redundant).
   - **Enterprise Slack Tool Mode (`benchmarks/deepnote_official.py`)**: Candidate Slack writes (`slack_reply_to_thread`, `slack_post_message`, `slack_send_dm`) are intercepted into a **Staging Buffer (`staged_writes`)** and deduplicated/consolidated per destination (`_consolidate_staged_writes`, eliminating multi-message spam). Before committing to Slack, `Pre-Delivery Skeptic-Auditor` verifies factual grounding against the read-tool trace and checks least-privilege mixed-request compliance (`APPROVE`, 1-round `REVISE`, or `VETO_TO_SILENT`).

> [!IMPORTANT]
> **Strict Zero-Contamination Guarantee (`tests/test_deepnote_official.py`)**: All prompts in `templates.py` and `deepnote_official.py` are 100% domain-general (`GENERAL_ENTERPRISE_DISCIPLINE_BLOCK`) and verified by automated unit tests to contain zero task-specific hints from `Official-8`. Furthermore, `OfficialPromptMatchedDirectAgent` (`B0_prompt_matched_direct`) gives a single-agent baseline the **exact same** domain-general instructions and tool budget to isolate the causal contribution of multi-agent collaboration.

---

## 3. Benchmark Hierarchy & Scientific Honesty

1. **`duke-trust-lab/When2Speak` (`When2Speak-Main = 2,000` test turns + `N = 100` stratified gold `SPEAK` turns)**:
   - Evaluates **both** turn-taking timing (Macro-F1, SPEAK F1, FIR, MIR, AUROC, Platt-calibrated ECE, continuous FIR–MIR Pareto frontier) **and** utterance quality (Helpfulness, Conciseness, Appropriateness, Overall 0–10 against human `reference_response`).
2. **Deepnote Native `Official-8` (`deepnote/agent-initiative-benchmark`, `3` seeds $\times$ `8` tasks)**:
   - Authentic 11-service `SlackWorker` runtime with strict zero-condition integer rubric (`evaluate_agent_run.txt`). Evaluated across `gpt-4.1-mini`, `gpt-4.1`, and frontier `gpt-5.6-terra` over multiple seeds (`42, 1042, 2042`).
3. **`PageMantle-Hard` (`pm_hard_eval_32` held-out + `pm_hard_stream_24` stream, 0% record overlap)**:
   - Multi-channel benchmark grounded on real extracted `deepnote/pagemantle-company-dataset` records, incorporating multi-channel decoy threads, cross-service hidden silence traps (resolvable only by querying Linear/Notion/parent threads), and mixed-intent privacy guards.
4. **`PageMantle-Ext` (`pm_synthetic_sanity_8` / `pm_eval_32`) — Demoted to Synthetic Sanity Check**:
   - Retained solely as an offline/synthetic sanity suite and as part of our **Benchmark Difficulty Audit** quantifying why single-thread template benchmarks overestimate proactive agent competence compared to multi-channel cross-service environments.

---

## 4. Statistical & Reporting Standards

- **Exact Model IDs**: All tables and JSON artifacts report exact provider model IDs (`gpt-4.1-mini`, `gpt-4.1`, `gpt-4o`, `gpt-5.6-terra`, `DeepSeek-V4.1-Flash`, `GLM-5.3-flash`, `jev-latest`) rather than internal aliases.
- **Uncached Inference Cost (`uncached_cost_usd`)**: `OpenAIClient` and `TriageClient` compute token-pricing USD cost on **every** call—including SQLite cache hits—so method cost comparisons are 100% invariant to execution order or cache state.
- **Hypothesis Testing & Multiplicity Control (`eval/metrics.py`)**:
  - Primary binary timing significance: **Paired Bootstrap test (`paired_bootstrap_test`, 5,000 resamples)** directly on $\Delta\text{Macro-F1}$ with 95% CIs (supplemented by exact McNemar's test).
  - Rubric & task-score significance: **Exact Sign-Flip Permutation test** ($2^n$ exact enumeration for $n \le 16$; 10,000 Monte Carlo sign flips for $n > 16$) + **Wilcoxon Signed-Rank test** (`paired_nonparametric_test`).
  - Family-wise error control: **Step-down Holm–Bonferroni correction** (`holm_bonferroni_correction`) across all comparisons in each experiment family.
  - **Cross-Family Blinded Multi-Judge**: `gpt-4.1` (OpenAI) + `DeepSeek-V4.1-Flash` (AutoDL) averaged with randomized candidate blinding tags and pass-order reversal.
