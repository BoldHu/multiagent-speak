# Technical Design: DIAL-Skill
**When, Where, and How to Speak: Decision-Change Triage, Symmetric-Evidence Dialectic Gating, and Online Skill-Boundary Evolution for Proactive Enterprise Agents**

---

## 1. Problem Statement & Core Research Gap

Enterprise AI assistants embedded in group chats (e.g., Slack workspaces connected to CRM, issue trackers, billing, docs, and data warehouses) face a coupled decision problem:
1. **When to intervene (`Timing`)**: Distinguish genuine, unresolved team needs from normal human momentum where chiming in causes disruption.
2. **Where & how strongly to intervene (`Scope / De-escalation`)**: Choose an appropriate social channel (`SILENT` $\prec$ `DM_NUDGE` $\prec$ `THREAD_BRIEF` $\prec$ `CHANNEL_INTERVENE`) rather than binary "shout in `#general` vs. stay completely mute."
3. **How to investigate & respond (`Grounded Skill Execution`)**: Retrieve cross-system enterprise evidence within a strict time budget (180s) and deliver concise, accurate, non-redundant help.
4. **How to adapt online per workspace (`Lifelong Non-Parametric Adaptation`)**: Learn workspace norms, false-interruption boundaries, and investigation blind spots from user/environment feedback **without updating model weights**.

### Why Existing Paradigms Fall Short

| Approach | Core Mechanism | Fatal Failure Mode in Enterprise Group Chat |
| :--- | :--- | :--- |
| **Direct Chat Baseline** | Feeds raw chat history into an LLM with proactive instructions | Severe **over-interruption** ("too chatty") and wall-of-text replies; attention dilution on long channels; timing and answer quality collapse together. |
| **Small Binary Gating Model** | Trains a classifier/small LLM on `SPEAK`/`SILENT` | Decouples *when* from *what skill & what evidence*; brittle out-of-domain; ignores intervention scope (channel vs. thread vs. DM). |
| **When2Speak (SFT / RL)** *(arXiv:2605.05626)* | Weight-level SFT/RL on 173k labeled turns (`<` vs `>`) | Requires massive labeled corpora + GPU training; SFT suffers ~50% Missed Intervention Rate (MIR); purely binary; no external tool grounding; cannot adapt online per team. |
| **ProACT** *(Yang et al., Breakdown-Aware)* | Single-perspective LLM breakdown detector $\rightarrow$ binary speak/silent $\rightarrow$ static collaboration skill | Feeds full raw chat (no System-1 decision-change filtering); single perspective causes confirmation bias; binary speak/silent (no dynamic de-escalation); static text-only skills (no enterprise DB tools, no online skill-boundary evolution). |
| **Reflexion / AWM / SkillWeaver** | Appends free-form verbal lessons or induces tool action macros | Prompt bloat ($O(T)$ growth); **oscillation** (one "don't interrupt" triggers silence collapse); induces *how-to-act* procedures rather than *when-not-to-act* social veto boundaries. |

---

## 2. Core Innovations & Differentiation Matrix (vs. ProACT & Prior Art)

Our framework — **DIAL-Skill** (*Decision-Informed Adversarial-Margin Gating with Online Dual-Boundary Skill Evolution*) — introduces **four orthogonal, ablatable innovations**:

1. **System-1 Decision-Change Triage $\rightarrow$ Symmetric Shared Evidence Slate (SSES)**:
   - Uses **Jev** (TypeSafe AI's calibrated System-1 decision model, with OpenAI Luna structured fallback) to evaluate each turn's **Decision-Change Value of Information (VoI)**: *Could this message alter whether to intervene or which Skill to invoke?*
   - Retains **both** high-VoI positive-need signals and high-VoI resolution/deferral/scope-risk signals, compiling a compact, hash-verified **Symmetric Shared Evidence Slate ($\mathcal{E}_t$)** fed **identically** to downstream opposing roles — eliminating selective-attention confirmation bias while cutting long-chat tokens by 40–70%.
2. **Symmetric-Evidence Adversarial Margin Gating & Epistemic Disagreement De-escalation**:
   - Two independent, parallel roles inspect the **bit-for-bit identical** $\mathcal{E}_t$:
     - **Advocate (热心助人视角)**: Quantifies $S_{\text{adv}} \in [0,1]$ — strength of evidence for an unresolved, substantive team loss if AI stays silent.
     - **Skeptic (厌恶被打断的资深员工视角)**: Quantifies $S_{\text{skep}} \in [0,1]$ — strength of evidence for waiting, avoiding redundancy, preserving flow, or shrinking intervention scope.
   - Computes continuous differential margin $\Delta_t = S_{\text{adv}} - \lambda S_{\text{skep}} - \delta_{k^*}$ **and** epistemic disagreement $U_t = \min(S_{\text{adv}}, S_{\text{skep}})$.
   - Maps $(\Delta_t, U_t)$ onto a **4-tier dynamic intervention ladder** (`SILENT` / `DM_NUDGE` / `THREAD_BRIEF` / `CHANNEL_INTERVENE`). When both Advocate and Skeptic present strong evidence ($U_t \ge \tau_U$), DIAL-Skill **automatically de-escalates** (e.g., public channel $\rightarrow$ quiet thread or targeted DM) instead of forcing a brittle binary choice!
3. **Applicability-First Skill Contracts (Deepnote 8-Skill Seed)**:
   - Adopts Deepnote's 8 enterprise task categories (`find`, `explain`, `analyze`, `check`, `diagnose`, `decide`, `coordinate`, `guard`) as the initial functional partition, but elevates each skill from a flat prompt into a **Dual-Boundary Contract**:
     - Positive `trigger_conditions` (when applicable)
     - Negative `veto_conditions` (when to defer/abstain even if topic matches!)
     - `channel_policy` (privacy/scope defaults & de-escalation rules)
     - `investigation_plan` (minimal cross-service tool checklist & call budget across PageMantle's 11 services, eliminating 180s timeouts)
     - Per-skill calibrated offset $\delta_k$ and de-escalation bias $\gamma_k$.
4. **Online Dual-Boundary Counterfactual Skill Evolution (`DB-CSE`, Zero Weight Updates)**:
   - Replaces flat text reflection with:
     - **5-class typed social-operational feedback** (`TP_VALIDATED`, `FP_INTRUSION`, `FP_SCOPE_OVERREACH`, `FN_OMISSION`, `CONTENT_EVIDENCE_GAP`).
     - **Counterfactual Attribution Critic** separating calibration errors, scope errors, missing veto/trigger predicates, and tool-plan gaps.
     - **Decoupled Numeric + Symbolic Updates**: bounded scalar updates to $(\delta_k, \gamma_k)$ + generalized 1-sentence predicate patches to `trigger_conditions` / `veto_conditions` / `investigation_plan`.
     - **Counterfactual Replay Escrow Gate**: candidate symbolic patches are only promoted to `ACTIVE` if re-scoring fixes the failure **with 0 regressions** on $M=3$ historical anchor episodes, under a strict **$O(1)$ capacity cap** ($K_{\max}=5$ rules per boundary).

### Feature-by-Feature Comparison vs. Prior Work

| Dimension | Direct Chat | When2Speak (SFT/RL) | ProACT | Reflexion / AWM | **DIAL-Skill (Ours)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| System-1 VoI Pre-Filtering (Jev) | No | No | No | No | **Yes (Calibrated VoI + SSES)** |
| Symmetric-Evidence Dual Perspective | No | No | No (Single Breakdown) | No | **Yes (Advocate vs. Skeptic on Identical $\mathcal{E}_t$)** |
| Continuous Margin + Disagreement De-escalation | No | No (Binary) | No (Binary) | No | **Yes (4-Tier: Silent/DM/Thread/Channel)** |
| Enterprise Cross-Service Tool Grounding (11 SaaS) | Naive | No | No | Partial | **Yes (Silent Recon + Budgeted Skill Plans)** |
| Skill Schema Focus | None | None | Static Breakdown Prompt | Action Macros | **Dual-Boundary Applicability Contracts** |
| Online Adaptation Without Weight Updates | No | No (Offline GPU) | No (Static) | Unbounded Text | **Yes (DB-CSE + Escrow Gate + $O(1)$ Cap)** |

---

## 3. Formal Method Specification

### Stage 1: System-1 Decision-Change Triage (Jev) & SSES Construction

Given group chat history $H_{1:t} = \{(id_i, u_i, m_i, \tau_i)\}_{i=1}^t$ at decision turn $t$:

1. **Incremental Turn Scoring**: Each new turn $i$ (paired with its 2 preceding turns $m_{i-2:i-1}$ and channel metadata) is scored **once** and cached via `Jev` (`POST https://api.typesafe.ai/v1/systemone`, `jev-latest`) or `OpenAILunaTriageFallback`:
   - `p_chg(i) = Noul("decision_change")` $\in [0,1]$: probability that message $i$ changes intervention necessity, channel scope, or skill choice.
   - `stance(i) = Choice` $\in \{\texttt{need\_signal}, \texttt{defer\_or\_resolved\_signal}, \texttt{scope\_risk\_signal}, \texttt{background\_chatter}\}$ with full probability distribution $P_{\text{stance}}(i)$ and calibrated confidence.
   - `skill_hint(i) = Choice` $\in \{\texttt{find}, \texttt{explain}, \texttt{analyze}, \texttt{check}, \texttt{diagnose}, \texttt{decide}, \texttt{coordinate}, \texttt{guard}, \texttt{none}\}$ with distribution $P_{\text{skill}}(i)$.
   - `urgency(i) = Score` $\in [0,1]$ over `["none", "low_async", "medium_thread", "high_immediate"]`.

2. **Fast-Silence Short-Circuit (Compute Optimization)**:
   - Let $W_t$ be the recent active window (last 10 turns). If:
     $$p_{\text{chg}}(t) < \theta_{\text{skip}} \quad \land \quad \max_{i \in W_t} P_{\text{stance}}(i)[\texttt{need\_signal}] < \theta_{\text{need}}$$
     (defaults: $\theta_{\text{skip}} = 0.15$, $\theta_{\text{need}} = 0.25$), immediately output `tier = 0 (SILENT)` without calling Stage 2 LLMs.
   - Setting $\theta_{\text{skip}} = 0.0$ disables short-circuiting for pure gating evaluation / ablations.

3. **Balanced Symmetric Shared Evidence Slate ($\mathcal{E}_t$)**:
   - Budget $K_{\text{slate}} = 8$ messages:
     - **Always include** the last 2 turns $\{t-1, t\}$ for immediate conversational continuity.
     - From remaining history $1 \dots t-2$, select top-3 turns ranked by $p_{\text{chg}}(i) \cdot P_{\text{stance}}(i)[\texttt{need\_signal}]$ AND top-3 turns ranked by $p_{\text{chg}}(i) \cdot (P_{\text{stance}}(i)[\texttt{defer\_or\_resolved\_signal}] + P_{\text{stance}}(i)[\texttt{scope\_risk\_signal}])$, deduplicated and sorted chronologically.
   - Compute aggregate Jev skill prior over selected slate turns weighted by recency decay $\beta^{t-i}$ ($\beta = 0.85$):
     $$\bar{P}_{\text{skill}}(k) = \frac{\sum_{i \in \mathcal{E}_t} \beta^{t-i} p_{\text{chg}}(i) P_{\text{skill}}(i)[k]}{\sum_{i \in \mathcal{E}_t} \beta^{t-i} p_{\text{chg}}(i) + \epsilon}$$
   - Attach the top-2 candidate Skill Cards $\{C_{k_1}, C_{k_2}\}$ (including their active `trigger_conditions`, `veto_conditions`, `channel_policy`).
   - Compute `slate_hash = sha256(canonical_json(E_t))`.

### Stage 1.5: Bounded Read-Only Silent Recon (Enterprise PageMantle Mode Only)

- Triggered ONLY when `env_mode == "pagemantle"` AND $\max_{i \in \mathcal{E}_t} p_{\text{chg}}(i) \ge \theta_{\text{recon}}$ (default `0.45`) AND top candidate skill $k_1 \in \{\texttt{find}, \texttt{check}, \texttt{analyze}, \texttt{diagnose}, \texttt{guard}\}$.
- Executes at most $B_{\text{recon}} = 3$ fast, read-only lookup calls guided by $C_{k_1}.\texttt{investigation\_plan}$ (within a `30s` soft cap).
- Appends retrieved factual snippets as `recon_items = [{recon_id, service, query, summary_fact}]` directly into $\mathcal{E}_t$ and recomputes `slate_hash` **before** Stage 2.
- **Why this is pivotal**: Both Advocate and Skeptic see verified enterprise ground truth (e.g., whether a claim in Slack actually contradicts Stripe/Linear or matches it!).

### Stage 2: Symmetric-Evidence Adversarial Margin Gating & Dynamic De-escalation

Both **Advocate** and **Skeptic** run asynchronously in parallel on the **exact same $\mathcal{E}_t$** (asserting `slate_hash` equality):
- **Advocate** outputs $(S_{\text{adv}} \in [0,1],\; k_{\text{adv}},\; \text{min\_channel}_{\text{adv}},\; \text{unmitigated\_loss},\; \text{cited\_ids})$.
- **Skeptic** outputs $(S_{\text{skep}} \in [0,1],\; \text{max\_channel}_{\text{skep}},\; \text{matched\_vetos},\; \text{disruption\_risk},\; \text{cited\_ids})$.

1. **Candidate Skill Selection**:
   - If $k_{\text{adv}} \neq \texttt{none}$ and $\bar{P}_{\text{skill}}(k_{\text{adv}}) \ge 0.10$, select $k^* = k_{\text{adv}}$; else $k^* = \arg\max_{k \neq \texttt{none}} \bar{P}_{\text{skill}}(k)$.

2. **Differential Margin & Epistemic Disagreement**:
   $$\Delta_t = S_{\text{adv}} - \lambda \cdot S_{\text{skep}} - \delta_{k^*}$$
   $$U_t = \min(S_{\text{adv}}, S_{\text{skep}})$$
   where $\lambda > 0$ is the global disruption-aversion weight (default `1.0`), and $\delta_{k^*} \in [-0.35, +0.35]$ is skill $k^*$'s online-calibrated margin offset (initialized to `0.0`).

3. **4-Tier Dynamic De-escalation Mapping**:
   - **Hard Need Floor**: If $S_{\text{adv}} < \theta_{\text{floor}}$ (default `0.30`), immediately set `final_tier = 0 (SILENT)`.
   - **Margin Ladder** (default thresholds $\tau_1 = 0.05,\; \tau_2 = 0.25,\; \tau_3 = 0.50$):
     $$\text{raw\_tier} = \begin{cases} 0 \;(\texttt{SILENT}) & \Delta_t < \tau_1 \\ 1 \;(\texttt{DM\_NUDGE}) & \tau_1 \le \Delta_t < \tau_2 \\ 2 \;(\texttt{THREAD\_BRIEF}) & \tau_2 \le \Delta_t < \tau_3 \\ 3 \;(\texttt{CHANNEL\_INTERVENE}) & \Delta_t \ge \tau_3 \end{cases}$$
   - **Epistemic Tension & Privacy De-escalation**:
     Let `deesc_steps = 0`:
     - If $\text{raw\_tier} \ge 2$ and $U_t \ge \tau_U$ (default `0.55`, high-stakes tension between need and disruption): `deesc_steps += 1`.
     - If Skeptic's `max_channel_skep` is strictly below `raw_tier` AND $S_{\text{skep}} \ge 0.60$: `deesc_steps += 1`.
     - Add skill's learned scope bias: `deesc_steps += int(round(gamma[k*]))`.
     - Apply privacy ceiling: if $C_{k^*}.\texttt{channel\_policy.max\_tier}$ (e.g., `guard` sensitive alerts capped at `DM_NUDGE` or `THREAD_BRIEF`) is lower, clip to ceiling.
     - Final tier (when `raw_tier >= 1`):
       $$\text{final\_tier} = \max\Big(1,\; \text{raw\_tier} - \text{deesc\_steps}\Big)$$
   - **Binary Projection (for When2Speak Benchmark)**:
     $$\hat{y}_t = \begin{cases} \texttt{SPEAK} & \text{if } \text{final\_tier} \ge 1 \\ \texttt{SILENT} & \text{if } \text{final\_tier} = 0 \end{cases}$$

### Stage 3: Applicability-Routed Skill Execution

Triggered iff `final_tier >= 1`:
1. **Budgeted Enterprise Investigation (PageMantle Mode)**:
   - Reuses all `recon_items` from Stage 1.5 without redundant calls.
   - Executes remaining tool calls strictly within $C_{k^*}.\texttt{investigation\_plan}$ (total tool budget $B_{\text{total}} \in [4, 8]$ depending on skill; soft model-time cap `90s`, hard benchmark ceiling `180s`).
2. **Channel-Shaped Synthesis**:
   - Formats output according to `final_tier` and $C_{k^*}.\texttt{response\_contract}$:
     - `DM_NUDGE`: targeted to specific user/owner, $\le 60$ words, low-pressure private heads-up + 1-line evidence link.
     - `THREAD_BRIEF`: posted to `thread_ts`, $\le 90$ words, direct finding + source ID + next step.
     - `CHANNEL_INTERVENE`: posted to main channel, structured 3-part card (`Summary` / `Verified Evidence` / `Recommended Action`), $\le 120$ words.

### Stage 4: Online Dual-Boundary Counterfactual Skill Evolution (`DB-CSE`)

After an episode completes (in `pm_online_stream_24` or online deployment), feedback arrives as a typed signal $f_t \in \{\texttt{TP\_VALIDATED}, \texttt{FP\_INTRUSION}, \texttt{FP\_SCOPE\_OVERREACH}, \texttt{FN\_OMISSION}, \texttt{CONTENT\_EVIDENCE\_GAP}\}$ plus brief textual critique.

1. **Anchor Memory Maintenance**:
   - Store verified `TP_VALIDATED` and `TN_VALIDATED` episode traces (cached $\mathcal{E}_t$, scores, label) in a per-skill rolling **Anchor Buffer** $\mathcal{A}_k$ (max size `8`, FIFO with diversity across positive/negative cases).
2. **Counterfactual Attribution Critic** (skipped if $f_t = \texttt{TP\_VALIDATED}$):
   - Diagnoses root cause $r_t \in \{\texttt{CALIBRATION\_SHIFT}, \texttt{SCOPE\_MISMATCH}, \texttt{VETO\_MISSING}, \texttt{TRIGGER\_MISSING}, \texttt{INVESTIGATION\_GAP}, \texttt{ROUTING\_CONFUSION}\}$.
3. **Bounded Numeric Calibration (Always Immediate, Zero Instability)**:
   - `FP_INTRUSION`: $\delta_{k^*} \leftarrow \text{clip}(\delta_{k^*} + 0.04,\; -0.35,\; +0.35)$
   - `FN_OMISSION`: $\delta_{k^*} \leftarrow \text{clip}(\delta_{k^*} - 0.06,\; -0.35,\; +0.35)$
   - `FP_SCOPE_OVERREACH`: $\gamma_{k^*} \leftarrow \text{clip}(\gamma_{k^*} + 0.35,\; 0.0,\; 2.0)$ *(de-escalates channel without pushing agent into silence!)*
   - `TP_VALIDATED`: mild regularization decay $\delta_{k^*} \leftarrow 0.98 \cdot \delta_{k^*}$ (prevents drift).
4. **Generalized Symbolic Patch + Counterfactual Replay Escrow Gate**:
   - When $r_t \in \{\texttt{VETO\_MISSING}, \texttt{TRIGGER\_MISSING}, \texttt{INVESTIGATION\_GAP}, \texttt{ROUTING\_CONFUSION}\}$:
     - Proposer outputs **1 atomic generalized predicate** ($\le 20$ words, checked by guardrail against proper-noun memorization).
     - Construct temporary candidate Skill Card $C'_{k^*}$.
     - **Escrow Verification**: Re-run Stage 2 (Advocate/Skeptic on cached $\mathcal{E}$, zero tool-call cost!) on:
       - The failed episode $e_{\text{fail}}$: require decision fix (e.g., $\Delta'$ flips across boundary or moves $\ge 0.10$ in the target direction).
       - $M = 3$ sampled anchor episodes from $\mathcal{A}_{k^*}$: require **0 regressions** (all anchors keep correct decision!).
     - Promote $C'_{k^*} \rightarrow \text{ACTIVE}$ **iff both checks pass**; otherwise reject and log to `skill_evolution_log.jsonl`.
5. **Hard $O(1)$ Prompt Capacity Cap**:
   - Each skill maintains at most $K_{\max} = 5$ `trigger_conditions`, $K_{\max} = 5$ `veto_conditions`, and $4$ `investigation_plan` steps.
   - Whenever insertion exceeds $K_{\max}$, invoke **Consolidate-or-Prune**: merge two overlapping rules or drop the lowest-utility rule (tracked via `utility = helpful_citations - 2 * regressions`), proving strict $O(1)$ token overhead regardless of stream length!

---

## 4. Benchmark Architecture & `PageMantle-Ext` Construction

### Benchmark 1: `duke-trust-lab/When2Speak` (Timing at Scale)
- **Source**: Hugging Face `duke-trust-lab/When2Speak` (both `token` and `dialogue` JSONL splits).
- **Exact Splits Used** (conversation-level stratified, `seed=42`, preserving ~13% `SPEAK` / ~87% `SILENT` and all 5 intervention types):
  - `When2Speak-ValCalib`: **200 turns** (~15 conversations from `validation`) — strictly for tuning $(\lambda, \tau_1)$ and verifying Jev $\theta_{\text{skip}}$.
  - `When2Speak-Pilot`: **400 turns** (~30 conversations from `test`) — fast preliminary verification.
  - `When2Speak-Main`: **2,000 turns** (~150 conversations from `test`) — main paper table (binomial SE $< 1.1\%$; comparable scale to ProACT's 3,244 turns).
  - `When2Speak-Full` (Optional): all **21,687 test turns** if desired via `--split full`.
- **Metrics**: Primary **Macro F1**, **FIR** ($\frac{\text{FP}}{\text{FP}+\text{TN}}$), **MIR** ($\frac{\text{FN}}{\text{FN}+\text{TP}}$), SPEAK P/R/F1, Margin AUROC/AUPRC, ECE, Token Reduction %, Latency, and Blinded Dialogue Judge Score (on 100 stratified TP turns).

### Benchmark 2: Deepnote Agent Initiative Benchmark + `PageMantle-Ext` (64 Tasks)
Deepnote publicly releases **8 official sample tasks** (1 per category) alongside the complete **10 GB PageMantle synthetic company database** covering **11 enterprise services** (Slack, Notion, Linear, HubSpot, Stripe, PostHog, Google Analytics, Google Search Console, Google Calendar, BigQuery, Snowflake).

To enable statistically solid evaluation AND leak-free online skill evolution, we build **`PageMantle-Ext` (64 total tasks)**:

| Partition | Count | Composition | Role in Paper | Online Evolution? |
| :--- | :---: | :--- | :--- | :---: |
| **`pm_official_8`** | **8** | Exact 8 official Deepnote public tasks (1 per category) | External Official Verification & Leaderboard Comparison | **STRICTLY HELD OUT** |
| **`pm_online_stream_24`** | **24** | 18 Action (2–3/cat) + 6 Silence Traps; split into **3 sequential Waves of 8** | Online Non-Parametric Skill Evolution (`DB-CSE` vs Reflexion/AWM) | **TRAINING STREAM** |
| **`pm_eval_32`** | **32** | 22 Action (~3/cat) + 10 Hard-Negative Silence Traps | Held-Out Generalization Test (Pre- vs Post-Evolution) | **STRICTLY HELD OUT** |

#### How the 56 New Tasks (`pm_online_stream_24` + `pm_eval_32`) Are Grounded (Zero Hallucination!)
Implemented deterministically in `src/dial_skill/benchmarks/build_pagemantle_ext.py`:
1. **Database Fact Mining**: Query real PageMantle local stores (Snowflake/BigQuery tables, Linear issues, Stripe invoices, HubSpot deals, Notion policies, PostHog/GA metrics, Slack archives) to extract verified cross-service fact pairs (e.g., an outdated figure vs. actual Snowflake value; an unresolved Linear bug matching a customer Slack complaint; a Notion compliance rule contradicting a proposed data share).
2. **Realistic Workplace Thread Synthesis**: Generate a 6–15 turn multi-participant Slack conversation around that verified fact pair, embedding realistic workplace noise (greetings, side questions, tangents) plus explicit metadata: `task_id`, `category` (1 of 8), `expected_action` (`INTERVENE` vs `SILENT`), `ideal_tier` (`SILENT`, `DM_NUDGE`, `THREAD_BRIEF`, `CHANNEL_INTERVENE`), `required_evidence_sources` (exact service + record IDs), and `reference_answer`.
3. **Hard-Negative Silence Traps (16 tasks = 6 stream + 10 eval)**: Grounded in real PageMantle entities where surface cues mimic one of the 8 skills, **but** intervention is wrong because of one of 4 realistic enterprise norms:
   - *Already Resolved*: Another teammate posts the correct answer 1–2 turns later.
   - *Explicit Deferral*: Team lead says "let's revisit after Friday's release / offline."
   - *Brainstorming Flow*: Team is exploring open hypotheses without factual error or request.
   - *Scope/Access Boundary*: Mentioned data is sensitive or unrelated to the current channel's purpose.
4. **Two-Stage Quality Assurance Gate**:
   - **Deterministic DB Existence Check**: Programmatically asserts every cited ticket ID, Notion title, Stripe/HubSpot entity, or SQL metric actually exists in PageMantle's local files.
   - **Cross-Model Ambiguity Audit**: `Terra` judge verifies unambiguous ground-truth timing & ideal channel; any ambiguous task is regenerated.

---

## 5. Complete Experiment Plan (6 Core Suites)

### Exp 1: Main Timing Benchmark (`When2Speak-Main`, 2,000 turns)
Compare on identical splits using `Luna` (primary) and `Terra` (scaling):
- `B0: Direct Chat`
- `B1: Single-Model Confidence Gate` (Jev-only & Luna-only single score)
- `B2: ProACT-Reimpl` (breakdown-aware single-perspective gating + static skill routing)
- `Ref: When2Speak Paper (arXiv:2605.05626)` (published Llama-3.3-70B SFT Macro F1 = 0.747, MIR ~ 0.50; RL MIR = 0.186–0.218)
- `Ours: DIAL-Skill` (Zero weight updates, calibrated only via 2 scalars $(\lambda, \tau_1)$ on 200 validation turns!)

### Exp 2: Main Enterprise Proactive Benchmark (`pm_official_8` + `pm_eval_32`)
Compare end-to-end across Deepnote's 4 official rubric dimensions (`Intervention/Silence`, `Content & Evidence`, `Clarity & Tone`, `Audience & Location`, each 0–10 + Overall 0–10) + Silence Accuracy + Action Recall + Timeout Rate + Cost/Task:
- `Ref: Deepnote Official Leaderboard` (`GPT-6 Luna = 6.38`, `Claude Sonnet 5.5 = 6.65`, `Claude Opus 5.5 = 7.84`, `GPT-6.1 Sol = 8.65`)
- `B0: Direct Proactive Agent` (`Luna` and `Terra`)
- `B1: Single-Perspective Gate + Tools`
- `B2: ProACT-Enterprise` (ProACT + static 8 skills + PageMantle tools, binary action)
- `Ours-v0: DIAL-Skill (Static Seed Skills)` (`Luna` and `Terra`)
- `Ours-Evolved: DIAL-Skill + DB-CSE` (after online stream on `pm_online_stream_24`, evaluated strictly on held-out `pm_official_8` + `pm_eval_32`!)

### Exp 3: Online Non-Parametric Skill Evolution (`pm_online_stream_24` $\rightarrow$ `pm_eval_32`)
Track performance across Wave 1 (tasks 1–8) $\rightarrow$ Wave 2 (9–16) $\rightarrow$ Wave 3 (17–24) $\rightarrow$ Held-out `pm_eval_32`:
- `Static-v0` (no updates)
- `Reflexion-Append` (appends textual reflection to prompt)
- `AWM-Workflow` (induces tool-workflow procedures only)
- `DB-CSE w/o Escrow Gate` (ablates counterfactual replay verification)
- `DB-CSE Full (Ours)`
Report: Wave-by-wave Score, Held-out Gain, Error-Rate Reduction, Prompt Token Growth (proving $O(1)$ vs Reflexion's linear bloat), and Escrow Gate Block/Regression Statistics.

### Exp 4: Systematic 7-Point Ablation Study
Run on `When2Speak-Main` (subset 1,000 turns or full 2,000) AND `pm_eval_32`:
- `A1: w/o Jev Triage` (raw full chat fed directly to Advocate/Skeptic)
- `A2: OpenAI Luna Triage vs. Jev System-1 Triage` (compares calibration ECE, latency, cost, F1)
- `A3: Asymmetric Evidence Feeding` (Advocate and Skeptic independently filter/select turns instead of Shared SSES — directly tests the symmetric-evidence hypothesis!)
- `A4: Single-Perspective Decomposition Ablation` (`Advocate-only` vs `Skeptic-only` vs `Single Combined Prompt` vs `Dual Advocate–Skeptic Margin`)
- `A5: w/o Dynamic De-escalation` (binary `SILENT` / `CHANNEL_INTERVENE` only)
- `A6: w/o Epistemic Disagreement ($U_t$) De-escalation` ($\Delta_t$ ladder only)
- `A7: w/o Applicability Skill Routing` (generic prompt instead of 8 Dual-Boundary Skills)

### Exp 5: Cross-Model Scaling & FIR–MIR / Cost–Score Pareto Analysis
- Sweep $\lambda \in \{0.5, 0.75, 1.0, 1.25, 1.5\}$ on `When2Speak-Main` to plot continuous FIR–MIR Pareto frontier vs. SFT/RL discrete points.
- Compare `Luna` vs `Terra` (and optional `Sol` on `pm_official_8`) on Cost (\$/task) vs. Overall Score (0–10).

### Exp 6: Judge Reliability & Qualitative Case Studies
- Blinded Dual-Judge evaluator (anonymized model tags, randomized A/B or independent rubric pass, temperature 0.0).
- Export 30 stratified episodes (`notebooks/06_analysis_and_paper_artifacts.ipynb`) for quick 30-minute human spot-check; compute Spearman $\rho$ & Cohen's $\kappa$.
- Auto-extract 3 high-impact paper case studies: (1) Silence Trap success, (2) High-$U_t$ DM/Thread de-escalation, (3) DB-CSE learned `veto_condition` transfer to held-out task.

---

## 6. Quantitative Go/No-Go & Paper-Readiness Criteria

### Phase 1: Preliminary Verification (Pilot) $\rightarrow$ Go/No-Go Gate to Full Experiments
**Scope**: `When2Speak-Pilot` (400 turns) + `pm_official_8` (8 tasks) + `pm_pilot_mini_8` (4 action + 4 silence traps) + 1 mini online wave (8 stream tasks).  
**Cost & Time**: ~1–2 hours, ~\$5–\$15 total API spend.

**PROCEED TO FULL EXPERIMENTS immediately when Conditions #1 and #2 hold AND at least 4 of 5 hold:**

| # | Check | Target Go Threshold |
| :---: | :--- | :--- |
| **1** | **When2Speak-Pilot Timing** | `DIAL-Skill (Luna)` $\ge$ `Direct Chat (Luna)` $+ 0.08$ Macro F1, AND Macro F1 $\ge 0.62$, FIR $\le 0.12$, MIR $\le 0.45$ |
| **2** | **PageMantle End-to-End** | `DIAL-Skill` $\ge$ `Direct Chat` $+ 0.6 / 10$ overall, Silence Accuracy $\ge 75\%$ (on 4 traps), Timeout Rate $\le 12.5\%$ ($\le 1/8$) |
| **3** | **Jev Triage Compression** | $\ge 40\%$ prompt token reduction on long chats ($>15$ turns) with $\ge 0.90$ recall retaining decisive turns in SSES |
| **4** | **Margin Discriminability** | $\mathbb{E}[\Delta \mid \text{SPEAK}] - \mathbb{E}[\Delta \mid \text{SILENT}] \ge 0.20$ and Margin AUROC $\ge 0.78$ |
| **5** | **DB-CSE Sanity** | Produces valid generalized patches, 0 regressions on anchor replay, held-out score non-degrading ($\ge -0.05$) |

#### Fast Diagnostic Tree (If a Pilot Metric Misses — Tune Before Full Run)
- **FIR > 0.12 (too chatty)**: Increase $\lambda$ (`1.0` $\rightarrow$ `1.25`) or raise $\tau_1$ (`0.05` $\rightarrow$ `0.12`); inspect Skeptic prompt to ensure redundancy penalties are crisp.
- **MIR > 0.45 (too quiet)**: Lower $\tau_1$ (`0.05` $\rightarrow$ `-0.05`), lower `theta_skip` (`0.15` $\rightarrow$ `0.08`), or lower hard floor $\theta_{\text{floor}}$ (`0.30` $\rightarrow$ `0.22`).
- **PageMantle Timeouts > 12.5%**: Tighten per-skill `max_tool_calls` from `8` to `5`, keep `B_recon = 2`, enable compact tool output truncation (`max_chars = 3000`).
- **Jev Quota Hit Early**: Flip `triage_backend: "openai_fallback"` in `configs/models.yaml` for high-volume sweeps, keeping Jev specifically for `pm_official_8`, `pm_eval_32`, and Ablation `A2`!

### Phase 2: Full Experiment $\rightarrow$ Paper-Ready Thresholds (Ready to Write & Submit!)
**STOP iterating and START drafting the paper once these 5 milestones are met:**

| Metric Group | Paper-Ready Target (ACL / EMNLP / NeurIPS / ICLR / AAAI) |
| :--- | :--- |
| **1. When2Speak-Main (2,000 turns)** | Macro F1 $\ge 0.72$ (zero weight updates!), FIR $\le 0.08$, MIR $\le 0.30$ (or MIR $\le 0.22$ on RL-matched Pareto point); beats `Direct Chat` by $\ge +0.12$ F1 and `ProACT-Reimpl` by $\ge +0.04$ F1 ($p < 0.01$). |
| **2. PageMantle (`pm_official_8` + `pm_eval_32`)** | `DIAL-Skill-Evolved (Luna)` $\ge 7.3/10$ (vs Deepnote official `GPT-6 Luna = 6.38`!), `DIAL-Skill-Evolved (Terra)` $\ge 8.1/10$; beats `ProACT-Enterprise` by $\ge +0.5/10$ overall and $\ge +1.0/10$ on `Audience & Location`; Silence Acc $\ge 80\%$ & Action Recall $\ge 80\%$. |
| **3. Online Evolution (`DB-CSE`)** | Held-out `pm_eval_32` improves $\ge +0.4/10$ (or $\ge 20\%$ relative error drop) over `DIAL-Skill-v0`; beats `Reflexion-Append` & `AWM-Workflow` by $\ge +0.25/10$ while prompt size stays $\le 1.15\times$ initial tokens. |
| **4. Ablation Clarity** | $\ge 5$ of 7 ablations show clear degradation when removed, explicitly confirming Jev Triage (`A1`), Symmetric Evidence (`A3`), Dual Margin (`A4`), and Dynamic De-escalation (`A5`). |
| **5. Judge Reliability** | 30-sample human spot-check achieves Spearman $\rho \ge 0.75$ or Cohen's $\kappa \ge 0.70$ against Blinded Dual-Judge. |

---

## 7. Compute Resources, Cost Budget & Hyperparameters

### Compute & Storage Requirements (ZERO GPU Needed!)
- **Colab Runtime**: **CPU Runtime** (Free 12.7 GB RAM works via disk-backed DuckDB/SQLite; **Colab Pro CPU High-RAM 51 GB** recommended for faster async concurrency and comfortable handling of PageMantle's ~10 GB dataset).
- **Google Drive Storage**: **~15–20 GB free space** (`~10 GB` cached PageMantle dataset + `~1 GB` When2Speak + `~1 GB` SQLite API caches, JSONL checkpoints, skill snapshots, figures).
- **Estimated API Budget**:
  - **Pilot Verification**: **\$5 – \$15** (Jev + `gpt-6-luna` + `gpt-6.1-terra` judge).
  - **Full Paper Experiments**: **\$45 – \$95** (across When2Speak-Main, PageMantle-Ext 64 tasks, all baselines, 7 ablations, online evolution waves, and blinded dual-judge; use persistent SQLite cache so Stage 1 & shared evaluations are never re-billed).

### Default Hyperparameter Reference (`configs/dial_skill.yaml`)

| Parameter | Default | Search / Ablation Range | Role |
| :--- | :---: | :---: | :--- |
| `triage.theta_skip` | `0.15` | `{0.0, 0.10, 0.15, 0.25}` | Fast-silence short-circuit on Jev `p_chg(t)` |
| `triage.theta_need` | `0.25` | `{0.15, 0.25, 0.35}` | Active window need threshold for short-circuit |
| `triage.k_slate` | `8` | `{6, 8, 12}` | Max messages in Symmetric Shared Evidence Slate |
| `triage.recency_beta` | `0.85` | `[0.75, 0.95]` | Recency decay weighting Jev skill priors |
| `recon.theta_recon` | `0.45` | `[0.35, 0.55]` | Trigger threshold for PageMantle Silent Recon |
| `recon.max_calls` | `3` | `{0, 2, 3}` | Hard read-only tool call cap in Stage 1.5 |
| `gate.lambda_disruption` | `1.0` | `{0.5, 0.75, 1.0, 1.25, 1.5}` | Disruption-aversion weight in $\Delta_t = S_{\text{adv}} - \lambda S_{\text{skep}} - \delta_k$ |
| `gate.theta_floor` | `0.30` | `[0.20, 0.35]` | Hard silence floor on $S_{\text{adv}}$ |
| `gate.tau_dm` ($\tau_1$) | `0.05` | `[-0.10, 0.20]` | Lower threshold for `DM_NUDGE` (and binary `SPEAK`) |
| `gate.tau_thread` ($\tau_2$) | `0.25` | `[0.15, 0.35]` | Lower threshold for `THREAD_BRIEF` |
| `gate.tau_channel` ($\tau_3$) | `0.50` | `[0.40, 0.65]` | Lower threshold for `CHANNEL_INTERVENE` |
| `gate.tau_disagreement` ($\tau_U$) | `0.55` | `[0.45, 0.70]` | Epistemic tension $U_t$ de-escalation trigger |
| `executor.soft_time_cap_sec` | `90.0` | `[60, 120]` | Soft budget inside Deepnote's 180s hard cap |
| `executor.max_tool_calls` | `7` | `{5, 7, 10}` | Per-task total tool call cap (including recon) |
| `evolution.eta_fp / eta_fn` | `0.04 / 0.06` | `[0.02, 0.08]` | Bounded asymmetric step size for $\delta_k$ |
| `evolution.eta_scope` | `0.35` | `[0.20, 0.50]` | De-escalation bias step $\gamma_k$ on `FP_SCOPE_OVERREACH` |
| `evolution.delta_clip` | `0.35` | `0.35` | Clip bound $[-0.35, +0.35]$ for $\delta_k$ |
| `evolution.escrow_anchors_m` | `3` | `{3, 5}` | Anchor replays required with 0 regressions |
| `evolution.max_rules_per_side` | `5` | `{4, 5, 6}` | Hard $O(1)$ cap on `trigger_conditions` & `veto_conditions` |
| `llm.temp_gate_judge` | `0.0` | `0.0` | Deterministic gating & blinded judging |
| `llm.temp_exec` | `0.2` | `[0.1, 0.3]` | Concise grounded natural-language replies |

---

## 8. Project Startup & VS Code + Colab + GitHub Workflow

1. **One-Time GitHub Setup (Local VS Code in `d:\python_project\when2speak`)**:
   - Verify `.gitignore` excludes `JEV_API_KEY.txt` and `OPENAI_KEY.txt` (`git status` must NOT list them!).
   - Commit the repo and push to your private/public GitHub repository (`git push -u origin main`).
2. **One-Time Google Drive Secret Setup**:
   - Upload `JEV_API_KEY.txt` and `OPENAI_KEY.txt` manually into Google Drive at:
     `MyDrive/when2speak_research/secrets/`
3. **Run in Colab (Browser or VS Code Colab Extension)**:
   - Open `notebooks/00_setup_and_preflight.ipynb` $\rightarrow$ mounts Drive, runs `git clone` (or `git pull`), installs `dial_skill` in editable mode, verifies OpenAI (`Luna`/`Terra`) & Jev APIs, and downloads/caches When2Speak + PageMantle datasets onto Google Drive ONCE.
   - Run `notebooks/01_build_pagemantle_ext.ipynb` $\rightarrow$ generates & verifies the 64-task `PageMantle-Ext` suite onto Drive.
   - Run `notebooks/02_pilot_verification.ipynb` $\rightarrow$ runs the \$5–\$15 Pilot Suite and prints an automatic **GO / NO-GO Scorecard**!
   - Once Pilot is **GREEN**, flip `MODE = "full"` and run Notebooks `03` (When2Speak Main), `04` (PageMantle Main + Online DB-CSE), `05` (7 Ablations + Scaling/Pareto), and `06` (Dual-Judge + LaTeX Tables & Vector Figures).
4. **Daily Iteration Loop**:
   - Edit Python code locally in VS Code $\rightarrow$ `git commit && git push` $\rightarrow$ run Cell 1 (`!git pull`) in Colab $\rightarrow$ `%autoreload 2` picks up changes instantly while all datasets, SQLite API caches, and JSONL run checkpoints persist safely on Google Drive!
