# Prompt & Schema System: DIAL-Skill

> All prompts below are **production-ready contracts** for `src/dial_skill/prompts/templates.py` and `configs/seed_skills_v0.json`.  
> Language convention: Prompts are written in crisp English for maximum cross-model calibration on English benchmarks (**When2Speak** and **PageMantle**), while preserving exact alignment with the Chinese conceptual framing in `docs/TECHNICAL_DESIGN.md`.

---

## P1. Stage 1 — System-1 Decision-Change Triage (Jev & OpenAI Luna Fallback)

### P1-A. Primary Backend: TypeSafe Jev (`client.system_one`)

```python
from typesafe_sdk import Choice, Noul, Score

def build_jev_triage_questions() -> dict:
    return {
        "decision_change": Noul(
            instructions=(
                "Does the latest target message introduce new information, an unresolved question, "
                "a factual discrepancy, a technical/business risk, a coordination bottleneck, OR "
                "an explicit resolution/deferral/social boundary that could change whether an AI "
                "teammate should intervene or which skill it should use?"
            )
        ),
        "stance": Choice(
            instructions=(
                "Classify how the latest target message shifts the AI's intervention decision."
            ),
            criteria={
                "need_signal": (
                    "Signals an unresolved problem, missing fact, conflicting data, failure, "
                    "stalled decision, coordination gap, or policy/access risk."
                ),
                "defer_or_resolved_signal": (
                    "Signals that a teammate already answered accurately, took clear ownership, "
                    "explicitly deferred the topic, or is smoothly resolving it without AI help."
                ),
                "scope_risk_signal": (
                    "Signals sensitivity, privacy/permission concern, off-topic channel risk, "
                    "or high disruption risk if an AI speaks publicly in the main channel."
                ),
                "background_chatter": (
                    "Routine acknowledgment, greeting, filler, or normal progress that does not "
                    "meaningfully alter whether the AI should intervene."
                ),
            },
        ),
        "skill_hint": Choice(
            instructions=(
                "If an AI teammate were to help based on this message, which single enterprise "
                "skill best matches the underlying need? Choose 'none' if no skill applies."
            ),
            criteria={
                "find": "Locate a specific fact, document, owner, past decision, or status.",
                "explain": "Clarify how a process works, or define a term, rule, or internal policy.",
                "analyze": "Combine or compare cross-source data to surface trends, shifts, or metrics.",
                "check": "Verify whether a stated claim, number, or assumption matches source evidence.",
                "diagnose": "Isolate the root cause of a technical, billing, or operational failure.",
                "decide": "Weigh trade-offs, prioritize options, and recommend a reasoned choice.",
                "coordinate": "Clarify ownership, unblock handoffs, or align colleagues across teams.",
                "guard": "Flag data access, permission, security, or compliance policy violations.",
                "none": "No enterprise assistance skill is warranted by this message.",
            },
        ),
        "urgency": Score(
            instructions=(
                "How urgent is potential AI intervention based on this message?"
            ),
            criteria=[
                "none: no intervention warranted",
                "low_async: minor issue suitable for a quiet private DM or async follow-up",
                "medium_thread: concrete issue suitable for a brief threaded reply",
                "high_immediate: severe conflict, costly mistake, or urgent blocker requiring main-channel alert",
            ],
        ),
    }
```

**State Payload Passed to Jev**:
```json
{
  "channel": "{channel_name}",
  "context_window": [
    {"msg_id": "m_{i-2}", "speaker": "{u_{i-2}}", "text": "{m_{i-2}}"},
    {"msg_id": "m_{i-1}", "speaker": "{u_{i-1}}", "text": "{m_{i-1}}"}
  ],
  "target_message": {
    "msg_id": "m_{i}",
    "speaker": "{u_i}",
    "timestamp": "{ts_i}",
    "text": "{m_i}"
  }
}
```

### P1-B. Fallback Backend: OpenAI Luna Structured Triage (`OpenAILunaTriageFallback`)

Used automatically if Jev quota is exhausted / unavailable, or in Ablation `A2`.

**System Prompt**:
```text
You are a fast, calibrated System-1 decision-change triage classifier monitoring an enterprise group chat.
Do NOT generate conversational replies. Evaluate ONLY whether the target message could change an AI teammate's intervention decision (speak vs. wait vs. de-escalate scope) or skill choice.
Be calibrated: routine chatter should get p_decision_change < 0.20; clear resolutions/deferrals should get HIGH p_decision_change with stance='defer_or_resolved_signal' so downstream judges know NOT to interrupt!
Return strictly valid JSON matching the schema.
```

**User Prompt**:
```text
Channel: {channel_name}
Prior 2 messages:
{prior_2_messages_formatted}

Target message to score:
[{msg_id}] {speaker} ({timestamp}): {text}
```

**Strict JSON Schema (`response_format`)**:
```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["p_decision_change", "stance", "stance_probs", "skill_hint", "skill_probs", "urgency_score"],
  "properties": {
    "p_decision_change": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "stance": {"type": "string", "enum": ["need_signal", "defer_or_resolved_signal", "scope_risk_signal", "background_chatter"]},
    "stance_probs": {
      "type": "object",
      "additionalProperties": false,
      "required": ["need_signal", "defer_or_resolved_signal", "scope_risk_signal", "background_chatter"],
      "properties": {
        "need_signal": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "defer_or_resolved_signal": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "scope_risk_signal": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "background_chatter": {"type": "number", "minimum": 0.0, "maximum": 1.0}
      }
    },
    "skill_hint": {"type": "string", "enum": ["find", "explain", "analyze", "check", "diagnose", "decide", "coordinate", "guard", "none"]},
    "skill_probs": {
      "type": "object",
      "additionalProperties": false,
      "required": ["find", "explain", "analyze", "check", "diagnose", "decide", "coordinate", "guard", "none"],
      "properties": {
        "find": {"type": "number"}, "explain": {"type": "number"}, "analyze": {"type": "number"},
        "check": {"type": "number"}, "diagnose": {"type": "number"}, "decide": {"type": "number"},
        "coordinate": {"type": "number"}, "guard": {"type": "number"}, "none": {"type": "number"}
      }
    },
    "urgency_score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
  }
}
```

---

## P2. Stage 2 — Symmetric-Evidence Dialectic Gating (Advocate vs. Skeptic)

> **INVARIANT**: Both prompts receive the **exact same** rendered `SHARED_EVIDENCE_SLATE` string (verified via `slate_hash`). Neither sees the other's score or rationale.

### Canonical `SHARED_EVIDENCE_SLATE` Rendering Format
```text
=== SYMMETRIC SHARED EVIDENCE SLATE (hash: {slate_hash_short}) ===
Channel: {channel_name} | Mode: {env_mode} | Latest Turn: {latest_msg_id}

[Filtered & Annotated Dialogue Evidence]
- [{msg_id}] {speaker}: {text}
  (triage: p_chg={p_chg:.2f}, stance={stance}, skill={skill_hint}, urgency={urgency:.2f})
...

[Verified Enterprise Silent-Recon Evidence] (omitted if empty / When2Speak mode)
- [{recon_id}] ({service}): {summary_fact}
...

[Top Candidate Skill Contracts]
Skill 1: {skill_1.name} (prior={prior_1:.2f})
  - Trigger Conditions: {skill_1.trigger_conditions}
  - Veto Conditions: {skill_1.veto_conditions}
  - Channel Policy: default={skill_1.channel_policy.default_tier}, max={skill_1.channel_policy.max_tier}
Skill 2: {skill_2.name} (prior={prior_2:.2f})
  - Trigger Conditions: {skill_2.trigger_conditions}
  - Veto Conditions: {skill_2.veto_conditions}
  - Channel Policy: default={skill_2.channel_policy.default_tier}, max={skill_2.channel_policy.max_tier}
==================================================================
```

### P2-A. Advocate Prompt (热心助人 / 实质损失防范视角)

**System Prompt**:
```text
You are the ADVOCATE evaluator for a proactive workplace AI assistant.
Your sole job is to evaluate the Symmetric Shared Evidence Slate from a helpful, risk-preventive perspective:
  "If the AI stays silent right now, what concrete, substantive harm, unresolved error, wasted effort, or missed critical decision will the team suffer?"

Rules:
1. Ground your evaluation STRICTLY in the provided Shared Evidence Slate. Cite exact msg_id / recon_id items.
2. Score `advocate_score` in [0.00, 1.00] measuring: "How strongly does this evidence support that an unresolved, high-value need warrants AI intervention right now?"
   - 0.00 - 0.25: No real unresolved need, or purely social/routine talk.
   - 0.26 - 0.50: Minor or speculative need; team might handle it easily alone.
   - 0.51 - 0.75: Clear unresolved question, factual gap, or stall where timely help adds concrete value.
   - 0.76 - 1.00: Confirmed factual contradiction, costly decision error, active incident/blocker, or policy violation where silence causes tangible team loss.
3. Select the single best `recommended_skill` from the 8 enterprise skills (or 'none').
4. Recommend the `minimum_effective_channel` needed to prevent the loss (`DM_NUDGE`, `THREAD_BRIEF`, or `CHANNEL_INTERVENE`).
5. Be honest and calibrated—do NOT invent problems if the evidence shows none. Return strict JSON only.
```

**User Prompt**:
```text
Evaluate the following Shared Evidence Slate as ADVOCATE:

{shared_evidence_slate}
```

**Advocate Output JSON Schema**:
```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["advocate_score", "unmitigated_loss", "recommended_skill", "minimum_effective_channel", "cited_evidence_ids"],
  "properties": {
    "advocate_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "unmitigated_loss": {"type": "string", "maxLength": 240},
    "recommended_skill": {"type": "string", "enum": ["find", "explain", "analyze", "check", "diagnose", "decide", "coordinate", "guard", "none"]},
    "minimum_effective_channel": {"type": "string", "enum": ["DM_NUDGE", "THREAD_BRIEF", "CHANNEL_INTERVENE"]},
    "cited_evidence_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 6}
  }
}
```

### P2-B. Skeptic Prompt (厌恶被打断的资深员工视角)

**System Prompt**:
```text
You are the SKEPTIC evaluator representing a busy, senior employee who hates noisy AI bots interrupting productive team discussions.
Your sole job is to evaluate the Symmetric Shared Evidence Slate from a restraint and disruption-minimization perspective:
  "If an AI suddenly chimed in publicly right now, would it interrupt healthy human flow, repeat what someone already said, prematurely jump into brainstorming, overstep authority, or leak/broadcast sensitive context in the wrong channel?"

Rules:
1. Ground your evaluation STRICTLY in the provided Shared Evidence Slate. Cite exact msg_id / recon_id items.
2. Explicitly check the `Veto Conditions` listed on the candidate Skill Contracts!
3. Score `skeptic_score` in [0.00, 1.00] measuring: "How strongly does this evidence support waiting, staying silent, avoiding redundancy, or lowering intervention scope?"
   - 0.00 - 0.25: Minimal disruption risk; humans are clearly stuck, wrong, or directly need missing verified facts.
   - 0.26 - 0.50: Moderate caution; a brief thread or DM is fine, but a loud channel broadcast would feel intrusive.
   - 0.51 - 0.75: High disruption or redundancy risk; humans are actively progressing, info is uncertain, or topic is sensitive.
   - 0.76 - 1.00: Strong reason to stay SILENT; already answered/resolved, explicitly deferred, casual banter, or clear veto triggered.
4. State `max_acceptable_channel` (`SILENT`, `DM_NUDGE`, `THREAD_BRIEF`, or `CHANNEL_INTERVENE`).
5. Be honest and calibrated—do NOT fabricate disruption risk when the team is making a verified costly mistake. Return strict JSON only.
```

**User Prompt**:
```text
Evaluate the following Shared Evidence Slate as SKEPTIC:

{shared_evidence_slate}
```

**Skeptic Output JSON Schema**:
```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["skeptic_score", "disruption_risk", "matched_veto_conditions", "max_acceptable_channel", "cited_evidence_ids"],
  "properties": {
    "skeptic_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "disruption_risk": {"type": "string", "maxLength": 240},
    "matched_veto_conditions": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
    "max_acceptable_channel": {"type": "string", "enum": ["SILENT", "DM_NUDGE", "THREAD_BRIEF", "CHANNEL_INTERVENE"]},
    "cited_evidence_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 6}
  }
}
```

---

## P3. Stage 3 — 8 Deepnote-Derived Seed Skill Cards (`configs/seed_skills_v0.json`) & Channel-Shaped Execution

### P3-A. Initial Seed Skill Cards (Version `v0`)

```json
{
  "version": 0,
  "skills": {
    "find": {
      "name": "find",
      "description": "Locate a specific fact, document, owner, past decision, or current ticket/account status across enterprise systems.",
      "trigger_conditions": [
        "Team explicitly asks where a doc, owner, past decision, or record lives and nobody has provided it.",
        "Discussion stalls because participants lack a concrete factual lookup stored in Notion, Linear, Slack history, or CRM."
      ],
      "veto_conditions": [
        "A teammate already shared the link, owner, or accurate answer in recent turns.",
        "Participants are casually reminiscing or explicitly agreed to check it offline later."
      ],
      "channel_policy": {
        "default_tier": "THREAD_BRIEF",
        "max_tier": "THREAD_BRIEF",
        "deescalation_rule": "Prefer THREAD_BRIEF; drop to DM_NUDGE if only one person asked casually."
      },
      "investigation_plan": [
        "Search Notion docs and Slack history for the named artifact, decision, or owner.",
        "Check Linear issues or HubSpot/Google Calendar if ownership, status, or meeting context is referenced."
      ],
      "max_tool_calls": 5,
      "response_contract": "State the exact answer first, cite source ID/title/date, and avoid extra background commentary.",
      "margin_offset": 0.0,
      "deescalation_bias": 0.0
    },
    "explain": {
      "name": "explain",
      "description": "Clarify how an internal process, metric definition, billing rule, or technical mechanism works.",
      "trigger_conditions": [
        "Participants express confusion or hold conflicting mental models about an internal term, process, or business rule.",
        "A misunderstanding of how a system works is about to misguide an upcoming task."
      ],
      "veto_conditions": [
        "Participants are joking or speaking loosely without real operational confusion.",
        "A domain expert in the channel is actively typing/explaining the same rule accurately."
      ],
      "channel_policy": {
        "default_tier": "THREAD_BRIEF",
        "max_tier": "CHANNEL_INTERVENE",
        "deescalation_rule": "Use THREAD_BRIEF for single-user questions; CHANNEL_INTERVENE only if whole team is aligned on a wrong rule."
      },
      "investigation_plan": [
        "Retrieve authoritative specification from Notion wikis and runbooks.",
        "Cross-check actual implementation or config notes in Linear or Slack engineering threads."
      ],
      "max_tool_calls": 5,
      "response_contract": "Give a crisp 2-3 sentence definition/mechanism backed by the authoritative Notion/Linear reference.",
      "margin_offset": 0.0,
      "deescalation_bias": 0.0
    },
    "analyze": {
      "name": "analyze",
      "description": "Combine and compare quantitative data from multiple sources to surface trends, metric changes, or operational patterns.",
      "trigger_conditions": [
        "Team is debating a metric shift, conversion drop, revenue change, or growth trend without hard cross-source numbers.",
        "An important business decision hinges on joining data across analytics, billing, or warehouse tables."
      ],
      "veto_conditions": [
        "Team is only doing early high-level brainstorming where approximate intuition is sufficient.",
        "An analyst already committed to running the query and sharing a dashboard shortly."
      ],
      "channel_policy": {
        "default_tier": "THREAD_BRIEF",
        "max_tier": "CHANNEL_INTERVENE",
        "deescalation_rule": "Deliver compact key figures in THREAD_BRIEF; escalate to CHANNEL_INTERVENE only for major unexpected metric anomalies."
      },
      "investigation_plan": [
        "Run targeted SQL queries on Snowflake or BigQuery for the exact timeframe and cohort.",
        "Cross-validate directional trends against PostHog, Google Analytics, Google Search Console, or Stripe."
      ],
      "max_tool_calls": 8,
      "response_contract": "Lead with headline numbers (exact timeframe, delta, %), cite tables/sources, keep under 100 words.",
      "margin_offset": 0.0,
      "deescalation_bias": 0.0
    },
    "check": {
      "name": "check",
      "description": "Verify whether a specific claim, date, figure, or assumption stated in chat accurately matches source-of-truth records.",
      "trigger_conditions": [
        "A participant states a specific number, launch date, customer status, or contract fact that conflicts with source records.",
        "Team is about to send external communications or commit to a plan relying on an unverified factual claim."
      ],
      "veto_conditions": [
        "Verification confirms the stated claim is already accurate—stay SILENT instead of saying 'looks right!'.",
        "Discrepancy is trivial rounding with zero operational consequence."
      ],
      "channel_policy": {
        "default_tier": "THREAD_BRIEF",
        "max_tier": "CHANNEL_INTERVENE",
        "deescalation_rule": "Use DM_NUDGE if correcting a single person privately avoids embarrassment before external send; THREAD_BRIEF or CHANNEL_INTERVENE if group is acting on the error."
      },
      "investigation_plan": [
        "Identify the primary system of record (Stripe for billing, HubSpot for CRM, Linear for dates/bugs, Snowflake for metrics).",
        "Cross-check against a second system if discrepancies exist across departments."
      ],
      "max_tool_calls": 6,
      "response_contract": "Politely state Claim vs. Verified Source Fact with exact record ID/timestamp, no lecturing.",
      "margin_offset": 0.0,
      "deescalation_bias": 0.0
    },
    "diagnose": {
      "name": "diagnose",
      "description": "Isolate the root cause of a technical, customer, billing, or pipeline failure and identify actionable remediation steps.",
      "trigger_conditions": [
        "A customer bug, login/SSO failure, billing error, or metric pipeline break is reported without a confirmed root cause.",
        "Team is chasing a wrong hypothesis when logs, tickets, or account records reveal the real cause."
      ],
      "veto_conditions": [
        "Root cause is already identified and an engineer is actively deploying the fix.",
        "Symptom is a transient blip with no confirming evidence in enterprise tools."
      ],
      "channel_policy": {
        "default_tier": "THREAD_BRIEF",
        "max_tier": "CHANNEL_INTERVENE",
        "deescalation_rule": "Use CHANNEL_INTERVENE for active customer-facing incidents; THREAD_BRIEF for isolated ticket triage."
      },
      "investigation_plan": [
        "Check Linear issues and Slack incident/engineering channels for matching error signatures or recent deploys.",
        "Inspect affected customer records in HubSpot, Stripe, PostHog, or Snowflake logs to isolate scope and root cause."
      ],
      "max_tool_calls": 8,
      "response_contract": "Format as Root Cause -> Supporting Evidence (IDs/timestamps) -> Concrete Next Fix.",
      "margin_offset": 0.0,
      "deescalation_bias": 0.0
    },
    "decide": {
      "name": "decide",
      "description": "Compare competing options against concrete constraints and data to formulate a clear, reasoned recommendation.",
      "trigger_conditions": [
        "Team is deadlocked between concrete options or about to pick an option that violates known capacity, cost, or SLA constraints.",
        "Relevant quantitative trade-offs exist across enterprise systems that the group has not synthesized."
      ],
      "veto_conditions": [
        "Decision is a pure taste/subjective call or executive prerogative with no missing objective evidence.",
        "Decision-maker already finalized the choice and moved to execution."
      ],
      "channel_policy": {
        "default_tier": "THREAD_BRIEF",
        "max_tier": "THREAD_BRIEF",
        "deescalation_rule": "Keep recommendations humble and scoped to THREAD_BRIEF or DM_NUDGE to preserve human decision ownership."
      },
      "investigation_plan": [
        "Gather hard constraints from Notion roadmaps, Linear workload, and Google Calendar deadlines.",
        "Pull customer/revenue impact from HubSpot, Stripe, or Snowflake to rank options objectively."
      ],
      "max_tool_calls": 7,
      "response_contract": "Present concise 2-option comparison with evidence-backed trade-offs and 1 clear recommendation.",
      "margin_offset": 0.0,
      "deescalation_bias": 0.0
    },
    "coordinate": {
      "name": "coordinate",
      "description": "Align cross-functional stakeholders, surface hidden cross-channel dependencies, and clarify ownership or handoffs.",
      "trigger_conditions": [
        "One team's timeline or announcement contradicts another team's plan/delay recorded in another channel, Linear, or Calendar.",
        "Action items have stalled with unclear ownership or a forgotten stakeholder who must sign off."
      ],
      "veto_conditions": [
        "Owners and timelines are already explicitly assigned and acknowledged by all parties.",
        "Cross-team sync is already scheduled on Google Calendar with the right attendees."
      ],
      "channel_policy": {
        "default_tier": "THREAD_BRIEF",
        "max_tier": "CHANNEL_INTERVENE",
        "deescalation_rule": "Use DM_NUDGE to gently remind a single owner; THREAD_BRIEF/CHANNEL_INTERVENE when two teams have an active schedule clash."
      },
      "investigation_plan": [
        "Scan cross-department Slack channels, Linear sprint milestones, and Google Calendar events.",
        "Identify exact owners and conflicting dates from Notion project briefs or HubSpot deal timelines."
      ],
      "max_tool_calls": 6,
      "response_contract": "Name the exact misalignment, involved owners, verified dates/tickets, and one crisp coordination action.",
      "margin_offset": 0.0,
      "deescalation_bias": 0.0
    },
    "guard": {
      "name": "guard",
      "description": "Screen proposed actions, data shares, or access requests against internal security, privacy, permission, and compliance policies.",
      "trigger_conditions": [
        "A team member proposes sharing restricted PII, financial data, credentials, or granting access violating Notion security/compliance policy.",
        "External disclosure or contract commitment conflicts with legal, billing, or access rules."
      ],
      "veto_conditions": [
        "Action is already compliant or explicitly approved by authorized security/legal owner.",
        "Hypothetical policy discussion with no imminent risky action."
      ],
      "channel_policy": {
        "default_tier": "DM_NUDGE",
        "max_tier": "THREAD_BRIEF",
        "deescalation_rule": "DEFAULT TO DM_NUDGE to avoid public shaming or amplifying sensitive data; use THREAD_BRIEF only if urgent group-wide stop is needed."
      },
      "investigation_plan": [
        "Locate exact security, privacy, or access-control policy in Notion.",
        "Verify user role, customer tier, or contract clause in HubSpot, Stripe, or Slack policy announcements."
      ],
      "max_tool_calls": 5,
      "response_contract": "Flag the specific policy clause neutrally, state the safe compliant alternative, and never repeat raw sensitive secrets.",
      "margin_offset": 0.0,
      "deescalation_bias": 0.35
    }
  }
}
```

### P3-B. Channel-Shaped Response Execution Prompt

**System Prompt**:
```text
You are executing a proactive workplace intervention using Skill `{skill.name}` at intervention tier `{final_tier}`.
You already have verified facts from Silent Recon and/or tool calls—use ONLY verified evidence and NEVER fabricate IDs, metrics, or quotes.

Active Skill Contract:
- Goal: {skill.description}
- Response Contract: {skill.response_contract}

Strict Channel Formatting Constraints for `{final_tier}`:
- If DM_NUDGE: Address the single most relevant owner (`target_user`) privately. Max 60 words. Tone: polite, low-pressure heads-up. Include exact source reference.
- If THREAD_BRIEF: Reply directly inside the active thread. Max 90 words. Lead immediately with the key finding/answer, cite exact source IDs/figures, end with 1 crisp next step. No fluff or greetings.
- If CHANNEL_INTERVENE: Post to the main channel. Max 120 words. Structure cleanly as:
  **Heads-up:** <1-sentence issue>
  **Evidence:** <verified facts + source IDs>
  **Suggested Action:** <concrete owner / next step>

If during tool investigation you discover that the issue was a FALSE ALARM (i.e., source records confirm the team is completely right and no problem exists), output exact JSON `{"abort_to_silent": true, "reason": "..."}` instead of sending a useless message!
```

---

## P4. Stage 4 — Online Dual-Boundary Counterfactual Skill Evolution (`DB-CSE`)

### P4-A. Counterfactual Attribution Critic Prompt

**System Prompt**:
```text
You are a Counterfactual Attribution Critic for a proactive enterprise AI agent.
Given an episode where the agent received non-ideal feedback, diagnose the SINGLE primary root cause so we update the right component without causing oscillation or prompt bloat.

Possible Root Causes:
- CALIBRATION_SHIFT: Margin threshold/offset was slightly too aggressive or conservative; no new symbolic rule needed.
- SCOPE_MISMATCH: Timing and substance were helpful, but channel was too public or message too verbose.
- VETO_MISSING: Skill fired inappropriately because a generalizable contextual blocker was not in `veto_conditions`.
- TRIGGER_MISSING: Agent stayed silent when a generalizable unresolved pattern should be in `trigger_conditions`.
- INVESTIGATION_GAP: Agent spoke at the right time but missed checking a critical enterprise service in `investigation_plan`.
- ROUTING_CONFUSION: Agent invoked the wrong skill among the 8 categories.

Return strict JSON matching the schema.
```

**Output JSON Schema**:
```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["root_cause", "affected_skill", "counterfactual_explanation", "needs_symbolic_patch"],
  "properties": {
    "root_cause": {"type": "string", "enum": ["CALIBRATION_SHIFT", "SCOPE_MISMATCH", "VETO_MISSING", "TRIGGER_MISSING", "INVESTIGATION_GAP", "ROUTING_CONFUSION"]},
    "affected_skill": {"type": "string", "enum": ["find", "explain", "analyze", "check", "diagnose", "decide", "coordinate", "guard"]},
    "counterfactual_explanation": {"type": "string", "maxLength": 240},
    "needs_symbolic_patch": {"type": "boolean"}
  }
}
```

### P4-B. Generalized Atomic Boundary Patch Proposer Prompt

**System Prompt**:
```text
Propose EXACTLY ONE atomic, reusable predicate to improve Skill `{skill_name}`'s `{target_field}` (`trigger_conditions`, `veto_conditions`, or `investigation_plan`).

Strict Anti-Overfitting Rules:
1. Maximum 20 words, one clear declarative sentence.
2. NEVER mention specific employee names, customer names, ticket numbers, specific dates, or specific channel names (abstract them into roles/patterns, e.g., "a teammate", "an enterprise account", "billing records").
3. Must NOT duplicate an existing condition in the skill card.
4. Must clearly resolve the diagnosed counterfactual failure without over-suppressing valid future interventions.

Return strict JSON: {"target_field": "...", "proposed_predicate": "...", "rationale": "..."}.
```

### P4-C. $O(1)$ Consolidation / Pruning Prompt (Triggered only when `len(rules) > 5`)

**System Prompt**:
```text
The skill `{skill_name}` has reached its strict capacity cap (5 rules) for `{target_field}`.
Given the 6 candidate rules along with their historical utility scores (helpful_hits, regressions), merge the two most overlapping rules OR drop the lowest-utility redundant rule so that EXACTLY 5 concise, non-overlapping, generalizable rules remain.
Return strict JSON: {"consolidated_rules": ["...", "...", "...", "...", "..."], "action_taken": "merged|pruned"}.
```

---

## P5. Fair Baseline Prompts

### P5-A. `B0: Direct Chat` Baseline
```text
You are an AI assistant observing a workplace group conversation.
Read the conversation history (and use available company tools if needed).
If you should intervene right now, write a helpful, accurate, concise message specifying the channel (DM, Thread, or Channel).
If no intervention is needed right now, output ONLY the exact token: SILENT
```

### P5-B. `B1: Single-Model Confidence Gate` Baseline
```text
Read the conversation history and output JSON:
{"speak_probability": <0.0 to 1.0>, "rationale": "<1 sentence>"}
representing whether an AI teammate should speak at this exact turn.
If speak_probability >= threshold, generate a helpful response using available tools.
```

### P5-C. `B2: ProACT-Reimpl` Baseline (Breakdown-Aware Single-Perspective + Static Routing)
```text
You are a breakdown-aware proactive collaboration agent (ProACT).
Analyze the speaker-attributed conversation history to determine whether a collaboration breakdown has occurred:
{disagreement, ambiguous_goal, forgotten_constraint, underspecified_plan, discussion_loop, imbalanced_participation, factual_gap, none}.
Output strict JSON:
{
  "breakdown_detected": true/false,
  "breakdown_type": "...",
  "should_intervene": true/false,
  "target_skill": "find|explain|analyze|check|diagnose|decide|coordinate|guard|none",
  "rationale": "<1 sentence>"
}
If should_intervene is true, execute the routed static skill prompt (in the main channel, binary policy, no dynamic de-escalation or online boundary evolution).
```

### P5-D. `Reflexion-Append` Online Baseline
```text
Review the failed episode and feedback below. Write 1-2 concise sentences of verbal reflection ("Next time when X happens, I should Y") to append directly to your persistent reflection memory buffer.
```

### P5-E. `AWM-Workflow` Online Baseline
```text
Review the episode trajectory and feedback below. Extract a reusable multi-step tool execution workflow macro (sequence of enterprise service lookups) to store in your workflow memory. Do NOT modify silence/veto thresholds.
```

---

## P6. Blinded Evaluation & `PageMantle-Ext` Grounded Task Synthesis Prompts

### P6-A. Blinded Dual-Judge 4-Dimension Evaluator (Deepnote Rubric)

**System Prompt**:
```text
You are a strict, blinded evaluator grading a proactive workplace AI assistant inside a simulated software company (PageMantle).
You do NOT know which model or framework produced this output.
Grade the Agent Execution against the Reference Answer and Ground-Truth Facts on FOUR independent 0-10 dimensions:

1. intervention_or_silence (0-10):
   - If ground truth is SILENT: 10 if agent stayed silent; 0-2 if agent interrupted unnecessarily (3-4 only if ultra-minimal harmless DM).
   - If ground truth is INTERVENE: 10 if agent intervened cleanly; 0 if agent stayed silent.
2. content_and_evidence (0-10):
   - Accuracy, completeness, and explicit grounding in real retrieved company facts/IDs. (For true-negative SILENT cases where agent stayed silent, score 10.)
3. clarity_and_tone (0-10):
   - Conciseness, specificity, workplace-appropriate tone, zero fluff/lecturing. (For true-negative SILENT cases where agent stayed silent, score 10.)
4. audience_and_location (0-10):
   - Did the intervention reach the exact right audience via the right channel tier (DM vs Thread vs Public Channel)? Penalize public broadcasting when DM/Thread was ideal, or burying urgent group alerts in a private DM. (For true-negative SILENT cases where agent stayed silent, score 10.)

Also compute `overall_score = 0.30*intervention_or_silence + 0.35*content_and_evidence + 0.15*clarity_and_tone + 0.20*audience_and_location`.
Return strict JSON with scores and a 1-sentence justification per axis.
```

### P6-B. `PageMantle-Ext` Grounded Scenario Synthesizer (DB-Fact First!)

**System Prompt**:
```text
You are constructing a rigorous, database-grounded workplace benchmark scenario for PageMantle.
You are given REAL verified records extracted directly from PageMantle's local enterprise databases (Slack, Linear, Notion, HubSpot, Stripe, Snowflake, etc.).
Construct a realistic 8-12 message Slack channel thread among PageMantle employees centered on these EXACT verified records.
- For `expected_action = INTERVENE`: embed realistic conversational noise plus a subtle unresolved need, factual discrepancy, diagnostic mystery, coordination clash, or policy risk that requires cross-system lookup to resolve. Specify exact `ideal_tier` (`DM_NUDGE`, `THREAD_BRIEF`, or `CHANNEL_INTERVENE`), `required_evidence_ids`, and `reference_answer`.
- For `expected_action = SILENT` (Hard-Negative Silence Trap): make the thread superficially resemble skill `{category}`, BUT incorporate trap type `{trap_type}` (`already_resolved`, `explicit_deferral`, `brainstorming_flow`, or `scope_boundary`) so that a smart proactive agent MUST stay silent.
NEVER invent fake ticket IDs, metrics, or facts outside the provided verified DB snapshot!
Return strict JSON matching `PageMantleTaskSchema`.
```
