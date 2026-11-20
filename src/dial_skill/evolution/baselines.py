"""Online evolution baseline comparators for Exp 3: Reflexion-Append and AWM-Workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dial_skill.core.triage import estimate_tokens
from dial_skill.llm.openai_client import OpenAIClient
from dial_skill.prompts.templates import AWM_WORKFLOW_PROMPT, REFLEXION_APPEND_PROMPT
from dial_skill.schemas import EpisodeFeedback, EpisodeResult


class ReflexionAppendUpdater:
    """Appends free-form verbal reflections unconditionally to a persistent memory buffer (O(T) growth)."""

    def __init__(
        self,
        openai_client: OpenAIClient,
        model: str,
        seed: int = 42,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.seed = seed
        self.reflections: list[dict[str, str]] = []

    async def process_episode(
        self,
        episode: EpisodeResult,
        feedback: EpisodeFeedback,
    ) -> str | None:
        if feedback.feedback_type in ("TP_VALIDATED", "TN_VALIDATED"):
            return None
        user_prompt = (
            f"Episode {episode.sample_id} (predicted={episode.gate.final_tier.value}, "
            f"skill={episode.gate.selected_skill})\n"
            f"Feedback: {feedback.feedback_type} — {feedback.critique}\n"
            f"Response: {(episode.response_text or 'SILENT')[:300]}"
        )
        res = await self.openai_client.complete_text(
            model=self.model,
            messages=[
                {"role": "system", "content": REFLEXION_APPEND_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            seed=self.seed,
        )
        text = res.content.strip()
        if text:
            self.reflections.append(
                {
                    "sample_id": episode.sample_id,
                    "feedback_type": feedback.feedback_type,
                    "reflection": text,
                }
            )
        return text

    def render_memory_block(self) -> str:
        if not self.reflections:
            return ""
        lines = ["[Reflexion Persistent Memory]"]
        for idx, item in enumerate(self.reflections, start=1):
            lines.append(f"{idx}. ({item['feedback_type']}) {item['reflection']}")
        return "\n".join(lines)

    def estimate_memory_tokens(self) -> int:
        return estimate_tokens(self.render_memory_block())

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.reflections, indent=2, ensure_ascii=False), encoding="utf-8")


class AWMWorkflowUpdater:
    """Agent Workflow Memory baseline: induces reusable tool workflows only (never modifies vetos/thresholds)."""

    def __init__(
        self,
        openai_client: OpenAIClient,
        model: str,
        seed: int = 42,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.seed = seed
        self.workflows: list[dict[str, Any]] = []

    async def process_episode(
        self,
        episode: EpisodeResult,
        feedback: EpisodeFeedback,
    ) -> str | None:
        tool_trace = episode.metadata.get("tool_trace", [])
        # AWM induces procedural tool macros from action/evidence episodes (does NOT alter silence vetos)
        if feedback.feedback_type == "TN_VALIDATED":
            return None
        user_prompt = (
            f"Episode {episode.sample_id} (skill={episode.gate.selected_skill})\n"
            f"Feedback: {feedback.feedback_type} — {feedback.critique}\n"
            f"Tool Trace: {json.dumps(tool_trace, ensure_ascii=False)[:800]}"
        )
        res = await self.openai_client.complete_text(
            model=self.model,
            messages=[
                {"role": "system", "content": AWM_WORKFLOW_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            seed=self.seed,
        )
        workflow_text = res.content.strip()
        if workflow_text:
            self.workflows.append(
                {
                    "sample_id": episode.sample_id,
                    "skill": episode.gate.selected_skill,
                    "workflow": workflow_text,
                }
            )
        return workflow_text

    def render_workflows_block(self) -> str:
        if not self.workflows:
            return ""
        lines = ["[AWM Induced Tool Workflows]"]
        for idx, wf in enumerate(self.workflows, start=1):
            lines.append(f"{idx}. [{wf['skill']}] {wf['workflow']}")
        return "\n".join(lines)

    def estimate_memory_tokens(self) -> int:
        return estimate_tokens(self.render_workflows_block())

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.workflows, indent=2, ensure_ascii=False), encoding="utf-8")
