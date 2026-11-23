"""Native bridge, agents, and Deepnote-faithful judge for Deepnote's 8 official benchmark tasks.

Loads the real open-source task packages (`tasks/index.json` + `tasks/task-01..08/package/task.json`
+ `config.json` + `delta/<service>/seed.json`) from `deepnote/agent-initiative-benchmark`, executes
both the official Direct baseline (`B0_official_direct`) and `DIAL-Skill` (`DIAL_Skill_official`)
against Deepnote's genuine 11-service runtime (`SlackWorker`), and grades delivered Slack messages
(or silence) with Deepnote's strict 0-10 rubric including `important_mistakes` zero-conditions.

All `benchmark.runtime.*` imports are lazy so catalog parsing, rubric evaluation, and unit tests
run cleanly offline on Windows as well as live on Linux/Colab Python 3.13.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from dial_skill.config import DialSkillConfig, load_dial_skill_config
from dial_skill.core.executor import ReconResult, _canonical_call_key, _safe_parse_tool_args, _truncate_output
from dial_skill.core.gate import DialecticGate
from dial_skill.core.skill_store import SkillStore
from dial_skill.core.triage import TriageStage, rebuild_slate_with_recon
from dial_skill.eval.judge import _anonymize_candidate_tag, _coerce_rubric_score
from dial_skill.llm.jev_client import TriageClient
from dial_skill.llm.openai_client import OpenAIClient
from dial_skill.prompts.templates import build_execution_system_prompt
from dial_skill.schemas import (
    ChatMessage,
    GateDecision,
    InterventionTier,
    ReconSnippet,
    SkillName,
)

logger = logging.getLogger(__name__)

SLACK_WRITE_TOOLS: frozenset[str] = frozenset(
    {"slack_post_message", "slack_reply_to_thread", "slack_send_dm"}
)

CATEGORY_TO_SKILL: dict[str, SkillName] = {
    "Find": "find",
    "Explain": "explain",
    "Analyze": "analyze",
    "Verify": "check",
    "Diagnose": "diagnose",
    "Decide": "decide",
    "Coordinate": "coordinate",
    "Guard": "guard",
}

FINISH_TASK_OPENAI_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "finish_task",
        "description": (
            "Signal that you have finished evaluating the recent Slack snapshot and have delivered "
            "all needed Slack messages (or intentionally chosen silence). Call with no arguments on its own."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Data schemas for Deepnote Official-8
# ---------------------------------------------------------------------------


class DeepnoteOfficialTask(BaseModel):
    """Exact parsed representation of one official Deepnote benchmark task (`task-01`..`task-08`)."""

    task_id: str
    title: str
    category: str
    mapped_skill: SkillName
    version: int = 1
    intent: Literal["action", "silence"]
    expected_action: Literal["INTERVENE", "SILENT"]
    situation: str
    ideal_action: str | None = None
    notes: str = ""
    snapshot_time: str = ""
    build_id: str = ""
    tool_scope: list[str] = Field(default_factory=list)
    delta_services: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    example_messages: list[dict[str, Any]] = Field(default_factory=list)
    important_mistakes: list[str] = Field(default_factory=list)
    reference_reason: str | None = None
    limits: dict[str, Any] = Field(default_factory=dict)
    package_dir: str
    config_path: str


class DeliveredSlackMessage(BaseModel):
    """A genuine Slack write committed via Deepnote's SlackEnvironment overlay."""

    tool: str
    channel_id: str = ""
    channel_name: str = ""
    ts: str = ""
    thread_ts: str | None = None
    destination_user_id: str | None = None
    text: str = ""


class OfficialEpisodeResult(BaseModel):
    """Complete execution trace & outcome of an agent on one Deepnote Official-8 task."""

    task_id: str
    title: str
    category: str
    expected_intent: Literal["action", "silence"]
    method: str
    model_id: str
    predicted_action: Literal["INTERVENE", "SILENT"]
    delivered_messages: list[DeliveredSlackMessage] = Field(default_factory=list)
    stop_reason: str = "finish_task"
    tool_calls_used: int = 0
    read_calls_used: int = 0
    write_calls_used: int = 0
    model_elapsed_sec: float = 0.0
    timed_out: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    gate: GateDecision | None = None
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    initial_context_summary: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeepnoteRubricJudgment(BaseModel):
    """Deepnote-faithful 0-10 integer rubric evaluation (independent overall + mistake flags)."""

    task_id: str
    method: str
    model_id: str
    judge_model: str
    status: Literal["scored", "technical_error"] = "scored"
    intervention_or_silence: float
    correctness_and_evidence: float | None = None
    manner_of_intervention: float | None = None
    audience_and_location: float | None = None
    overall_score: float
    triggered_important_mistakes: list[str] = Field(default_factory=list)
    serious_mistake_zero: bool = False
    justifications: dict[str, str] = Field(default_factory=dict)
    pass_scores: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Offline-safe task catalog parser
# ---------------------------------------------------------------------------


def load_deepnote_official_catalog(
    repo_dir: str | Path,
    selected_ids: list[str] | None = None,
) -> list[DeepnoteOfficialTask]:
    """Load Deepnote's real 8 official tasks from `tasks/index.json` + task packages.

    Never synthesizes fake fallback tasks — raises FileNotFoundError if `tasks/index.json` is absent.
    """
    root = Path(repo_dir).resolve()
    tasks_root = root / "tasks"
    index_path = tasks_root / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Deepnote official tasks index not found at {index_path}. "
            "Clone https://github.com/deepnote/agent-initiative-benchmark first."
        )

    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    rows = index_data.get("tasks", [])
    if not rows:
        raise ValueError(f"No tasks listed in {index_path}")

    wanted = set(selected_ids) if selected_ids else None
    catalog: list[DeepnoteOfficialTask] = []

    for row in rows:
        task_id = str(row["id"])
        if wanted is not None and task_id not in wanted:
            continue

        package_dir = (tasks_root / row["package"]).resolve()
        config_path = (tasks_root / row["config"]).resolve()
        task_json_path = package_dir / "task.json"
        if not task_json_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                f"Incomplete task package for {task_id}: expected {task_json_path} and {config_path}"
            )

        task_doc = json.loads(task_json_path.read_text(encoding="utf-8"))
        cfg_doc = json.loads(config_path.read_text(encoding="utf-8"))

        brief = task_doc.get("brief", {})
        ref = task_doc.get("reference", {})
        meta = task_doc.get("meta", {})
        intent: Literal["action", "silence"] = (
            "silence" if str(brief.get("intent", "action")).strip().lower() == "silence" else "action"
        )
        category = str(row.get("category", "Find"))
        mapped_skill = CATEGORY_TO_SKILL.get(category, "find")

        delta_services = sorted(
            {
                str(ch["service"])
                for ch in task_doc.get("changes", [])
                if isinstance(ch, dict) and ch.get("service")
            }
        )

        catalog.append(
            DeepnoteOfficialTask(
                task_id=task_id,
                title=str(row.get("title", task_id)),
                category=category,
                mapped_skill=mapped_skill,
                version=int(row.get("version", 1)),
                intent=intent,
                expected_action="SILENT" if intent == "silence" else "INTERVENE",
                situation=str(brief.get("situation", "")),
                ideal_action=brief.get("ideal_action"),
                notes=str(task_doc.get("notes", "")),
                snapshot_time=str(cfg_doc.get("snapshot_time", meta.get("snapshot_time", ""))),
                build_id=str((cfg_doc.get("build_ref") or {}).get("build_id", "")),
                tool_scope=[str(t) for t in cfg_doc.get("tool_scope", [])],
                delta_services=delta_services,
                evidence=list(task_doc.get("evidence", [])),
                example_messages=list(ref.get("example_messages", [])),
                important_mistakes=[str(m) for m in ref.get("important_mistakes", [])],
                reference_reason=ref.get("reason"),
                limits=dict(cfg_doc.get("limits", {})),
                package_dir=str(package_dir),
                config_path=str(config_path),
            )
        )

    if wanted is not None:
        found = {t.task_id for t in catalog}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"Requested Deepnote task IDs not found: {missing}")

    return catalog


def format_initial_context_messages(initial_context: dict[str, Any]) -> list[ChatMessage]:
    """Convert Deepnote's real multi-channel `initial_context` into `ChatMessage` objects.

    Preserves exact `channel_id`, `channel.name`, `author.real_name`, `author.title`, `user_id`,
    `ts`, and `thread_ts` so both Stage-1 Triage / Stage-2 Gate and Stage-3 Slack write tools have
    complete, unambiguous addressing information.
    """
    snapshot_time = str(initial_context.get("snapshot_time", ""))
    raw_msgs = initial_context.get("messages", [])
    converted: list[ChatMessage] = []

    for idx, m in enumerate(raw_msgs, start=1):
        ch = m.get("channel") or {}
        ch_name = str(ch.get("name") or m.get("channel_id") or "slack")
        ch_id = str(m.get("channel_id") or "")
        author = m.get("author") or {}
        real_name = str(author.get("real_name") or author.get("display_name") or m.get("user_id") or "Unknown")
        title = str(author.get("title") or "").strip()
        user_id = str(m.get("user_id") or "")
        ts = str(m.get("ts") or "")
        thread_ts = m.get("thread_ts")
        text = str(m.get("text") or "")

        role_tag = f", {title}" if title else ""
        thread_tag = f" thread_ts={thread_ts}" if thread_ts else " (new message)"
        enriched_text = (
            f"[#{ch_name} ({ch_id}) | ts={ts}{thread_tag} | user_id={user_id}{role_tag}] {text}"
        )

        converted.append(
            ChatMessage(
                msg_id=f"m{idx:02d}",
                speaker=real_name,
                text=enriched_text,
                timestamp=ts or snapshot_time,
                thread_ts=str(thread_ts) if thread_ts else ts,
                is_agent=False,
            )
        )

    return converted


def convert_deepnote_tool_to_openai(tool_def: dict[str, Any]) -> dict[str, Any]:
    """Convert one Deepnote native tool schema into OpenAI Chat Completions function tool format."""
    return {
        "type": "function",
        "function": {
            "name": str(tool_def["name"]),
            "description": str(tool_def.get("description", "")),
            "parameters": dict(tool_def.get("parameters", {"type": "object", "properties": {}})),
        },
    }


def _extract_delivered_message(
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    channel_names: dict[str, str],
) -> DeliveredSlackMessage:
    """Build a canonical DeliveredSlackMessage from a successful Slack write tool call."""
    msg = result.get("message") if isinstance(result.get("message"), dict) else result
    ch_id = str(msg.get("channel_id") or arguments.get("channel_id") or "")
    dest = result.get("destination") if isinstance(result.get("destination"), dict) else {}
    dest_user = dest.get("user_id") or arguments.get("user_id")
    thread_ts = msg.get("thread_ts") or arguments.get("thread_ts")
    text = str(msg.get("text") or arguments.get("text") or "")
    return DeliveredSlackMessage(
        tool=tool_name,
        channel_id=ch_id,
        channel_name=channel_names.get(ch_id, ch_id),
        ts=str(msg.get("ts") or ""),
        thread_ts=str(thread_ts) if thread_ts else None,
        destination_user_id=str(dest_user) if dest_user else None,
        text=text,
    )


# ---------------------------------------------------------------------------
# Live Deepnote Runtime Session (lazy-imports `benchmark.runtime.*` on Linux/Colab)
# ---------------------------------------------------------------------------


class DeepnoteOfficialTaskSession:
    """Async context manager wrapping Deepnote's genuine `SlackWorker` for a single attempt.

    Provides exact `initial_context`, exact `rendered_official_instruction` (with `{service_guide}`),
    OpenAI-compatible tool schemas (read-only or full), tool execution with delivery interception,
    and clean subprocess teardown.
    """

    def __init__(
        self,
        *,
        task: DeepnoteOfficialTask,
        data_dir: str | Path,
        cache_dir: str | Path,
        work_root: str | Path | None = None,
    ) -> None:
        self.task = task
        self.data_dir = Path(data_dir).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.work_root = Path(work_root).resolve() if work_root is not None else None

        self._work_dir: Path | None = None
        self._worker: Any = None
        self._services: Any = None

        self.config: dict[str, Any] = {}
        self.limits: dict[str, Any] = {}
        self.initial_context: dict[str, Any] = {}
        self.raw_tools: list[dict[str, Any]] = []
        self.rendered_official_instruction: str = ""
        self.service_guide: str = ""
        self.channel_names: dict[str, str] = {}

        self.delivered_messages: list[DeliveredSlackMessage] = []
        self.tool_trace: list[dict[str, Any]] = []
        self.read_calls_used: int = 0
        self.write_calls_used: int = 0
        self.setup_seconds: float = 0.0

    async def __aenter__(self) -> DeepnoteOfficialTaskSession:
        await self.open()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    async def open(self) -> None:
        # Lazy imports so Windows offline unit tests never import Linux-only `fcntl` modules.
        from benchmark.runtime import agent_protocol, agent_runner, service_registry as services
        from benchmark.runtime.attempt_worker import SlackWorker
        from benchmark.runtime.contracts import load_config
        from benchmark.runtime.packages import verify_package
        from benchmark.runtime.slack_environment import SlackEnvironment

        self._services = services
        config_path = Path(self.task.config_path).resolve()
        package_dir = Path(self.task.package_dir).resolve()
        verify_package(package_dir, config_path=config_path)

        self.config = load_config(config_path)
        self.limits = agent_runner.resolve_limits(self.config)
        settings = agent_runner.baseline_settings(
            self.config,
            "verified_local",
            str(self.data_dir),
            None,
            None,
        )
        assert settings is not None
        settings["cache_dir"] = str(self.cache_dir)

        base_tmp = self.work_root if self.work_root is not None else Path(tempfile.gettempdir())
        base_tmp.mkdir(parents=True, exist_ok=True)
        self._work_dir = base_tmp / f"dn_{self.task.task_id}_{uuid.uuid4().hex[:8]}"
        self._work_dir.mkdir(parents=True, exist_ok=False)

        self.rendered_official_instruction = agent_protocol.render_instruction(self.config, self.limits)
        self.service_guide = services.discovery_guide(self.config["tool_scope"])

        t0 = time.monotonic()
        self._worker = SlackWorker(
            {
                "package": str(package_dir),
                "config_path": str(config_path),
                "baseline_dir": str(self.data_dir),
                "baseline_build_ref": self.config["build_ref"],
                "baseline_settings": settings,
                "linear_baseline_dir": None,
                "notion_baseline_dir": None,
                "service_baseline_dirs": {},
                "work_dir": str(self._work_dir / "environment"),
                "initial_context_max_bytes": self.limits["initial_context_max_bytes"],
                "recent_activity_window_minutes": self.limits["recent_activity_window_minutes"],
                "max_result_bytes": self.limits["max_result_bytes"],
            },
            SlackEnvironment,
            None,
        )
        setup_timeout = float(self.limits.get("environment_setup_timeout_seconds", 600))
        ready = await self._worker.receive_async(setup_timeout)
        self.setup_seconds = round(time.monotonic() - t0, 3)

        self.initial_context = ready["context"]
        self.raw_tools = [services.agent_definition(t) for t in ready["tools"]]

        for m in self.initial_context.get("messages", []):
            ch_id = str(m.get("channel_id") or "")
            ch_name = str((m.get("channel") or {}).get("name") or "")
            if ch_id and ch_name:
                self.channel_names[ch_id] = ch_name

    def get_openai_tool_schemas(
        self,
        *,
        read_only: bool = False,
        include_finish_task: bool = False,
    ) -> list[dict[str, Any]]:
        schemas: list[dict[str, Any]] = []
        for tool in self.raw_tools:
            name = str(tool["name"])
            if read_only and name in SLACK_WRITE_TOOLS:
                continue
            schemas.append(convert_deepnote_tool_to_openai(tool))
        if include_finish_task:
            schemas.append(FINISH_TASK_OPENAI_TOOL)
        return schemas

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        read_only: bool = False,
        max_result_chars: int = 12_000,
    ) -> tuple[dict[str, Any], str]:
        """Execute a native Deepnote tool inside the isolated worker and record trace/deliveries."""
        if read_only and name in SLACK_WRITE_TOOLS:
            err = {
                "ok": False,
                "error": "forbidden_write",
                "message": "Write tools are disabled during read-only Silent Recon.",
            }
            return err, json.dumps(err)

        exec_args = self._services.agent_arguments(name, arguments)
        tool_timeout = float(self.limits.get("tool_timeout_seconds", 600))
        t0 = time.monotonic()
        try:
            raw_result = await self._worker.call_async(name, exec_args, tool_timeout)
        except Exception as exc:
            raw_result = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
        elapsed = round(time.monotonic() - t0, 3)

        is_write = name in SLACK_WRITE_TOOLS
        if is_write:
            self.write_calls_used += 1
            if isinstance(raw_result, dict) and raw_result.get("ok"):
                delivered = _extract_delivered_message(name, exec_args, raw_result, self.channel_names)
                self.delivered_messages.append(delivered)
        else:
            self.read_calls_used += 1

        truncated = _truncate_output(raw_result, max_result_chars)
        self.tool_trace.append(
            {
                "tool": name,
                "kind": "write" if is_write else "read",
                "arguments": exec_args,
                "ok": bool(isinstance(raw_result, dict) and raw_result.get("ok", False)),
                "elapsed_sec": elapsed,
                "result_preview": truncated[:900],
            }
        )
        return raw_result, truncated

    def close(self) -> None:
        if self._worker is not None:
            try:
                self._worker.close(graceful=True)
            except Exception as exc:
                logger.warning("Error closing SlackWorker for %s: %s", self.task.task_id, exc)
            self._worker = None
        if self._work_dir is not None and self._work_dir.exists():
            shutil.rmtree(self._work_dir, ignore_errors=True)
            self._work_dir = None


# ---------------------------------------------------------------------------
# Agents on Deepnote Official-8: Direct Baseline (B0) & DIAL-Skill
# ---------------------------------------------------------------------------


def _build_assistant_tool_call_message(content: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content or "",
        "tool_calls": [
            {
                "id": tc.get("id") or f"tc_{idx}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": (
                        tc["arguments"]
                        if isinstance(tc["arguments"], str)
                        else json.dumps(tc["arguments"], ensure_ascii=False)
                    ),
                },
            }
            for idx, tc in enumerate(tool_calls)
        ],
    }


def _summarize_initial_context(initial_context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for m in initial_context.get("messages", []):
        rows.append(
            {
                "channel_id": m.get("channel_id"),
                "channel_name": (m.get("channel") or {}).get("name"),
                "user_id": m.get("user_id"),
                "author": (m.get("author") or {}).get("real_name"),
                "title": (m.get("author") or {}).get("title"),
                "ts": m.get("ts"),
                "thread_ts": m.get("thread_ts"),
                "text": m.get("text"),
            }
        )
    return rows


class OfficialDirectAgent:
    """Faithful Deepnote Official Direct Baseline (`B0_official_direct`).

    Uses Deepnote's EXACT `assess_recent_slack.txt` + `{service_guide}` prompt, exact JSON
    `initial_context`, full native tools + `finish_task`, and Deepnote's `180s` model-time budget.
    """

    def __init__(
        self,
        *,
        openai_client: OpenAIClient,
        model: str,
        method_name: str = "B0_official_direct",
        max_turns: int = 24,
        max_result_chars: int = 12_000,
        seed: int = 42,
    ) -> None:
        self.openai_client = openai_client
        self.model = model
        self.method_name = method_name
        self.max_turns = max_turns
        self.max_result_chars = max_result_chars
        self.seed = seed

    async def run_task(self, session: DeepnoteOfficialTaskSession) -> OfficialEpisodeResult:
        model_budget_sec = float(session.limits.get("agent_model_seconds", 180.0))
        max_parallel_reads = int(session.limits.get("max_read_calls_per_response", 8))
        tools = session.get_openai_tool_schemas(read_only=False, include_finish_task=True)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": session.rendered_official_instruction},
            {"role": "user", "content": json.dumps(session.initial_context, ensure_ascii=False)},
        ]

        model_elapsed = 0.0
        total_cost = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        timed_out = False
        stop_reason = "max_turns"
        reminded_once = False

        for _turn in range(self.max_turns):
            if model_elapsed >= model_budget_sec:
                timed_out = True
                stop_reason = "model_budget"
                break

            llm_res = await self.openai_client.complete_text(
                model=self.model,
                messages=messages,
                temperature=0.0,
                seed=self.seed,
                tools=tools,
                tool_choice="auto",
                use_cache=False,
            )
            model_elapsed += llm_res.elapsed_sec
            total_cost += llm_res.cost_usd
            prompt_tokens += llm_res.prompt_tokens
            completion_tokens += llm_res.completion_tokens

            if model_elapsed >= model_budget_sec:
                timed_out = True
                stop_reason = "model_budget"
                break

            if not llm_res.tool_calls:
                messages.append({"role": "assistant", "content": llm_res.content or ""})
                if reminded_once:
                    stop_reason = "no_tool_calls"
                    break
                reminded_once = True
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Reminder: Plain assistant text does not deliver a Slack message. "
                            "Use a Slack messaging tool if you want to intervene, or call finish_task "
                            "with no arguments when done (including if you choose silence)."
                        ),
                    }
                )
                continue

            messages.append(_build_assistant_tool_call_message(llm_res.content or "", llm_res.tool_calls))

            finished = False
            read_in_turn = 0
            for idx, tc in enumerate(llm_res.tool_calls):
                tc_id = tc.get("id") or f"tc_{idx}"
                name = str(tc["name"])
                args = _safe_parse_tool_args(tc.get("arguments", "{}"))

                if name == "finish_task":
                    finished = True
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": json.dumps({"ok": True})})
                    continue

                if name not in SLACK_WRITE_TOOLS:
                    read_in_turn += 1
                    if read_in_turn > max_parallel_reads:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": json.dumps(
                                    {
                                        "ok": False,
                                        "error": "too_many_read_calls",
                                        "message": f"At most {max_parallel_reads} read calls per turn.",
                                    }
                                ),
                            }
                        )
                        continue

                _, obs_str = await session.call_tool(
                    name,
                    args,
                    read_only=False,
                    max_result_chars=self.max_result_chars,
                )
                messages.append({"role": "tool", "tool_call_id": tc_id, "content": obs_str})

            if finished:
                stop_reason = "finish_task"
                break

        predicted: Literal["INTERVENE", "SILENT"] = (
            "INTERVENE" if len(session.delivered_messages) > 0 else "SILENT"
        )
        return OfficialEpisodeResult(
            task_id=session.task.task_id,
            title=session.task.title,
            category=session.task.category,
            expected_intent=session.task.intent,
            method=self.method_name,
            model_id=self.model,
            predicted_action=predicted,
            delivered_messages=list(session.delivered_messages),
            stop_reason=stop_reason,
            tool_calls_used=session.read_calls_used + session.write_calls_used,
            read_calls_used=session.read_calls_used,
            write_calls_used=session.write_calls_used,
            model_elapsed_sec=round(model_elapsed, 4),
            timed_out=timed_out,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_cost_usd=round(total_cost, 8),
            gate=None,
            tool_trace=list(session.tool_trace),
            initial_context_summary=_summarize_initial_context(session.initial_context),
            metadata={"setup_seconds": session.setup_seconds},
        )


class OfficialDialSkillAgent:
    """Authentic 4-Stage DIAL-Skill running live on Deepnote's 8 real official tasks.

    - **Stage 1**: Jev System-1 Decision-Change Triage over the full multi-channel Slack snapshot.
    - **Stage 1.5**: Bounded Read-Only Silent Recon using Deepnote's real 11-service read tools
      (critical for cross-service verification such as discovering `task-03` is already owned in Linear).
    - **Stage 2**: Parallel Symmetric-Evidence Advocate vs Skeptic Dialectic Gate (`slate_hash` lock)
      with Dual-Boundary Skill Routing & 4-Tier Channel De-escalation.
    - **Stage 3**: Applicability-routed budgeted execution using Deepnote's real read tools + Slack write
      tools (`slack_reply_to_thread`, `slack_post_message`, `slack_send_dm`), deduplicating Stage-1.5
      reads and supporting Stage-3 abort-to-silent.
    """

    def __init__(
        self,
        *,
        triage_client: TriageClient,
        openai_client: OpenAIClient,
        gate_model: str,
        executor_model: str,
        skill_store: SkillStore | None = None,
        cfg: DialSkillConfig | None = None,
        method_name: str = "DIAL_Skill_official",
        recon_max_calls: int = 8,
        recon_max_turns: int = 5,
        recon_soft_cap_sec: float = 55.0,
        stage3_max_turns: int = 18,
        max_result_chars: int = 12_000,
        seed: int = 42,
    ) -> None:
        self.cfg = cfg or load_dial_skill_config()
        self.skill_store = skill_store or SkillStore.from_json_file()
        self.triage_client = triage_client
        self.openai_client = openai_client
        self.gate_model = gate_model
        self.executor_model = executor_model
        self.method_name = method_name
        self.recon_max_calls = recon_max_calls
        self.recon_max_turns = recon_max_turns
        self.recon_soft_cap_sec = recon_soft_cap_sec
        self.stage3_max_turns = stage3_max_turns
        self.max_result_chars = max_result_chars
        self.seed = seed

        # Ensure multi-channel snapshot (typically 5-12 messages across 2-4 channels) is kept intact in SSES
        triage_cfg = self.cfg.model_copy(deep=True)
        triage_cfg.triage.k_slate = max(triage_cfg.triage.k_slate, 16)
        self.triage_stage = TriageStage(triage_client=triage_client, cfg=triage_cfg)
        self.gate_stage = DialecticGate(
            openai_client=openai_client,
            gate_model=gate_model,
            cfg=self.cfg,
            seed=seed,
        )

    async def _run_official_silent_recon(
        self,
        *,
        session: DeepnoteOfficialTaskSession,
        slate_text: str,
        candidate_skill_plans: str,
    ) -> tuple[ReconResult, int, int]:
        """Stage 1.5 read-only cross-service recon against Deepnote's real 11-service runtime."""
        recon = ReconResult()
        prompt_tokens = 0
        completion_tokens = 0

        read_tools = session.get_openai_tool_schemas(read_only=True, include_finish_task=False)
        if not read_tools or self.recon_max_calls <= 0:
            return recon, 0, 0

        system_prompt = (
            "You are running Stage-1.5 SILENT RECON (strictly read-only) for a proactive workplace AI on Slack.\n"
            f"You have at most {self.recon_max_calls} read-only tool calls across up to {self.recon_max_turns} turns.\n"
            "Your goal BEFORE deciding whether to intervene is to verify across company systems:\n"
            "1. Which recent Slack thread(s) contain a genuinely UNRESOLVED need vs already-handled / acknowledged distractors.\n"
            "2. Whether an apparent problem is ALREADY owned, resolved, or explicitly decided in Linear, Notion, older Slack threads, or analytics.\n"
            "3. Concrete factual evidence (metrics, dates, policies, owners, issue status, counterevidence) needed to judge whether silence or intervention is warranted.\n"
            "Rules:\n"
            "- Do NOT send Slack messages (write tools are disabled in Stage 1.5).\n"
            "- Group independent read calls in parallel when helpful.\n"
            "- Stop calling tools as soon as you have clear evidence for/against intervening.\n\n"
            f"Candidate Skill Investigation Plans:\n{candidate_skill_plans}\n\n"
            f"Service Discovery Guide:\n{session.service_guide}"
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Shared Evidence Slate:\n{slate_text}\n\n"
                    f"Exact Snapshot Metadata & Messages:\n{json.dumps(session.initial_context, ensure_ascii=False)}"
                ),
            },
        ]

        for _turn in range(self.recon_max_turns):
            if recon.tool_calls_used >= self.recon_max_calls or recon.model_elapsed_sec >= self.recon_soft_cap_sec:
                break

            llm_res = await self.openai_client.complete_text(
                model=self.executor_model,
                messages=messages,
                temperature=0.0,
                seed=self.seed,
                tools=read_tools,
                tool_choice="auto",
                use_cache=False,
            )
            recon.model_elapsed_sec += llm_res.elapsed_sec
            recon.cost_usd += llm_res.cost_usd
            prompt_tokens += llm_res.prompt_tokens
            completion_tokens += llm_res.completion_tokens

            if not llm_res.tool_calls:
                break

            messages.append(_build_assistant_tool_call_message(llm_res.content or "", llm_res.tool_calls))

            for idx, tc in enumerate(llm_res.tool_calls):
                tc_id = tc.get("id") or f"recon_tc_{idx}"
                name = str(tc["name"])
                args = _safe_parse_tool_args(tc.get("arguments", "{}"))
                call_key = _canonical_call_key(name, args)

                if recon.tool_calls_used >= self.recon_max_calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": json.dumps({"ok": False, "error": "recon_budget_reached"}),
                        }
                    )
                    continue

                if call_key in recon.executed_calls:
                    obs_str = recon.executed_calls[call_key]
                else:
                    _, obs_str = await session.call_tool(
                        name,
                        args,
                        read_only=True,
                        max_result_chars=self.max_result_chars,
                    )
                    recon.executed_calls[call_key] = obs_str
                    recon.tool_calls_used += 1
                    service = name.split("_")[0] if "_" in name else name
                    recon.snippets.append(
                        ReconSnippet(
                            recon_id=f"recon_{recon.tool_calls_used}",
                            service=service,
                            query=f"{name}({json.dumps(args, sort_keys=True, ensure_ascii=False)})",
                            summary_fact=obs_str[:850],
                        )
                    )

                messages.append({"role": "tool", "tool_call_id": tc_id, "content": obs_str})

        return recon, prompt_tokens, completion_tokens

    async def run_task(self, session: DeepnoteOfficialTaskSession) -> OfficialEpisodeResult:
        model_budget_sec = float(session.limits.get("agent_model_seconds", 180.0))
        max_parallel_reads = int(session.limits.get("max_read_calls_per_response", 8))

        chat_messages = format_initial_context_messages(session.initial_context)
        triage_cost_before = self.triage_client.total_cost_usd

        # Stage 1: System-1 Decision-Change Triage & Balanced SSES
        triage_out = await self.triage_stage.run(
            conversation_id=session.task.task_id,
            channel_name="multi_channel_slack_snapshot",
            env_mode="pagemantle",
            messages=chat_messages,
            skill_store=self.skill_store,
        )
        triage_cost = max(0.0, self.triage_client.total_cost_usd - triage_cost_before)

        if triage_out.short_circuited:
            gate = GateDecision(
                short_circuited=True,
                slate_hash=triage_out.slate.slate_hash,
                selected_skill="none",
                advocate_score=0.0,
                skeptic_score=1.0,
                margin_delta=-1.0,
                disagreement_u=0.0,
                raw_tier=InterventionTier.SILENT,
                final_tier=InterventionTier.SILENT,
                binary_speak=False,
                deescalation_reasons=["stage1_fast_silence_short_circuit"],
            )
            return OfficialEpisodeResult(
                task_id=session.task.task_id,
                title=session.task.title,
                category=session.task.category,
                expected_intent=session.task.intent,
                method=self.method_name,
                model_id=f"{self.gate_model}+{self.executor_model}",
                predicted_action="SILENT",
                delivered_messages=[],
                stop_reason="stage1_short_circuit",
                tool_calls_used=0,
                read_calls_used=0,
                write_calls_used=0,
                model_elapsed_sec=0.0,
                timed_out=False,
                total_cost_usd=round(triage_cost, 8),
                gate=gate,
                tool_trace=[],
                initial_context_summary=_summarize_initial_context(session.initial_context),
                metadata={
                    "triage_backend_used": triage_out.triage_backend_used,
                    "recon_calls": 0,
                    "setup_seconds": session.setup_seconds,
                },
            )

        slate = triage_out.slate
        plan_blocks = []
        for card in slate.candidate_skills:
            steps = "\n".join(f"  - {s}" for s in card.investigation_plan)
            plan_blocks.append(f"Skill `{card.name}`:\n{steps}")
        candidate_plans_text = "\n".join(plan_blocks) or "- Inspect relevant Slack threads and linked enterprise records."

        # Stage 1.5: Bounded Read-Only Silent Recon
        recon, recon_pt, recon_ct = await self._run_official_silent_recon(
            session=session,
            slate_text=slate.canonical_text,
            candidate_skill_plans=candidate_plans_text,
        )
        if recon.snippets:
            slate = rebuild_slate_with_recon(slate, recon.snippets)

        # Stage 2: Parallel Symmetric-Evidence Dialectic Gating
        gate_decision, gate_cost, gate_elapsed = await self.gate_stage.evaluate(
            advocate_slate=slate,
            skeptic_slate=slate,
            skill_store=self.skill_store,
        )

        model_elapsed = recon.model_elapsed_sec + gate_elapsed
        total_cost = triage_cost + recon.cost_usd + gate_cost
        prompt_tokens = recon_pt
        completion_tokens = recon_ct

        if gate_decision.final_tier == InterventionTier.SILENT:
            return OfficialEpisodeResult(
                task_id=session.task.task_id,
                title=session.task.title,
                category=session.task.category,
                expected_intent=session.task.intent,
                method=self.method_name,
                model_id=f"{self.gate_model}+{self.executor_model}",
                predicted_action="SILENT",
                delivered_messages=[],
                stop_reason="stage2_gate_silent",
                tool_calls_used=session.read_calls_used + session.write_calls_used,
                read_calls_used=session.read_calls_used,
                write_calls_used=session.write_calls_used,
                model_elapsed_sec=round(model_elapsed, 4),
                timed_out=False,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_cost_usd=round(total_cost, 8),
                gate=gate_decision,
                tool_trace=list(session.tool_trace),
                initial_context_summary=_summarize_initial_context(session.initial_context),
                metadata={
                    "triage_backend_used": triage_out.triage_backend_used,
                    "recon_calls": recon.tool_calls_used,
                    "setup_seconds": session.setup_seconds,
                },
            )

        # Stage 3: Applicability-Routed Skill Execution with Deepnote Native Read + Slack Write Tools
        skill_card = self.skill_store.get(gate_decision.selected_skill)
        skill_contract_prompt = build_execution_system_prompt(skill_card, gate_decision.final_tier)
        stage3_system = (
            f"{session.rendered_official_instruction}\n\n"
            "=== DIAL-SKILL ROUTING & BOUNDARY CONTRACT ===\n"
            f"{skill_contract_prompt}\n\n"
            "IMPORTANT DELIVERY & ABORT RULES FOR THIS REAL SLACK ENVIRONMENT:\n"
            "1. Only Slack messaging tools (`slack_reply_to_thread`, `slack_post_message`, `slack_send_dm`) deliver messages! "
            "Plain assistant text is NOT delivered.\n"
            "2. Reuse the Stage-1.5 Silent Recon findings below rather than repeating identical read queries, "
            "and run any additional read tool checks needed so every number, date, policy rule, owner, and caveat is verified.\n"
            "3. Respect the Stage-2 channel/audience recommendation AND disclosure boundaries: keep sensitive/restricted PII "
            "out of public or unauthorized channels; when a request mixes a legitimate aggregate question with an unsafe "
            "restricted data request, answer the safe aggregate part in-thread and clearly state the policy limit and authorized next step.\n"
            "4. STAGE-3 ABORT-TO-SILENT: If deeper read verification reveals that every recent thread is ALREADY handled, owned, "
            "or resolved with no remaining unmet need, send ZERO Slack messages and simply call `finish_task`.\n"
            "5. When finished (whether after delivering Slack message(s) or choosing silence), call `finish_task` with no arguments on its own."
        )

        recon_summary_lines = [
            f"[{s.recon_id} | {s.service}] {s.query}\n  -> {s.summary_fact}"
            for s in recon.snippets
        ]
        recon_summary_block = "\n".join(recon_summary_lines) or "(none)"

        stage3_user = (
            f"Exact Initial Slack Snapshot:\n{json.dumps(session.initial_context, ensure_ascii=False)}\n\n"
            f"Stage-1.5 Silent Recon Findings Already Retrieved:\n{recon_summary_block}\n\n"
            f"Stage-2 Dialectic Gate Assessment:\n"
            f"- Selected Skill: {gate_decision.selected_skill}\n"
            f"- Recommended Tier: {gate_decision.final_tier.value} "
            f"(margin={gate_decision.margin_delta:.3f}, disagreement={gate_decision.disagreement_u:.3f})\n"
            f"- Advocate Need Assessment: {gate_decision.advocate.unmitigated_loss if gate_decision.advocate else 'N/A'}\n"
            f"- Skeptic Caution / Vetos: {gate_decision.skeptic.disruption_risk if gate_decision.skeptic else 'N/A'}\n"
        )

        tools = session.get_openai_tool_schemas(read_only=False, include_finish_task=True)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": stage3_system},
            {"role": "user", "content": stage3_user},
        ]

        executed_cache = dict(recon.executed_calls)
        timed_out = False
        stop_reason = "max_turns"
        reminded_once = False

        for _turn in range(self.stage3_max_turns):
            if model_elapsed >= model_budget_sec:
                timed_out = True
                stop_reason = "model_budget"
                break

            llm_res = await self.openai_client.complete_text(
                model=self.executor_model,
                messages=messages,
                temperature=0.0,
                seed=self.seed,
                tools=tools,
                tool_choice="auto",
                use_cache=False,
            )
            model_elapsed += llm_res.elapsed_sec
            total_cost += llm_res.cost_usd
            prompt_tokens += llm_res.prompt_tokens
            completion_tokens += llm_res.completion_tokens

            if model_elapsed >= model_budget_sec:
                timed_out = True
                stop_reason = "model_budget"
                break

            if not llm_res.tool_calls:
                messages.append({"role": "assistant", "content": llm_res.content or ""})
                if reminded_once:
                    stop_reason = "no_tool_calls"
                    break
                reminded_once = True
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Reminder: Plain assistant text does NOT deliver a Slack message. "
                            "Call `slack_reply_to_thread` / `slack_post_message` / `slack_send_dm` to deliver "
                            "your verified response, or call `finish_task` with no arguments if you are done or aborting to silence."
                        ),
                    }
                )
                continue

            messages.append(_build_assistant_tool_call_message(llm_res.content or "", llm_res.tool_calls))

            finished = False
            read_in_turn = 0
            for idx, tc in enumerate(llm_res.tool_calls):
                tc_id = tc.get("id") or f"s3_tc_{idx}"
                name = str(tc["name"])
                args = _safe_parse_tool_args(tc.get("arguments", "{}"))

                if name == "finish_task":
                    finished = True
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": json.dumps({"ok": True})})
                    continue

                if name not in SLACK_WRITE_TOOLS:
                    read_in_turn += 1
                    if read_in_turn > max_parallel_reads:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": json.dumps(
                                    {
                                        "ok": False,
                                        "error": "too_many_read_calls",
                                        "message": f"At most {max_parallel_reads} read calls per turn.",
                                    }
                                ),
                            }
                        )
                        continue
                    call_key = _canonical_call_key(name, args)
                    if call_key in executed_cache:
                        messages.append({"role": "tool", "tool_call_id": tc_id, "content": executed_cache[call_key]})
                        continue
                    _, obs_str = await session.call_tool(
                        name,
                        args,
                        read_only=False,
                        max_result_chars=self.max_result_chars,
                    )
                    executed_cache[call_key] = obs_str
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": obs_str})
                else:
                    _, obs_str = await session.call_tool(
                        name,
                        args,
                        read_only=False,
                        max_result_chars=self.max_result_chars,
                    )
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": obs_str})

            if finished:
                stop_reason = "finish_task" if session.delivered_messages else "stage3_abort_to_silent"
                break

        predicted: Literal["INTERVENE", "SILENT"] = (
            "INTERVENE" if len(session.delivered_messages) > 0 else "SILENT"
        )
        final_gate = gate_decision
        if predicted == "SILENT" and gate_decision.binary_speak:
            final_gate = gate_decision.model_copy(
                update={
                    "final_tier": InterventionTier.SILENT,
                    "binary_speak": False,
                    "deescalation_reasons": [
                        *gate_decision.deescalation_reasons,
                        f"stage3_zero_writes:{stop_reason}",
                    ],
                }
            )

        return OfficialEpisodeResult(
            task_id=session.task.task_id,
            title=session.task.title,
            category=session.task.category,
            expected_intent=session.task.intent,
            method=self.method_name,
            model_id=f"{self.gate_model}+{self.executor_model}",
            predicted_action=predicted,
            delivered_messages=list(session.delivered_messages),
            stop_reason=stop_reason,
            tool_calls_used=session.read_calls_used + session.write_calls_used,
            read_calls_used=session.read_calls_used,
            write_calls_used=session.write_calls_used,
            model_elapsed_sec=round(model_elapsed, 4),
            timed_out=timed_out,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_cost_usd=round(total_cost, 8),
            gate=final_gate,
            tool_trace=list(session.tool_trace),
            initial_context_summary=_summarize_initial_context(session.initial_context),
            metadata={
                "triage_backend_used": triage_out.triage_backend_used,
                "recon_calls": recon.tool_calls_used,
                "setup_seconds": session.setup_seconds,
            },
        )


# ---------------------------------------------------------------------------
# Strict Deepnote-Faithful Blinded Dual-Judge
# ---------------------------------------------------------------------------

DEEPNOTE_STRICT_JUDGE_SYSTEM_PROMPT = """You are the strict benchmark judge for Deepnote's Agent Initiative Benchmark (PageMantle).
Evaluate ONE blinded terminated agent attempt against its initial Slack snapshot, authoritative ground-truth notes, verified business evidence, example reference messages, and explicit `important_mistakes` checklist.

Evaluate ONLY the agent's ACTUAL DELIVERED SLACK MESSAGES (or SILENCE if zero Slack messages were delivered). Read calls are background evidence showing what the agent actually checked — they are NOT interventions.

STRICT DEEPNOTE RUBRIC RULES (from evaluate_agent_run.txt):
1. Use INTEGER scores from 0 to 10. Choose the highest anchor supported by the observed response and evidence.
2. APPLY EXPLICIT ZERO CONDITIONS FIRST:
   - If `intent == "silence"` and the agent stayed SILENT (0 delivered messages): `intervention_or_silence = 10`, `overall_score = 10`, and set `correctness_and_evidence = null`, `manner_of_intervention = null`, `audience_and_location = null`.
   - If `intent == "silence"` and the agent DELIVERED ANY unnecessary Slack message: `intervention_or_silence = 0`, `overall_score = 0`!
   - If `intent == "action"` and the agent stayed SILENT (0 delivered messages): `intervention_or_silence = 0`, `overall_score = 0`, and set the other three categories to `null`.
   - If the agent commits a SERIOUS MISTAKE from `important_mistakes` (e.g., false central conclusion, disclosing restricted PII/names/emails/user trails to an unauthorized audience, refusing an entire legitimate request instead of answering the safe aggregate part, claiming an unverified fact as certain, or missing the central issue completely): set `serious_mistake_zero = true` and `overall_score = 0` regardless of other strengths!
3. INDEPENDENT OVERALL SCORE:
   - Do NOT average category scores! Judge the complete intervention holistically against the 0-10 overall anchors:
     * 10: Complete intervention is suitable with no meaningful error or omission (or silence is appropriate). All applicable categories == 10 <=> overall == 10.
     * 8-9: Addresses all important needs appropriately; only a small/secondary flaw remains.
     * 6-7: Addresses the main need, with one meaningful bounded flaw/omission needing local or substantial correction.
     * 4-5: Useful elements present, but major gap or several significant bounded omissions leave the needed help incomplete.
     * 1-3: Only trace/partial help; most needed help is missing or needs major correction.
     * 0: Misses warranted need, intervenes when unwarranted, or commits a serious mistake rendering the intervention unacceptable.
4. FAIRNESS TO ALTERNATIVES:
   - The reference example messages illustrate one good answer, NOT the only valid wording or destination. Do not penalize cosmetic wording differences or supported alternative destinations if they meet the need and disclosure boundaries.
   - Reaching the model budget adds no separate penalty; judge only what was actually delivered.

Return STRICT JSON ONLY matching the provided schema."""

DEEPNOTE_STRICT_JUDGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "intervention_or_silence",
        "correctness_and_evidence",
        "manner_of_intervention",
        "audience_and_location",
        "overall_score",
        "triggered_important_mistakes",
        "serious_mistake_zero",
        "justifications",
    ],
    "properties": {
        "intervention_or_silence": {"type": "integer", "minimum": 0, "maximum": 10},
        "correctness_and_evidence": {"type": "integer", "minimum": 0, "maximum": 10},
        "manner_of_intervention": {"type": "integer", "minimum": 0, "maximum": 10},
        "audience_and_location": {"type": "integer", "minimum": 0, "maximum": 10},
        "overall_score": {"type": "integer", "minimum": 0, "maximum": 10},
        "triggered_important_mistakes": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 10,
        },
        "serious_mistake_zero": {"type": "boolean"},
        "justifications": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "intervention_or_silence",
                "correctness_and_evidence",
                "manner_of_intervention",
                "audience_and_location",
                "overall",
            ],
            "properties": {
                "intervention_or_silence": {"type": "string", "maxLength": 280},
                "correctness_and_evidence": {"type": "string", "maxLength": 280},
                "manner_of_intervention": {"type": "string", "maxLength": 280},
                "audience_and_location": {"type": "string", "maxLength": 280},
                "overall": {"type": "string", "maxLength": 360},
            },
        },
    },
}


def enforce_deepnote_score_consistency(
    *,
    expected_intent: Literal["action", "silence"],
    delivered_count: int,
    raw_payload: dict[str, Any],
) -> dict[str, Any]:
    """Deterministically enforce Deepnote's hard zero-conditions & consistency invariants."""
    # Pure deterministic silence cases (no ambiguity)
    if delivered_count == 0:
        score = 10 if expected_intent == "silence" else 0
        reason = (
            "Correctly stayed silent on a silence-intent task (0 Slack messages delivered)."
            if expected_intent == "silence"
            else "Missed warranted intervention: delivered 0 Slack messages on an action-intent task."
        )
        return {
            "intervention_or_silence": score,
            "correctness_and_evidence": None,
            "manner_of_intervention": None,
            "audience_and_location": None,
            "overall_score": score,
            "triggered_important_mistakes": [] if score == 10 else ["missed_warranted_intervention"],
            "serious_mistake_zero": score == 0,
            "justifications": {
                "intervention_or_silence": reason,
                "correctness_and_evidence": "N/A (silence)",
                "manner_of_intervention": "N/A (silence)",
                "audience_and_location": "N/A (silence)",
                "overall": reason,
            },
        }

    def _opt_int(val: Any) -> int:
        return int(round(_coerce_rubric_score(val)))

    ios = _opt_int(raw_payload.get("intervention_or_silence", 0))
    ce = _opt_int(raw_payload.get("correctness_and_evidence", 0))
    mo = _opt_int(raw_payload.get("manner_of_intervention", 0))
    al = _opt_int(raw_payload.get("audience_and_location", 0))
    overall = _opt_int(raw_payload.get("overall_score", 0))
    serious_zero = bool(raw_payload.get("serious_mistake_zero", False))
    mistakes = [str(m) for m in (raw_payload.get("triggered_important_mistakes") or [])]

    # Hard Deepnote zero-conditions
    if expected_intent == "silence" and delivered_count > 0:
        ios = 0
        overall = 0
        serious_zero = True
        if not mistakes:
            mistakes = ["unwarranted_intervention_on_silence_task"]
    elif serious_zero:
        overall = 0

    # Deepnote consistency invariant: all applicable categories == 10 <=> overall == 10
    if ios == 10 and ce == 10 and mo == 10 and al == 10 and not serious_zero:
        overall = 10
    elif overall == 10 and not (ios == 10 and ce == 10 and mo == 10 and al == 10):
        overall = min(9, max(ios, ce, mo, al))

    just_raw = raw_payload.get("justifications")
    just = {
        "intervention_or_silence": str(just_raw.get("intervention_or_silence", "")) if isinstance(just_raw, dict) else "",
        "correctness_and_evidence": str(just_raw.get("correctness_and_evidence", "")) if isinstance(just_raw, dict) else "",
        "manner_of_intervention": str(just_raw.get("manner_of_intervention", "")) if isinstance(just_raw, dict) else "",
        "audience_and_location": str(just_raw.get("audience_and_location", "")) if isinstance(just_raw, dict) else "",
        "overall": str(just_raw.get("overall", "")) if isinstance(just_raw, dict) else "",
    }
    return {
        "intervention_or_silence": ios,
        "correctness_and_evidence": ce,
        "manner_of_intervention": mo,
        "audience_and_location": al,
        "overall_score": overall,
        "triggered_important_mistakes": mistakes,
        "serious_mistake_zero": serious_zero,
        "justifications": just,
    }


class DeepnoteOfficialJudge:
    """Blinded Dual-Pass Deepnote-Faithful Judge (supports primary OpenAI & cross-family AutoDL judges)."""

    def __init__(
        self,
        *,
        openai_client: OpenAIClient,
        judge_model: str,
        num_passes: int = 2,
        seed: int = 42,
    ) -> None:
        self.openai_client = openai_client
        self.judge_model = judge_model
        self.num_passes = max(1, num_passes)
        self.seed = seed

    def _build_blinded_prompt(
        self,
        *,
        task: DeepnoteOfficialTask,
        episode: OfficialEpisodeResult,
        pass_index: int,
    ) -> str:
        blind_tag = _anonymize_candidate_tag(
            f"{task.task_id}:{episode.model_id}", episode.method, self.seed + pass_index
        )
        snapshot_lines = []
        for m in episode.initial_context_summary:
            th = f" thread_ts={m.get('thread_ts')}" if m.get("thread_ts") else " (root)"
            snapshot_lines.append(
                f"- #{m.get('channel_name') or m.get('channel_id')} ({m.get('channel_id')}) | "
                f"ts={m.get('ts')}{th} | {m.get('author')} ({m.get('title') or 'member'}, {m.get('user_id')}): "
                f"{m.get('text')}"
            )
        snapshot_block = "\n".join(snapshot_lines) or "(snapshot unavailable)"

        evidence_lines = [
            f"- [{ev.get('service', '?')} / {ev.get('origin', '?')}] {ev.get('why', '')}"
            for ev in task.evidence
        ]
        evidence_block = "\n".join(evidence_lines) or "(see notes)"

        example_lines = []
        for i, ex in enumerate(task.example_messages, start=1):
            dest = ex.get("destination") or {}
            example_lines.append(f"Example {i} (destination={json.dumps(dest)}):\n{ex.get('text', '')}")
        example_block = "\n\n".join(example_lines) or "(SILENCE — no message should be sent)"

        mistakes_block = "\n".join(f"- {m}" for m in task.important_mistakes) or "(none listed)"

        if not episode.delivered_messages:
            delivery_block = "SILENT (0 Slack messages delivered)"
        else:
            d_lines = []
            for i, dm in enumerate(episode.delivered_messages, start=1):
                dest_desc = (
                    f"DM to user_id={dm.destination_user_id}"
                    if dm.tool == "slack_send_dm"
                    else f"#{dm.channel_name} ({dm.channel_id})"
                    + (f" thread_ts={dm.thread_ts}" if dm.thread_ts else " (channel root)")
                )
                d_lines.append(f"Delivered Message {i} via `{dm.tool}` -> {dest_desc}:\n{dm.text}")
            delivery_block = "\n\n".join(d_lines)

        read_trace_lines = []
        for idx, tr in enumerate(episode.tool_trace, start=1):
            if tr.get("kind") == "read":
                read_trace_lines.append(
                    f"{idx}. {tr.get('tool')}({json.dumps(tr.get('arguments', {}), ensure_ascii=False)[:180]}) "
                    f"ok={tr.get('ok')} -> {str(tr.get('result_preview', ''))[:260]}"
                )
        trace_block = "\n".join(read_trace_lines[:20]) or "(no read tool calls)"

        task_section = (
            f"Task ID: {task.task_id} | Category: {task.category} | Title: {task.title}\n"
            f"Authoritative Intent: {task.intent.upper()}\n"
            f"Situation: {task.situation}\n"
            f"Ideal Action: {task.ideal_action or 'STAY SILENT (no intervention needed)'}\n\n"
            f"Initial Recent Slack Snapshot (eligible messages):\n{snapshot_block}\n\n"
            f"Authoritative Ground-Truth Notes:\n{task.notes}\n\n"
            f"Verified Business Evidence Facts:\n{evidence_block}\n\n"
            f"Reference Example Message(s):\n{example_block}\n\n"
            f"IMPORTANT MISTAKES CHECKLIST (serious failures require overall_score=0):\n{mistakes_block}"
        )
        candidate_section = (
            f"Blinded Submission ID: {blind_tag}\n"
            f"Stop Reason: {episode.stop_reason} | Read Calls: {episode.read_calls_used} | Write Calls: {episode.write_calls_used}\n\n"
            f"ACTUAL DELIVERED SLACK MESSAGES:\n{delivery_block}\n\n"
            f"Background Read-Tool Trace (evidence only, NOT interventions):\n{trace_block}"
        )

        if pass_index % 2 == 0:
            return f"{task_section}\n\n{'=' * 40}\n{candidate_section}"
        return f"{candidate_section}\n\n{'=' * 40}\n{task_section}"

    async def grade_episode(
        self,
        *,
        task: DeepnoteOfficialTask,
        episode: OfficialEpisodeResult,
    ) -> DeepnoteRubricJudgment:
        # Pure silence episodes are scored deterministically per Deepnote's exact zero/ten rule
        if len(episode.delivered_messages) == 0:
            det = enforce_deepnote_score_consistency(
                expected_intent=task.intent,
                delivered_count=0,
                raw_payload={},
            )
            return DeepnoteRubricJudgment(
                task_id=task.task_id,
                method=episode.method,
                model_id=episode.model_id,
                judge_model=f"deterministic_silence:{self.judge_model}",
                intervention_or_silence=float(det["intervention_or_silence"]),
                correctness_and_evidence=None,
                manner_of_intervention=None,
                audience_and_location=None,
                overall_score=float(det["overall_score"]),
                triggered_important_mistakes=det["triggered_important_mistakes"],
                serious_mistake_zero=det["serious_mistake_zero"],
                justifications=det["justifications"],
                pass_scores=[det],
            )

        pass_results: list[dict[str, Any]] = []
        for pass_idx in range(self.num_passes):
            prompt = self._build_blinded_prompt(task=task, episode=episode, pass_index=pass_idx)
            res = await self.openai_client.complete_json(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": DEEPNOTE_STRICT_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                json_schema=DEEPNOTE_STRICT_JUDGE_JSON_SCHEMA,
                schema_name="deepnote_official_strict_rubric",
                temperature=0.0,
                seed=self.seed + pass_idx,
            )
            checked = enforce_deepnote_score_consistency(
                expected_intent=task.intent,
                delivered_count=len(episode.delivered_messages),
                raw_payload=dict(res.parsed_json or {}),
            )
            pass_results.append(checked)

        def _mean_key(key: str) -> float:
            vals = [float(p[key]) for p in pass_results if p.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        merged_mistakes = sorted(
            {m for p in pass_results for m in p.get("triggered_important_mistakes", [])}
        )
        return DeepnoteRubricJudgment(
            task_id=task.task_id,
            method=episode.method,
            model_id=episode.model_id,
            judge_model=f"dual{self.num_passes}:{self.judge_model}",
            intervention_or_silence=_mean_key("intervention_or_silence"),
            correctness_and_evidence=_mean_key("correctness_and_evidence"),
            manner_of_intervention=_mean_key("manner_of_intervention"),
            audience_and_location=_mean_key("audience_and_location"),
            overall_score=_mean_key("overall_score"),
            triggered_important_mistakes=merged_mistakes,
            serious_mistake_zero=any(bool(p.get("serious_mistake_zero")) for p in pass_results),
            justifications=dict(pass_results[0].get("justifications", {})),
            pass_scores=pass_results,
        )


# ---------------------------------------------------------------------------
# Summary & Comparison Helpers (Official-8 vs PageMantle-Ext diagnostic)
# ---------------------------------------------------------------------------


def summarize_official_8_results(
    episodes: list[OfficialEpisodeResult],
    judgments: dict[str, DeepnoteRubricJudgment],
) -> dict[str, Any]:
    """Compute aggregate Deepnote Official-8 metrics per (method, model_id)."""
    if not episodes:
        return {}

    overall_scores: list[float] = []
    ios_scores: list[float] = []
    ce_scores: list[float] = []
    mo_scores: list[float] = []
    al_scores: list[float] = []
    sil_correct = 0
    sil_total = 0
    act_correct = 0
    act_total = 0
    serious_zeros = 0
    total_tools = 0
    total_reads = 0
    total_writes = 0
    total_cost = 0.0
    total_model_sec = 0.0

    for ep in episodes:
        j = judgments.get(ep.task_id)
        if j is not None:
            overall_scores.append(j.overall_score)
            ios_scores.append(j.intervention_or_silence)
            if j.correctness_and_evidence is not None:
                ce_scores.append(j.correctness_and_evidence)
            if j.manner_of_intervention is not None:
                mo_scores.append(j.manner_of_intervention)
            if j.audience_and_location is not None:
                al_scores.append(j.audience_and_location)
            if j.serious_mistake_zero:
                serious_zeros += 1

        if ep.expected_intent == "silence":
            sil_total += 1
            if ep.predicted_action == "SILENT":
                sil_correct += 1
        else:
            act_total += 1
            if ep.predicted_action == "INTERVENE":
                act_correct += 1

        total_tools += ep.tool_calls_used
        total_reads += ep.read_calls_used
        total_writes += ep.write_calls_used
        total_cost += ep.total_cost_usd
        total_model_sec += ep.model_elapsed_sec

    n = len(episodes)
    avg = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    return {
        "n_tasks": n,
        "overall_mean": avg(overall_scores) or 0.0,
        "intervention_or_silence_mean": avg(ios_scores) or 0.0,
        "correctness_and_evidence_mean": avg(ce_scores),
        "manner_of_intervention_mean": avg(mo_scores),
        "audience_and_location_mean": avg(al_scores),
        "silence_accuracy": round(sil_correct / sil_total, 4) if sil_total else None,
        "silence_correct": f"{sil_correct}/{sil_total}",
        "action_recall": round(act_correct / act_total, 4) if act_total else None,
        "action_correct": f"{act_correct}/{act_total}",
        "serious_mistake_zeros": serious_zeros,
        "avg_tool_calls": round(total_tools / n, 2),
        "avg_read_calls": round(total_reads / n, 2),
        "avg_write_calls": round(total_writes / n, 2),
        "avg_model_sec": round(total_model_sec / n, 2),
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_usd": round(total_cost / n, 6),
    }


__all__ = [
    "CATEGORY_TO_SKILL",
    "DEEPNOTE_STRICT_JUDGE_JSON_SCHEMA",
    "DEEPNOTE_STRICT_JUDGE_SYSTEM_PROMPT",
    "DeepnoteOfficialJudge",
    "DeepnoteOfficialTask",
    "DeepnoteOfficialTaskSession",
    "DeepnoteRubricJudgment",
    "DeliveredSlackMessage",
    "FINISH_TASK_OPENAI_TOOL",
    "OfficialDirectAgent",
    "OfficialDialSkillAgent",
    "OfficialEpisodeResult",
    "SLACK_WRITE_TOOLS",
    "convert_deepnote_tool_to_openai",
    "enforce_deepnote_score_consistency",
    "format_initial_context_messages",
    "load_deepnote_official_catalog",
    "summarize_official_8_results",
]
