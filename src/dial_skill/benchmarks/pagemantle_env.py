"""11-service PageMantle disk-backed harness, native Deepnote repo wrapper, and 180s timer (Invariant #7)."""

from __future__ import annotations

import csv
import importlib
import json
import logging
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dial_skill.schemas import ACTIVE_SKILLS, ChatMessage, InterventionTier, PageMantleTask
from dial_skill.utils.drive import get_drive_root

logger = logging.getLogger(__name__)

ALL_11_SERVICES: tuple[str, ...] = (
    "slack",
    "notion",
    "linear",
    "hubspot",
    "stripe",
    "posthog",
    "google_analytics",
    "google_search_console",
    "google_calendar",
    "bigquery",
    "snowflake",
)

SQL_WAREHOUSE_SERVICES = {"bigquery", "snowflake"}


@dataclass
class ModelBudgetTimer:
    """Tracks pure model wait time (excluding tool execution time) against Deepnote's 180s ceiling."""

    hard_cap_sec: float = 180.0
    soft_cap_sec: float = 90.0
    model_elapsed_sec: float = 0.0
    tool_elapsed_sec: float = 0.0
    _model_start: float | None = None
    _tool_start: float | None = None

    def start_model(self) -> None:
        self._model_start = time.perf_counter()

    def stop_model(self) -> float:
        if self._model_start is None:
            return 0.0
        dt = time.perf_counter() - self._model_start
        self.model_elapsed_sec += dt
        self._model_start = None
        return dt

    def start_tool(self) -> None:
        self._tool_start = time.perf_counter()

    def stop_tool(self) -> float:
        if self._tool_start is None:
            return 0.0
        dt = time.perf_counter() - self._tool_start
        self.tool_elapsed_sec += dt
        self._tool_start = None
        return dt

    @property
    def timed_out(self) -> bool:
        return self.model_elapsed_sec > self.hard_cap_sec

    @property
    def soft_cap_reached(self) -> bool:
        return self.model_elapsed_sec >= self.soft_cap_sec


def _discover_service_files(dataset_dir: Path, service: str) -> list[Path]:
    """Find disk files (json/jsonl/csv/parquet/sqlite/duckdb) associated with a service."""
    if not dataset_dir.exists():
        return []
    aliases = {
        "google_analytics": ["google_analytics", "ga", "ga4"],
        "google_search_console": ["google_search_console", "gsc", "search_console"],
        "google_calendar": ["google_calendar", "gcal", "calendar"],
    }.get(service, [service])

    found: list[Path] = []
    for alias in aliases:
        sub = dataset_dir / alias
        if sub.is_dir():
            for ext in ("*.json", "*.jsonl", "*.csv", "*.parquet", "*.sqlite", "*.db", "*.duckdb"):
                found.extend(sorted(sub.rglob(ext)))
        for ext in (".json", ".jsonl", ".csv", ".parquet", ".sqlite", ".db", ".duckdb"):
            direct = dataset_dir / f"{alias}{ext}"
            if direct.is_file():
                found.append(direct)
    # Deduplicate preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in found:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _iter_file_records(path: Path, max_rows: int = 5000) -> list[dict[str, Any]]:
    """Stream up to max_rows records from disk without loading 10 GB datasets into RAM."""
    suffix = path.suffix.lower()
    records: list[dict[str, Any]] = []

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_rows:
                    break
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            records.append(obj)
                    except json.JSONDecodeError:
                        continue
        return records

    if suffix == ".json":
        try:
            raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return []
        if isinstance(raw, list):
            return [r for r in raw[:max_rows] if isinstance(r, dict)]
        if isinstance(raw, dict):
            for val in raw.values():
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    return val[:max_rows]
            return [raw]
        return []

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                records.append(dict(row))
        return records

    if suffix == ".parquet":
        try:
            import duckdb  # type: ignore[import-not-found]

            con = duckdb.connect(database=":memory:")
            cur = con.execute(f"SELECT * FROM read_parquet(?) LIMIT {int(max_rows)}", [str(path)])
            cols = [c[0] for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            con.close()
            return rows
        except Exception:
            try:
                import pyarrow.parquet as pq

                table = pq.read_table(path)
                return table.slice(0, max_rows).to_pylist()
            except Exception:
                return []

    if suffix in (".sqlite", ".db"):
        con = sqlite3.connect(str(path))
        con.row_factory = sqlite3.Row
        try:
            tables = [
                r[0]
                for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            ]
            for tbl in tables:
                for row in con.execute(f'SELECT * FROM "{tbl}" LIMIT ?', (max_rows,)).fetchall():
                    rec = dict(row)
                    rec.setdefault("_table", tbl)
                    records.append(rec)
                    if len(records) >= max_rows:
                        return records
        finally:
            con.close()
        return records

    return records


def _run_sql_on_service_files(files: list[Path], sql: str, limit: int = 50) -> list[dict[str, Any]]:
    """Execute a read-only SQL query against Snowflake/BigQuery parquet/csv/sqlite/duckdb files."""
    sql_clean = sql.strip().rstrip(";")
    if not sql_clean.lower().startswith(("select", "with", "show", "describe", "pragma")):
        raise ValueError("Only read-only SELECT/WITH/SHOW queries are permitted.")

    # Prefer DuckDB if installed (supports parquet, csv, json, duckdb directly on disk)
    try:
        import duckdb  # type: ignore[import-not-found]

        duckdb_files = [f for f in files if f.suffix.lower() == ".duckdb"]
        if duckdb_files:
            con = duckdb.connect(str(duckdb_files[0]), read_only=True)
        else:
            con = duckdb.connect(database=":memory:")
            for f in files:
                tbl = f.stem.replace("-", "_")
                suf = f.suffix.lower()
                if suf == ".parquet":
                    con.execute(f'CREATE OR REPLACE VIEW "{tbl}" AS SELECT * FROM read_parquet(\'{f.as_posix()}\')')
                elif suf == ".csv":
                    con.execute(f'CREATE OR REPLACE VIEW "{tbl}" AS SELECT * FROM read_csv_auto(\'{f.as_posix()}\')')
                elif suf in (".json", ".jsonl"):
                    con.execute(f'CREATE OR REPLACE VIEW "{tbl}" AS SELECT * FROM read_json_auto(\'{f.as_posix()}\')')
        cur = con.execute(sql_clean)
        cols = [c[0] for c in cur.description] if cur.description else []
        rows = [dict(zip(cols, r)) for r in cur.fetchmany(limit)]
        con.close()
        return rows
    except ImportError:
        pass

    # Fallback: SQLite in-memory/file connector (zero extra deps, safe on 12 GB RAM)
    sqlite_files = [f for f in files if f.suffix.lower() in (".sqlite", ".db")]
    if sqlite_files:
        con = sqlite3.connect(str(sqlite_files[0]))
        con.row_factory = sqlite3.Row
        try:
            cur = con.execute(sql_clean)
            return [dict(r) for r in cur.fetchmany(limit)]
        finally:
            con.close()

    # Load small CSV/JSON tables into transient SQLite for offline queries
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    try:
        for f in files:
            recs = _iter_file_records(f, max_rows=2000)
            if not recs:
                continue
            tbl = f.stem.replace("-", "_")
            cols = sorted({k for r in recs for k in r.keys()})
            col_defs = ", ".join(f'"{c}" TEXT' for c in cols)
            con.execute(f'CREATE TABLE IF NOT EXISTS "{tbl}" ({col_defs})')
            placeholders = ", ".join("?" for _ in cols)
            quoted_cols = ", ".join(f'"{c}"' for c in cols)
            con.executemany(
                f'INSERT INTO "{tbl}" ({quoted_cols}) VALUES ({placeholders})',
                [[str(r.get(c, "")) for c in cols] for r in recs],
            )
        cur = con.execute(sql_clean)
        return [dict(r) for r in cur.fetchmany(limit)]
    finally:
        con.close()


class PageMantleEnv:
    """Disk-backed 11-service PageMantle harness wrapping Deepnote's native registry when available."""

    def __init__(
        self,
        *,
        dataset_dir: str | Path | None = None,
        benchmark_repo_dir: str | Path | None = None,
        max_tool_calls: int = 8,
        hard_time_cap_sec: float = 180.0,
        soft_time_cap_sec: float = 90.0,
    ) -> None:
        self.dataset_dir = (
            Path(dataset_dir) if dataset_dir else get_drive_root() / "datasets" / "pagemantle"
        )
        self.benchmark_repo_dir = Path(benchmark_repo_dir) if benchmark_repo_dir else None
        self.max_tool_calls = max_tool_calls
        self.timer = ModelBudgetTimer(
            hard_cap_sec=hard_time_cap_sec,
            soft_cap_sec=soft_time_cap_sec,
        )
        self.tool_call_count = 0
        self.call_history: list[dict[str, Any]] = []
        self._service_files: dict[str, list[Path]] = {
            svc: _discover_service_files(self.dataset_dir, svc) for svc in ALL_11_SERVICES
        }
        self._native_registry: Any | None = self._try_load_native_registry()

    def _try_load_native_registry(self) -> Any | None:
        """Inspect cloned deepnote/agent-initiative-benchmark and wrap its native tool registry if present."""
        if self.benchmark_repo_dir is None or not self.benchmark_repo_dir.is_dir():
            return None
        repo_str = str(self.benchmark_repo_dir.resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        for mod_name in (
            "agent_initiative_benchmark.tools",
            "pagemantle.tools",
            "src.tools",
            "tools",
        ):
            try:
                mod = importlib.import_module(mod_name)
                if hasattr(mod, "get_tools") or hasattr(mod, "ToolRegistry"):
                    logger.info("Wrapped native Deepnote tool module: %s", mod_name)
                    return mod
            except Exception:
                continue
        return None

    def reset_budget(self) -> None:
        self.tool_call_count = 0
        self.call_history.clear()
        self.timer = ModelBudgetTimer(
            hard_cap_sec=self.timer.hard_cap_sec,
            soft_cap_sec=self.timer.soft_cap_sec,
        )

    def available_services(self) -> list[str]:
        return [svc for svc, files in self._service_files.items() if files]

    def get_openai_tool_schemas(self, read_only: bool = False) -> list[dict[str, Any]]:
        """Expose OpenAI function-calling schemas covering all 11 PageMantle services."""
        schemas: list[dict[str, Any]] = []

        for svc in ALL_11_SERVICES:
            if svc in SQL_WAREHOUSE_SERVICES:
                schemas.append(
                    {
                        "type": "function",
                        "function": {
                            "name": f"{svc}_query_sql",
                            "description": f"Run a read-only SQL SELECT query on PageMantle's {svc} warehouse.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "sql": {"type": "string", "description": "Read-only SQL SELECT query"},
                                    "limit": {"type": "integer", "default": 25},
                                },
                                "required": ["sql"],
                            },
                        },
                    }
                )
                schemas.append(
                    {
                        "type": "function",
                        "function": {
                            "name": f"{svc}_list_tables",
                            "description": f"List available tables/files in PageMantle's {svc} warehouse.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                )
            else:
                schemas.append(
                    {
                        "type": "function",
                        "function": {
                            "name": f"{svc}_search",
                            "description": f"Search read-only records in PageMantle's {svc} service by keyword or ID.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "Keyword, entity name, or record ID"},
                                    "limit": {"type": "integer", "default": 10},
                                },
                                "required": ["query"],
                            },
                        },
                    }
                )
                schemas.append(
                    {
                        "type": "function",
                        "function": {
                            "name": f"{svc}_get_record",
                            "description": f"Fetch a specific record by ID or title from PageMantle's {svc} service.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "record_id": {"type": "string", "description": "Exact ID, ticket key, or title"},
                                },
                                "required": ["record_id"],
                            },
                        },
                    }
                )

        if not read_only:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "slack_post_message",
                        "description": "Record a proactive Slack message into a channel, thread, or DM.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "channel": {"type": "string"},
                                "text": {"type": "string"},
                                "thread_ts": {"type": "string"},
                            },
                            "required": ["channel", "text"],
                        },
                    },
                }
            )
        return schemas

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Dispatch a tool call across the 11 services while enforcing call budget & timing exclusion."""
        if self.tool_call_count >= self.max_tool_calls:
            return {
                "error": f"Tool call budget exceeded ({self.max_tool_calls}). Synthesize answer now."
            }

        self.tool_call_count += 1
        self.timer.start_tool()
        try:
            result = self._dispatch(name, arguments)
            self.call_history.append({"tool": name, "arguments": arguments, "ok": True})
            return result
        except Exception as exc:
            self.call_history.append({"tool": name, "arguments": arguments, "ok": False, "error": str(exc)})
            return {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            self.timer.stop_tool()

    def _dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        if name == "slack_post_message":
            return {"status": "posted", "channel": arguments.get("channel"), "text": arguments.get("text")}

        # Match service prefix (longest match first so google_search_console beats google_analytics)
        matched_service: str | None = None
        action = ""
        for svc in sorted(ALL_11_SERVICES, key=len, reverse=True):
            if name.startswith(f"{svc}_"):
                matched_service = svc
                action = name[len(svc) + 1 :]
                break

        if matched_service is None:
            raise ValueError(f"Unknown PageMantle tool '{name}'")

        files = self._service_files.get(matched_service, [])
        if action == "list_tables":
            return {"service": matched_service, "tables": [f.stem for f in files]}

        if action == "query_sql":
            if not files:
                return {"service": matched_service, "rows": [], "note": "No warehouse files found."}
            rows = _run_sql_on_service_files(
                files,
                sql=str(arguments.get("sql", "")),
                limit=int(arguments.get("limit", 25)),
            )
            return {"service": matched_service, "row_count": len(rows), "rows": rows}

        if action == "search":
            query = str(arguments.get("query", "")).strip().lower()
            limit = int(arguments.get("limit", 10))
            return self._search_service(matched_service, query=query, limit=limit)

        if action == "get_record":
            rec_id = str(arguments.get("record_id", "")).strip()
            return self._get_record(matched_service, record_id=rec_id)

        raise ValueError(f"Unsupported tool action '{action}' for service '{matched_service}'")

    def _search_service(self, service: str, query: str, limit: int = 10) -> dict[str, Any]:
        files = self._service_files.get(service, [])
        if not files:
            return {"service": service, "matches": []}
        tokens = [t for t in query.split() if len(t) >= 2]
        matches: list[dict[str, Any]] = []
        for path in files:
            for rec in _iter_file_records(path, max_rows=3000):
                hay = json.dumps(rec, ensure_ascii=False, default=str).lower()
                if not tokens or all(tok in hay for tok in tokens) or query in hay:
                    matches.append(rec)
                    if len(matches) >= limit:
                        return {"service": service, "matches": matches}
        return {"service": service, "matches": matches}

    def _get_record(self, service: str, record_id: str) -> dict[str, Any]:
        files = self._service_files.get(service, [])
        needle = record_id.strip().lower()
        id_keys = ("id", "record_id", "ticket_id", "issue_id", "invoice_id", "deal_id", "key", "title", "name", "slug")
        for path in files:
            for rec in _iter_file_records(path, max_rows=5000):
                for k in id_keys:
                    val = rec.get(k)
                    if val is not None and str(val).strip().lower() == needle:
                        return {"service": service, "found": True, "record": rec}
                # Fallback substring match if exact key differs
                if needle and needle in json.dumps(rec, ensure_ascii=False, default=str).lower():
                    return {"service": service, "found": True, "record": rec}
        return {"service": service, "found": False, "record": None}

    def verify_record_exists(self, service: str, record_id: str) -> bool:
        """Deterministic DB-existence verification used by PageMantle-Ext quality gate."""
        if service not in ALL_11_SERVICES:
            return False
        if service in SQL_WAREHOUSE_SERVICES:
            files = self._service_files.get(service, [])
            if any(f.stem.lower() == record_id.lower() for f in files):
                return True
        res = self._get_record(service, record_id)
        return bool(res.get("found"))

    def mine_verified_facts(self, limit_per_service: int = 8) -> list[dict[str, Any]]:
        """Extract real, verifiable records across all available PageMantle services."""
        facts: list[dict[str, Any]] = []
        id_keys = ("id", "record_id", "ticket_id", "issue_id", "invoice_id", "deal_id", "key", "title", "name")
        for svc in ALL_11_SERVICES:
            files = self._service_files.get(svc, [])
            svc_count = 0
            for path in files:
                for idx, rec in enumerate(_iter_file_records(path, max_rows=limit_per_service * 3)):
                    rec_id = None
                    for k in id_keys:
                        if rec.get(k):
                            rec_id = str(rec[k])
                            break
                    if rec_id is None:
                        rec_id = f"{path.stem}_{idx}"
                    facts.append(
                        {
                            "service": svc,
                            "source_file": path.name,
                            "record_id": rec_id,
                            "record": rec,
                        }
                    )
                    svc_count += 1
                    if svc_count >= limit_per_service:
                        break
                if svc_count >= limit_per_service:
                    break
        return facts


def load_official_tasks(
    *,
    benchmark_repo_dir: str | Path | None = None,
    cached_jsonl: str | Path | None = None,
) -> list[PageMantleTask]:
    """Load the 8 official Deepnote public tasks (pm_official_8) from JSONL or cloned repo."""
    if cached_jsonl is not None and Path(cached_jsonl).is_file():
        tasks = []
        for line in Path(cached_jsonl).read_text(encoding="utf-8").splitlines():
            if line.strip():
                tasks.append(PageMantleTask.model_validate_json(line))
        return tasks

    if benchmark_repo_dir is not None and Path(benchmark_repo_dir).is_dir():
        repo = Path(benchmark_repo_dir)
        raw_tasks: list[dict[str, Any]] = []
        for pattern in ("**/tasks*.json", "**/tasks*.jsonl", "**/scenarios/*.json", "**/eval_tasks/*.json"):
            for fp in sorted(repo.glob(pattern)):
                if fp.suffix == ".jsonl":
                    for line in fp.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            raw_tasks.append(json.loads(line))
                else:
                    obj = json.loads(fp.read_text(encoding="utf-8"))
                    if isinstance(obj, list):
                        raw_tasks.extend(x for x in obj if isinstance(x, dict))
                    elif isinstance(obj, dict):
                        raw_tasks.append(obj)
            if raw_tasks:
                break

        parsed: list[PageMantleTask] = []
        for idx, raw in enumerate(raw_tasks[:8]):
            cat = str(raw.get("category") or ACTIVE_SKILLS[idx % len(ACTIVE_SKILLS)]).lower()
            if cat not in ACTIVE_SKILLS:
                cat = ACTIVE_SKILLS[idx % len(ACTIVE_SKILLS)]
            raw_msgs = raw.get("messages") or raw.get("thread") or []
            msgs = [
                ChatMessage(
                    msg_id=str(m.get("msg_id") or f"off_{idx}_m{j}"),
                    speaker=str(m.get("speaker") or m.get("user") or f"emp_{j % 3}"),
                    text=str(m.get("text") or m.get("content") or ""),
                    timestamp=str(m.get("timestamp") or ""),
                    channel=str(raw.get("channel") or "#general"),
                )
                for j, m in enumerate(raw_msgs)
                if isinstance(m, dict)
            ]
            if not msgs:
                msgs = [
                    ChatMessage(
                        msg_id=f"off_{idx}_m0",
                        speaker="teammate",
                        text=str(raw.get("prompt") or raw.get("description") or f"Official task {idx + 1}"),
                        channel=str(raw.get("channel") or "#general"),
                    )
                ]
            expected_act = "SILENT" if str(raw.get("expected_action", "INTERVENE")).upper() == "SILENT" else "INTERVENE"
            tier_str = str(raw.get("ideal_tier") or ("SILENT" if expected_act == "SILENT" else "THREAD_BRIEF"))
            parsed.append(
                PageMantleTask(
                    task_id=str(raw.get("task_id") or raw.get("id") or f"pm_official_{idx + 1:02d}"),
                    split="pm_official_8",
                    category=cat,  # type: ignore[arg-type]
                    channel_name=str(raw.get("channel") or "#general"),
                    messages=msgs,
                    expected_action=expected_act,  # type: ignore[arg-type]
                    ideal_tier=InterventionTier(tier_str),
                    required_evidence_sources=list(raw.get("required_evidence_sources") or []),
                    reference_answer=str(raw.get("reference_answer") or raw.get("rubric") or ""),
                    ground_truth_facts=list(raw.get("ground_truth_facts") or []),
                )
            )
        if parsed:
            return parsed

    return []
