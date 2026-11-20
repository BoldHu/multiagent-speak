"""11-service PageMantle disk-backed harness, native Deepnote repo wrapper, and 180s timer (Invariant #7)."""

from __future__ import annotations

import csv
import importlib
import json
import logging
import sqlite3
import sys
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from dial_skill.schemas import ACTIVE_SKILLS, ChatMessage, InterventionTier, PageMantleTask
from dial_skill.utils.drive import get_drive_root, is_colab

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


_DATA_EXTENSIONS = {".json", ".jsonl", ".csv", ".parquet", ".sqlite", ".sqlite3", ".db", ".duckdb", ".md"}
_SKIP_DIR_NAMES = {".git", ".cache", ".huggingface", "__pycache__"}
_SKIP_FILE_NAMES = {"readme.md", ".gitattributes", ".gitignore", "license", "license.md"}

_SERVICE_ALIASES: dict[str, tuple[str, ...]] = {
    "slack": ("slack",),
    "notion": ("notion", "wiki"),
    "linear": ("linear",),
    "hubspot": ("hubspot", "crm"),
    "stripe": ("stripe", "billing"),
    "posthog": ("posthog",),
    "google_analytics": ("google_analytics", "google-analytics", "analytics", "ga4", "googleanalytics"),
    "google_search_console": ("google_search_console", "google-search-console", "search_console", "search-console", "gsc"),
    "google_calendar": ("google_calendar", "google-calendar", "gcal", "calendar"),
    "bigquery": ("bigquery", "bq"),
    "snowflake": ("snowflake",),
}

ARCHIVE_TO_SERVICE: dict[str, str] = {
    "slack": "slack",
    "notion": "notion",
    "linear": "linear",
    "hubspot": "hubspot",
    "stripe": "stripe",
    "posthog": "posthog",
    "analytics": "google_analytics",
    "google_analytics": "google_analytics",
    "search_console": "google_search_console",
    "google_search_console": "google_search_console",
    "calendar": "google_calendar",
    "google_calendar": "google_calendar",
    "bigquery": "bigquery",
    "snowflake": "snowflake",
}

_PRIMARY_ID_KEYS: tuple[str, ...] = (
    "id",
    "record_id",
    "ticket_id",
    "issue_id",
    "identifier",
    "invoice_id",
    "deal_id",
    "contact_id",
    "company_id",
    "hs_object_id",
    "customer_id",
    "subscription_id",
    "charge_id",
    "payment_intent",
    "event_id",
    "distinct_id",
    "uuid",
    "ts",
    "client_msg_id",
    "message_id",
    "channel_id",
    "page_id",
    "ical_uid",
    "key",
    "title",
    "summary",
    "name",
    "email",
    "url",
    "query",
    "slug",
)


def default_extract_dir(dataset_dir: Path) -> Path:
    """Use fast local SSD on Colab (/content/pagemantle_extracted) to avoid slow Drive FUSE writes."""
    if is_colab():
        return Path("/content/pagemantle_extracted")
    return dataset_dir / "extracted"


def _safe_extract_tar(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as tf:
        try:
            tf.extractall(path=dest_dir, filter="data")
        except TypeError:
            # Python < 3.12 fallback with path-traversal guard
            base = dest_dir.resolve()
            safe_members = []
            for m in tf.getmembers():
                target = (dest_dir / m.name).resolve()
                if str(target).startswith(str(base)):
                    safe_members.append(m)
            tf.extractall(path=dest_dir, members=safe_members)


def extract_pagemantle_archives(
    dataset_dir: str | Path,
    extract_dir: str | Path | None = None,
    services: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Idempotently unpack PageMantle data/<service>.tar.gz archives onto fast local storage."""
    ds_dir = Path(dataset_dir)
    out_root = Path(extract_dir) if extract_dir is not None else default_extract_dir(ds_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    wanted = set(services) if services is not None else set(ALL_11_SERVICES)
    status: dict[str, str] = {}

    archive_dirs = [ds_dir / "data", ds_dir]
    seen_services: set[str] = set()
    for adir in archive_dirs:
        if not adir.is_dir():
            continue
        for archive in sorted(adir.glob("*.tar.gz")) + sorted(adir.glob("*.tgz")):
            stem = archive.name
            if stem.endswith(".tar.gz"):
                stem = stem[:-7]
            elif stem.endswith(".tgz"):
                stem = stem[:-4]
            svc = ARCHIVE_TO_SERVICE.get(stem.lower().replace("-", "_"))
            if svc is None or svc not in wanted or svc in seen_services:
                continue
            seen_services.add(svc)
            svc_dest = out_root / svc
            marker = svc_dest / ".extracted_ok"
            if marker.is_file() and not force:
                status[svc] = "already_extracted"
                continue
            size_mb = archive.stat().st_size / (1024 * 1024)
            logger.info("Extracting %s (%.1f MB) -> %s", archive.name, size_mb, svc_dest)
            _safe_extract_tar(archive, svc_dest)
            marker.write_text(f"source={archive.name}\n", encoding="utf-8")
            status[svc] = "extracted"

    return status


def _list_all_data_files(dataset_dir: Path) -> list[Path]:
    """Recursively list all candidate data files under dataset_dir (ignoring .cache/.git/README)."""
    if not dataset_dir.exists():
        return []
    files: list[Path] = []
    for p in sorted(dataset_dir.rglob("*")):
        if not p.is_file():
            continue
        if any(part in _SKIP_DIR_NAMES or part.startswith(".cache") for part in p.relative_to(dataset_dir).parts):
            continue
        if p.name.lower() in _SKIP_FILE_NAMES:
            continue
        if p.suffix.lower() in _DATA_EXTENSIONS:
            files.append(p)
    return files


def _path_matches_service(rel_path_lower: str, service: str) -> bool:
    tokens = set(
        rel_path_lower.replace("\\", "/").replace("-", "_").replace(".", "/").split("/")
    )
    aliases = _SERVICE_ALIASES.get(service, (service,))
    for alias in aliases:
        norm_alias = alias.replace("-", "_")
        if norm_alias in tokens:
            return True
        if len(norm_alias) >= 4 and norm_alias in rel_path_lower.replace("-", "_"):
            return True
    # Special token check for short `ga` folder/stem
    if service == "google_analytics" and "ga" in tokens:
        return True
    return False


def _list_db_tables(path: Path) -> list[str]:
    suf = path.suffix.lower()
    if suf in (".sqlite", ".sqlite3", ".db"):
        try:
            con = sqlite3.connect(str(path))
            tables = [
                str(r[0]).lower()
                for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            ]
            con.close()
            return tables
        except Exception:
            return []
    if suf == ".duckdb":
        try:
            import duckdb  # type: ignore[import-not-found]

            con = duckdb.connect(str(path), read_only=True)
            tables = [str(r[0]).lower() for r in con.execute("SHOW TABLES").fetchall()]
            con.close()
            return tables
        except Exception:
            return []
    return []


def _discover_service_files(dataset_dir: Path, service: str) -> list[Path]:
    """Recursively find disk files (nested data/, parquet, sqlite, duckdb, json, md) for a service."""
    all_files = _list_all_data_files(dataset_dir)
    found: list[Path] = []
    unmatched_dbs: list[Path] = []

    for p in all_files:
        rel_lower = p.relative_to(dataset_dir).as_posix().lower()
        if _path_matches_service(rel_lower, service):
            found.append(p)
        elif p.suffix.lower() in (".sqlite", ".sqlite3", ".db", ".duckdb"):
            if not any(_path_matches_service(rel_lower, s) for s in ALL_11_SERVICES):
                unmatched_dbs.append(p)

    # If there is a unified company DB (e.g. pagemantle.duckdb / pagemantle.sqlite), inspect its tables
    for db_path in unmatched_dbs:
        tables = _list_db_tables(db_path)
        aliases = _SERVICE_ALIASES.get(service, (service,))
        if any(a.replace("-", "_") in tbl for tbl in tables for a in aliases):
            found.append(db_path)

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

    if suffix == ".md":
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                return []
            first_line = text.splitlines()[0].lstrip("#").strip() or path.stem
            return [{"id": path.stem, "title": first_line, "file": path.name, "content": text[:2000]}]
        except Exception:
            return []

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

    if suffix == ".duckdb":
        try:
            import duckdb  # type: ignore[import-not-found]

            con = duckdb.connect(str(path), read_only=True)
            try:
                tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
                for tbl in tables:
                    cur = con.execute(f'SELECT * FROM "{tbl}" LIMIT {int(max_rows)}')
                    cols = [c[0] for c in cur.description]
                    for row in cur.fetchall():
                        rec = dict(zip(cols, row))
                        rec.setdefault("_table", tbl)
                        records.append(rec)
                        if len(records) >= max_rows:
                            return records
            finally:
                con.close()
        except Exception:
            return []
        return records

    if suffix in (".sqlite", ".sqlite3", ".db"):
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
    sqlite_files = [f for f in files if f.suffix.lower() in (".sqlite", ".sqlite3", ".db")]
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
        extract_dir: str | Path | None = None,
        benchmark_repo_dir: str | Path | None = None,
        max_tool_calls: int = 8,
        hard_time_cap_sec: float = 180.0,
        soft_time_cap_sec: float = 90.0,
        auto_extract: bool = False,
    ) -> None:
        self.dataset_dir = (
            Path(dataset_dir) if dataset_dir else get_drive_root() / "datasets" / "pagemantle"
        )
        self.extract_dir = (
            Path(extract_dir) if extract_dir is not None else default_extract_dir(self.dataset_dir)
        )
        self.benchmark_repo_dir = Path(benchmark_repo_dir) if benchmark_repo_dir else None
        self.max_tool_calls = max_tool_calls
        self.timer = ModelBudgetTimer(
            hard_cap_sec=hard_time_cap_sec,
            soft_cap_sec=soft_time_cap_sec,
        )
        self.tool_call_count = 0
        self.call_history: list[dict[str, Any]] = []
        self._service_files: dict[str, list[Path]] = {}
        if auto_extract:
            self.ensure_extracted()
        else:
            self._refresh_service_files()
        self._native_registry: Any | None = self._try_load_native_registry()

    def _refresh_service_files(self) -> None:
        """Discover data files for all 11 services, preferring extracted full archives over sample/."""
        self._service_files = {}
        for svc in ALL_11_SERVICES:
            extracted_svc_dir = self.extract_dir / svc
            extracted_files = _list_all_data_files(extracted_svc_dir) if extracted_svc_dir.is_dir() else []
            if extracted_files:
                self._service_files[svc] = extracted_files
            else:
                self._service_files[svc] = _discover_service_files(self.dataset_dir, svc)

    def ensure_extracted(
        self,
        services: Sequence[str] | None = None,
        force: bool = False,
    ) -> dict[str, str]:
        """Extract data/*.tar.gz archives onto fast local SSD (idempotent) and refresh service index."""
        status = extract_pagemantle_archives(
            dataset_dir=self.dataset_dir,
            extract_dir=self.extract_dir,
            services=services,
            force=force,
        )
        self._refresh_service_files()
        return status

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

    def describe_dataset_layout(self, max_files: int = 40) -> dict[str, Any]:
        """Return diagnostic summary of raw files on disk vs matched services."""
        raw_files: list[str] = []
        if self.dataset_dir.exists():
            for p in sorted(self.dataset_dir.rglob("*")):
                if p.is_file() and not any(
                    part in _SKIP_DIR_NAMES or part.startswith(".cache")
                    for part in p.relative_to(self.dataset_dir).parts
                ):
                    size_mb = round(p.stat().st_size / (1024 * 1024), 2)
                    raw_files.append(f"{p.relative_to(self.dataset_dir).as_posix()} ({size_mb} MB)")
                    if len(raw_files) >= max_files:
                        break
        return {
            "dataset_dir": str(self.dataset_dir),
            "extract_dir": str(self.extract_dir),
            "raw_files_sample": raw_files,
            "service_file_counts": {svc: len(files) for svc, files in self._service_files.items()},
        }

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
        if not needle:
            return {"service": service, "found": False, "record": None}

        # O(1) fast path for Markdown/document files whose filename stem IS the record ID (e.g. Notion UUIDs)
        for path in files:
            if path.stem.lower() == needle:
                recs = _iter_file_records(path, max_rows=1)
                if recs:
                    return {"service": service, "found": True, "record": recs[0]}

        for path in files[:200]:
            for rec in _iter_file_records(path, max_rows=5000):
                for k in _PRIMARY_ID_KEYS:
                    val = rec.get(k)
                    if val is not None and str(val).strip().lower() == needle:
                        return {"service": service, "found": True, "record": rec}
                # Fallback substring match if exact key differs
                if needle in json.dumps(rec, ensure_ascii=False, default=str).lower():
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

    @staticmethod
    def _extract_record_id(rec: dict[str, Any]) -> str | None:
        for k in _PRIMARY_ID_KEYS:
            val = rec.get(k)
            if val is not None:
                s = str(val).strip()
                if s and len(s) <= 120:
                    return s
        # Fallback: first real non-empty scalar value from the record itself (never invent fake IDs!)
        for k, val in rec.items():
            if k.startswith("_") or val is None or isinstance(val, (dict, list, bool)):
                continue
            s = str(val).strip()
            if 3 <= len(s) <= 80:
                return s
        return None

    def mine_verified_facts(self, limit_per_service: int = 8) -> list[dict[str, Any]]:
        """Extract real, verifiable records across all available PageMantle services."""
        facts: list[dict[str, Any]] = []
        for svc in ALL_11_SERVICES:
            files = self._service_files.get(svc, [])
            svc_count = 0
            seen_ids: set[str] = set()
            per_file_cap = max(2, limit_per_service) if len(files) <= 3 else max(1, limit_per_service // min(len(files), limit_per_service))
            for path in files[: max(limit_per_service * 3, 24)]:
                for rec in _iter_file_records(path, max_rows=per_file_cap * 3):
                    rec_id = self._extract_record_id(rec)
                    if not rec_id or rec_id in seen_ids:
                        continue
                    seen_ids.add(rec_id)
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
