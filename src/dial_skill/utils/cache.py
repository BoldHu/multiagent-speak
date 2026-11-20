"""Persistent SQLite API response cache keyed by (provider, model, prompt_hash, temperature, seed)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from dial_skill.utils.drive import get_drive_root


def compute_prompt_hash(payload: Any) -> str:
    """Compute a deterministic SHA-256 hash over any JSON-serializable prompt payload."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SQLiteAPICache:
    """Thread-safe persistent SQLite response cache for Jev & OpenAI API calls.

    Enforces Invariant #6:
    Keyed strictly by `(provider, model, prompt_hash, temperature, seed)`.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            path = get_drive_root() / "caches" / "api_cache.sqlite"
        else:
            path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()
        self.hits = 0
        self.misses = 0

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_responses (
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    seed INTEGER NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (provider, model, prompt_hash, temperature, seed)
                )
                """
            )
            self._conn.commit()

    @staticmethod
    def _normalize_seed(seed: int | None) -> int:
        return -1 if seed is None else int(seed)

    @staticmethod
    def _normalize_temp(temperature: float | None) -> float:
        return round(0.0 if temperature is None else float(temperature), 6)

    def get(
        self,
        provider: str,
        model: str,
        prompt_hash: str,
        temperature: float = 0.0,
        seed: int | None = 42,
    ) -> dict[str, Any] | None:
        norm_temp = self._normalize_temp(temperature)
        norm_seed = self._normalize_seed(seed)
        with self._lock:
            row = self._conn.execute(
                """
                SELECT response_json FROM api_responses
                WHERE provider = ? AND model = ? AND prompt_hash = ? AND temperature = ? AND seed = ?
                """,
                (provider, model, prompt_hash, norm_temp, norm_seed),
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def set(
        self,
        provider: str,
        model: str,
        prompt_hash: str,
        response: dict[str, Any],
        temperature: float = 0.0,
        seed: int | None = 42,
    ) -> None:
        norm_temp = self._normalize_temp(temperature)
        norm_seed = self._normalize_seed(seed)
        serialized = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO api_responses
                (provider, model, prompt_hash, temperature, seed, response_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (provider, model, prompt_hash, norm_temp, norm_seed, serialized, time.time()),
            )
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM api_responses").fetchone()
            return int(row[0]) if row else 0

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "entries": self.count()}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
