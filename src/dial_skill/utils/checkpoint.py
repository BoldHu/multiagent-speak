"""Atomic JSONL checkpoint writer, resilient --resume loader, and run metadata logger."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class JsonlCheckpointWriter:
    """Append-only JSONL writer enforcing append + flush + os.fsync per record (Invariant #6)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: BaseModel | dict[str, Any]) -> None:
        if isinstance(record, BaseModel):
            line = record.model_dump_json()
        else:
            line = json.dumps(record, ensure_ascii=False, default=str)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


def load_jsonl(path: str | Path, ignore_corrupt_tail: bool = True) -> list[dict[str, Any]]:
    """Read JSONL records safely, optionally tolerating a partial final line from Colab disconnects."""
    file_path = Path(path)
    if not file_path.exists():
        return []
    records: list[dict[str, Any]] = []
    lines = file_path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            is_last = idx == len(lines) - 1
            if ignore_corrupt_tail and is_last:
                logger.warning("Ignoring truncated trailing JSONL line in %s: %s", file_path, exc)
                break
            raise
    return records


def load_completed_ids(path: str | Path, id_field: str = "sample_id") -> set[str]:
    """Return set of already-completed IDs in a checkpoint JSONL file for idempotent --resume."""
    completed: set[str] = set()
    for row in load_jsonl(path, ignore_corrupt_tail=True):
        val = row.get(id_field)
        if val is not None:
            completed.add(str(val))
    return completed


def get_git_revision(cwd: str | Path | None = None) -> str:
    """Return current git HEAD SHA or 'uncommitted' if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def save_run_metadata(
    run_dir: str | Path,
    config_payload: dict[str, Any],
    filename: str = "run_config.json",
) -> Path:
    """Persist run_config.json alongside git commit SHA and UTC timestamp."""
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": get_git_revision(),
        "config": config_payload,
    }
    out_path = out_dir / filename
    tmp_path = out_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)
    return out_path
