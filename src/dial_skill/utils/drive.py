"""Colab / Google Drive and local persistent storage path resolver."""

from __future__ import annotations

import os
from pathlib import Path

COLAB_DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/when2speak_research")
DRIVE_SUBDIRS = (
    "secrets",
    "config_cache",
    "datasets/pagemantle",
    "datasets/when2speak",
    "datasets/pagemantle_ext",
    "caches",
    "runs",
    "paper_artifacts/tables",
    "paper_artifacts/figures",
)


def is_colab() -> bool:
    """Return True when running inside Google Colab."""
    return "COLAB_RELEASE_TAG" in os.environ or Path("/content").exists()


def get_drive_root(override: str | Path | None = None) -> Path:
    """Resolve persistent research root ($DIAL_SKILL_DRIVE_ROOT, Drive on Colab, or ./local_drive_mirror)."""
    if override is not None:
        return Path(override).expanduser().resolve()
    env_root = os.environ.get("DIAL_SKILL_DRIVE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    if is_colab() and Path("/content/drive/MyDrive").exists():
        return COLAB_DEFAULT_DRIVE_ROOT
    return (Path.cwd() / "local_drive_mirror").resolve()


def ensure_drive_layout(drive_root: str | Path | None = None) -> Path:
    """Create standard persistent subdirectories under DRIVE_ROOT and return the root Path."""
    root = get_drive_root(drive_root)
    for sub in DRIVE_SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root
