"""Multi-source secret loader enforcing Invariant #6 (never hardcode or commit keys).

Lookup order:
1. Environment variables (`TYPESAFE_API_KEY` / `JEV_API_KEY`, `OPENAI_API_KEY_2` / `OPENAI_API_KEY`)
2. Google Drive secrets folder (`/content/drive/MyDrive/when2speak_research/secrets/` or `$DIAL_SKILL_DRIVE_ROOT/secrets/`)
   - Automatically prefers higher-numbered key files (e.g. `OPENAI_KEY_2.txt` before `OPENAI_KEY.txt`).
3. Colab `userdata` secrets (`google.colab.userdata`)
4. Local workspace root (`./OPENAI_KEY_2.txt`, `./OPENAI_KEY.txt`, `./JEV_API_KEY.txt`)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

from dial_skill.config import find_repo_root
from dial_skill.utils.drive import COLAB_DEFAULT_DRIVE_ROOT, get_drive_root


def mask_secret(value: str | None) -> str:
    """Return a safe masked representation for logs."""
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def _parse_key_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    # Pass 1: explicit key/token assignment (`KEY=...`, `API_KEY: ...`, etc.)
    for line in lines:
        for sep in ("=", ":"):
            if sep in line:
                lhs, rhs = line.split(sep, 1)
                lhs_l = lhs.strip().lower()
                cand = rhs.strip().strip('"').strip("'")
                if cand and not cand.startswith("http") and any(
                    tok in lhs_l for tok in ("key", "token", "secret")
                ):
                    return cand
    # Pass 2: bare API-key-like line (`sk-...`, `apik...`, or long opaque token without spaces/URLs)
    for line in lines:
        cand = line.strip('"').strip("'")
        if cand.startswith(("sk-", "apik", "ak-")) and " " not in cand:
            return cand
    for line in lines:
        if "=" in line:
            _, rhs = line.split("=", 1)
            candidate = rhs.strip().strip('"').strip("'")
            if candidate and not candidate.startswith("http"):
                return candidate
        cand = line.strip('"').strip("'")
        if not cand.startswith("http") and " " not in cand and len(cand) >= 24:
            return cand
    # Fallback: first non-empty non-comment line
    for line in lines:
        if "=" in line:
            _, rhs = line.split("=", 1)
            candidate = rhs.strip().strip('"').strip("'")
            if candidate:
                return candidate
        return line.strip('"').strip("'")
    return None


def _load_from_colab_userdata(secret_names: Sequence[str]) -> str | None:
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
    except Exception:
        return None
    for name in secret_names:
        try:
            val = userdata.get(name)
            if val and str(val).strip():
                return str(val).strip()
        except Exception:
            continue
    return None


def _list_candidate_key_files(directory: Path, primary_filename: str) -> list[Path]:
    """Return key files in priority order: highest numbered `<stem>_<N><suffix>` first, then `<primary_filename>`."""
    primary_path = directory / primary_filename
    stem = primary_path.stem
    suffix = primary_path.suffix
    numbered: list[tuple[int, Path]] = []
    if directory.is_dir():
        pattern = re.compile(rf"^{re.escape(stem)}_(\d+){re.escape(suffix)}$", re.IGNORECASE)
        try:
            for child in directory.iterdir():
                if not child.is_file():
                    continue
                match = pattern.match(child.name)
                if match:
                    numbered.append((int(match.group(1)), child))
        except OSError:
            pass
    numbered.sort(key=lambda pair: pair[0], reverse=True)
    candidates = [path for _, path in numbered]
    candidates.append(primary_path)
    return candidates


def _resolve_all_secrets(
    env_names: Sequence[str],
    filename: str,
    drive_root: str | Path | None = None,
    workspace_root: str | Path | None = None,
) -> list[tuple[str, str]]:
    """Return all discovered `(secret_value, source_label)` pairs in 4-tier priority order, deduplicated by value."""
    found: list[tuple[str, str]] = []
    seen_values: set[str] = set()

    def _add(val: str | None, source: str) -> None:
        if val and val not in seen_values:
            seen_values.add(val)
            found.append((val, source))

    # 1. Environment variables
    for env_name in env_names:
        val = os.environ.get(env_name)
        if val and val.strip():
            _add(val.strip(), f"env:{env_name}")

    # 2. Google Drive secrets folder (numbered keys e.g. OPENAI_KEY_2.txt preferred over OPENAI_KEY.txt)
    drive_dirs: list[Path] = [get_drive_root(drive_root) / "secrets"]
    if drive_root is None:
        colab_secrets_dir = COLAB_DEFAULT_DRIVE_ROOT / "secrets"
        if colab_secrets_dir not in drive_dirs:
            drive_dirs.append(colab_secrets_dir)
    for secrets_dir in drive_dirs:
        for candidate in _list_candidate_key_files(secrets_dir, filename):
            _add(_parse_key_file(candidate), f"drive:{candidate.name}")

    # 3. Colab userdata
    val = _load_from_colab_userdata(env_names)
    if val:
        _add(val, "colab_userdata")

    # 4. Local workspace root (numbered keys preferred over primary filename)
    ws_dirs: list[Path] = []
    if workspace_root is not None:
        ws_dirs.append(Path(workspace_root))
    else:
        ws_dirs.append(Path.cwd())
        repo_root = find_repo_root()
        if repo_root not in ws_dirs:
            ws_dirs.append(repo_root)
    for ws_dir in ws_dirs:
        for candidate in _list_candidate_key_files(ws_dir, filename):
            _add(_parse_key_file(candidate), f"local:{candidate.name}")

    return found


def _resolve_secret(
    env_names: Sequence[str],
    filename: str,
    drive_root: str | Path | None = None,
    workspace_root: str | Path | None = None,
) -> tuple[str | None, str]:
    """Return (secret_value, source_label) following the authoritative 4-tier lookup order."""
    resolved = _resolve_all_secrets(
        env_names=env_names,
        filename=filename,
        drive_root=drive_root,
        workspace_root=workspace_root,
    )
    if resolved:
        return resolved[0]
    return None, "missing"


def load_jev_api_key(
    drive_root: str | Path | None = None,
    workspace_root: str | Path | None = None,
    required: bool = False,
) -> str | None:
    """Load TypeSafe Jev API key following the 4-tier lookup order."""
    val, _ = _resolve_secret(
        env_names=("TYPESAFE_API_KEY", "JEV_API_KEY"),
        filename="JEV_API_KEY.txt",
        drive_root=drive_root,
        workspace_root=workspace_root,
    )
    if required and not val:
        raise RuntimeError(
            "Missing TypeSafe Jev API key. Set TYPESAFE_API_KEY/JEV_API_KEY, "
            "place JEV_API_KEY.txt in Drive secrets/, Colab userdata, or workspace root."
        )
    return val


def load_openai_api_keys(
    drive_root: str | Path | None = None,
    workspace_root: str | Path | None = None,
) -> list[tuple[str, str]]:
    """Return all discovered OpenAI keys as `(source_label, value)` in priority order (newest numbered key first)."""
    resolved = _resolve_all_secrets(
        env_names=("OPENAI_API_KEY_2", "OPENAI_API_KEY"),
        filename="OPENAI_KEY.txt",
        drive_root=drive_root,
        workspace_root=workspace_root,
    )
    return [(src, val) for val, src in resolved]


def load_openai_api_key(
    drive_root: str | Path | None = None,
    workspace_root: str | Path | None = None,
    required: bool = False,
) -> str | None:
    """Load OpenAI API key following the 4-tier lookup order (preferring numbered keys like OPENAI_KEY_2.txt)."""
    val, _ = _resolve_secret(
        env_names=("OPENAI_API_KEY_2", "OPENAI_API_KEY"),
        filename="OPENAI_KEY.txt",
        drive_root=drive_root,
        workspace_root=workspace_root,
    )
    if required and not val:
        raise RuntimeError(
            "Missing OpenAI API key. Set OPENAI_API_KEY, place OPENAI_KEY_2.txt / OPENAI_KEY.txt "
            "in Drive secrets/, Colab userdata, or workspace root."
        )
    return val


def load_autodl_api_key(
    drive_root: str | Path | None = None,
    workspace_root: str | Path | None = None,
    required: bool = False,
) -> str | None:
    """Load AutoDL OpenAI-compatible API key following the 4-tier lookup order."""
    val, _ = _resolve_secret(
        env_names=("AUTODL_API_KEY",),
        filename="AUTODL_KEY.txt",
        drive_root=drive_root,
        workspace_root=workspace_root,
    )
    if required and not val:
        raise RuntimeError(
            "Missing AutoDL API key. Set AUTODL_API_KEY, place AUTODL_KEY.txt "
            "in Drive secrets/, Colab userdata, or workspace root."
        )
    return val


def describe_secret_sources(
    drive_root: str | Path | None = None,
    workspace_root: str | Path | None = None,
    include_autodl: bool = False,
) -> dict[str, dict[str, str]]:
    """Return non-sensitive diagnostic info showing where each key was found."""
    jev_val, jev_src = _resolve_secret(
        ("TYPESAFE_API_KEY", "JEV_API_KEY"), "JEV_API_KEY.txt", drive_root, workspace_root
    )
    oai_val, oai_src = _resolve_secret(
        ("OPENAI_API_KEY_2", "OPENAI_API_KEY"), "OPENAI_KEY.txt", drive_root, workspace_root
    )
    out = {
        "jev": {"status": "found" if jev_val else "missing", "source": jev_src, "masked": mask_secret(jev_val)},
        "openai": {"status": "found" if oai_val else "missing", "source": oai_src, "masked": mask_secret(oai_val)},
    }
    if include_autodl:
        adl_val, adl_src = _resolve_secret(
            ("AUTODL_API_KEY",), "AUTODL_KEY.txt", drive_root, workspace_root
        )
        out["autodl"] = {
            "status": "found" if adl_val else "missing",
            "source": adl_src,
            "masked": mask_secret(adl_val),
        }
    return out
