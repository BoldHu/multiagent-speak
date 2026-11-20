"""Validate all Colab notebooks (notebooks/00-06): nbformat schema, Python AST syntax, Drive/git header, zero secrets."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import nbformat

EXPECTED_NOTEBOOKS = [
    "00_setup_and_preflight.ipynb",
    "01_build_pagemantle_ext.ipynb",
    "02_pilot_verification.ipynb",
    "03_exp_when2speak_main.ipynb",
    "04_exp_pagemantle_and_online_dbcse.ipynb",
    "05_exp_ablations_and_scaling.ipynb",
    "06_analysis_and_paper_artifacts.ipynb",
]

_SECRET_LEAK_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b|\bjev_[A-Za-z0-9_-]{20,}\b")


def _strip_ipython_magics(source: str) -> str:
    """Replace line/shell magics (`%...`, `!...`) with valid no-op pass statements for AST parsing."""
    cleaned: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            indent = line[: len(line) - len(stripped)]
            cleaned.append(f"{indent}pass  # ipython magic")
        else:
            cleaned.append(line)
    return "\n".join(cleaned)


def validate_notebook_file(path: Path) -> list[str]:
    """Return a list of validation errors for a single .ipynb file (empty list = valid)."""
    errors: list[str] = []
    try:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
    except Exception as exc:
        return [f"{path.name}: nbformat validation failed: {exc}"]

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    if len(code_cells) < 3:
        errors.append(f"{path.name}: expected at least 3 code cells, found {len(code_cells)}")

    # Verify Universal Cell 0 (Drive mount) & Cell 1 (git pull + autoreload)
    all_code = "\n".join(c.source for c in code_cells)
    if "DIAL_SKILL_DRIVE_ROOT" not in all_code:
        errors.append(f"{path.name}: missing Universal Cell 0 (DIAL_SKILL_DRIVE_ROOT)")
    if "autoreload" not in all_code or "git" not in all_code:
        errors.append(f"{path.name}: missing Universal Cell 1 (git sync & autoreload)")

    # Verify zero hardcoded secret keys
    if _SECRET_LEAK_RE.search(all_code):
        errors.append(f"{path.name}: POTENTIAL HARDCODED API KEY DETECTED!")

    # Verify Python AST & bytecode compilation (allowing Colab/IPython top-level await)
    for idx, cell in enumerate(code_cells):
        sanitized = _strip_ipython_magics(cell.source)
        try:
            tree = ast.parse(sanitized)
            compile(tree, filename=f"{path.name}:cell_{idx}", mode="exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as exc:
            errors.append(f"{path.name} code cell #{idx}: SyntaxError line {exc.lineno}: {exc.msg}")

    return errors


def validate_all_notebooks(notebooks_dir: str | Path | None = None) -> list[str]:
    nb_dir = Path(notebooks_dir) if notebooks_dir else Path(__file__).resolve().parents[1] / "notebooks"
    all_errors: list[str] = []

    for filename in EXPECTED_NOTEBOOKS:
        fp = nb_dir / filename
        if not fp.is_file():
            all_errors.append(f"Missing expected notebook: {fp}")
            continue
        all_errors.extend(validate_notebook_file(fp))

    return all_errors


def main() -> int:
    errors = validate_all_notebooks()
    if errors:
        print("Notebook validation FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print(f"All {len(EXPECTED_NOTEBOOKS)} Colab notebooks validated cleanly (nbformat v4 + Python AST)!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
