"""Generate the *student* notebooks from the *solution* notebooks, one source of truth.

We never hand-maintain two copies of a notebook. A mentor authors the full, working
solution once under ``notebooks/solutions/`` (percent-format ``.py``), marks the cells (or
cell *fragments*) that students should write themselves, and this tool strips those marked
regions down to a ``# TODO`` + ``raise NotImplementedError``, producing the student copy
under ``notebooks/`` with an identical path tail. Edit the solution, re-run this, done.

Every notebook ships in two formats, kept in sync by this tool:

* the percent-format ``.py`` — the source of truth (clean diffs, lintable, code review);
* a committed ``.ipynb`` twin — what students open in JupyterLab, and what GitHub renders.

The ``.ipynb`` is generated from the ``.py`` (outputs stripped, deterministic cell ids), so
a fresh clone has working notebooks with no conversion step. Students still convert by hand
as an exercise, but nobody has to before they can open a notebook.

Two ways to mark an exercise inside a solution ``.py``:

1. **A fragment of a cell**, wrap the lines students must write::

       def iou(a, b):
           a = a > 0
           b = b > 0
           # >>> exercise: intersection / union (guard divide-by-zero)
           inter = (a & b).sum()
           union = (a | b).sum()
           return inter / union if union else 0.0
           # <<< exercise

   Everything *outside* the markers (signature, setup, the grader cell that follows) is
   copied verbatim; everything *between* them becomes the TODO + raise.

2. **A whole cell**, tag the percent cell header::

       # %% [exercise] write the precision/recall/F1 formulas
       precision = tp / (tp + fp)
       ...

   The whole cell body is replaced with the TODO + raise.

Run it::

    pixi run build-notebooks            # regenerate the student notebooks (.py + .ipynb)
    pixi run build-notebooks --check    # CI: fail if any generated notebook is stale (drift)

(For centralized autograding with per-student feedback, ``nbgrader`` is the heavier
alternative, overkill for one repo shared by six students, so we don't use it.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: Repo root is the parent of this ``tools/`` directory.
ROOT = Path(__file__).resolve().parents[1]
SOLUTIONS_DIR = ROOT / "notebooks" / "solutions"
OUT_DIR = ROOT / "notebooks"

_EXERCISE_OPEN = "# >>> exercise"
_EXERCISE_CLOSE = "# <<< exercise"
_CELL_MARK = "# %%"
_CELL_EXERCISE = "[exercise]"


def strip_exercises(source: str) -> str:
    """Return the student version of a solution notebook's source text.

    Replaces every ``# >>> exercise ... # <<< exercise`` fragment and every
    ``# %% [exercise]`` cell body with a ``# TODO`` line and a ``raise
    NotImplementedError`` carrying the hint. All other lines are preserved exactly.

    Args:
        source: The full text of a solution ``.py`` (percent format).

    Returns:
        The student-facing text, with exercises blanked.
    """
    lines = source.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # (1) Whole-cell exercise: a percent cell header tagged [exercise].
        if stripped.startswith(_CELL_MARK) and _CELL_EXERCISE in stripped:
            hint = stripped.split(_CELL_EXERCISE, 1)[1].strip() or "complete this cell"
            header = line.replace(f" {_CELL_EXERCISE}", "").replace(_CELL_EXERCISE, "").rstrip()
            out.append(header)
            out.append(f"# TODO(you): {hint}")
            out.append(f'raise NotImplementedError("Exercise: {hint}")')
            i += 1
            while i < n and not lines[i].lstrip().startswith(_CELL_MARK):
                i += 1
            continue

        # (2) Inline fragment exercise: replace everything between the markers.
        if stripped.startswith(_EXERCISE_OPEN):
            indent = line[: len(line) - len(line.lstrip())]
            hint = stripped[len(_EXERCISE_OPEN) :].lstrip(": ").strip() or "implement this"
            out.append(f"{indent}# TODO(you): {hint}")
            out.append(f'{indent}raise NotImplementedError("Exercise: {hint}")')
            i += 1
            while i < n and not lines[i].strip().startswith(_EXERCISE_CLOSE):
                i += 1
            i += 1  # skip the closing marker itself
            continue

        out.append(line)
        i += 1

    text = "\n".join(out)
    if source.endswith("\n"):
        text += "\n"
    return text


def _untag_exercise_headers(source: str) -> str:
    """Turn ``# %% [exercise] hint`` headers into plain ``# %%`` cells.

    The ``[exercise]`` tag is meaningful only to :func:`strip_exercises`; it is not a valid
    notebook cell type. In a *solution* the answer is already present, so for rendering we
    just drop the tag, leaving a normal code cell. Inline ``# >>> exercise`` markers are
    ordinary comments and are kept as-is (they show mentors the exercise boundaries).

    Args:
        source: The full text of a solution ``.py`` (percent format).

    Returns:
        The same text with every ``[exercise]`` cell tag removed from its header.
    """
    out: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(_CELL_MARK) and _CELL_EXERCISE in stripped:
            line = line.replace(f" {_CELL_EXERCISE}", "").replace(_CELL_EXERCISE, "").rstrip()
        out.append(line)
    text = "\n".join(out)
    if source.endswith("\n"):
        text += "\n"
    return text


def percent_to_ipynb(py_text: str) -> str:
    """Render percent-format ``.py`` text as a clean, deterministic ``.ipynb`` document.

    The result is a plain notebook (no jupytext round-trip metadata), with all outputs
    stripped and stable, index-based cell ids — so re-running the build produces a
    byte-identical file and ``--check`` can detect real drift rather than churn.

    Args:
        py_text: Percent-format notebook source (already exercise-stripped for students,
            or untagged for solutions).

    Returns:
        The serialized ``.ipynb`` JSON, newline-terminated.

    Raises:
        SystemExit: If jupytext / nbformat are not importable (run under a pixi env).
    """
    try:
        import jupytext
        import nbformat
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            f"build-notebooks needs '{exc.name}' to render .ipynb twins. "
            "Run it through pixi, e.g. `pixi run build-notebooks`."
        ) from exc

    nb = jupytext.reads(py_text, fmt="py:percent")
    nb.metadata.pop("jupytext", None)  # keep the .ipynb standalone, not a jupytext twin
    nb.nbformat, nb.nbformat_minor = 4, 5
    for index, cell in enumerate(nb.cells):
        cell["id"] = f"cell-{index}"  # deterministic ids => stable diffs & meaningful --check
        cell.get("metadata", {}).pop("lines_to_next_cell", None)
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    text = nbformat.writes(nb, version=4)
    return text if text.endswith("\n") else text + "\n"


def _artifacts_for(solution: Path, solutions_dir: Path, out_dir: Path) -> dict[Path, str]:
    """Compute every generated file for one solution ``.py`` as ``{dest: content}``.

    Produces the student ``.py`` (exercises blanked), the student ``.ipynb`` twin, and the
    solution ``.ipynb`` twin (rendered in place beside the solution source).
    """
    rel = solution.relative_to(solutions_dir)
    source = solution.read_text(encoding="utf-8")
    student_py = strip_exercises(source)
    return {
        out_dir / rel: student_py,
        (out_dir / rel).with_suffix(".ipynb"): percent_to_ipynb(student_py),
        solution.with_suffix(".ipynb"): percent_to_ipynb(_untag_exercise_headers(source)),
    }


def build(solutions_dir: Path, out_dir: Path, *, check: bool) -> tuple[list[Path], list[Path]]:
    """Generate (or, in ``check`` mode, verify) every generated notebook artifact.

    For each solution ``.py`` this writes the student ``.py``, the student ``.ipynb``, and
    the solution ``.ipynb`` (the solution ``.py`` itself is the source and never written).

    Args:
        solutions_dir: Directory of solution ``.py`` files (searched recursively).
        out_dir: Directory the student files are written under (mirroring the path tail
            relative to ``solutions_dir``).
        check: If ``True``, do not write, only report which destinations are missing or
            stale relative to what would be generated.

    Returns:
        ``(written, drift)`` as lists of destination paths relative to the repo root.
    """
    written: list[Path] = []
    drift: list[Path] = []
    for solution in sorted(solutions_dir.rglob("*.py")):
        for dest, content in _artifacts_for(solution, solutions_dir, out_dir).items():
            rel_to_root = dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest
            if check:
                current = dest.read_text(encoding="utf-8") if dest.exists() else None
                if current != content:
                    drift.append(rel_to_root)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
                written.append(rel_to_root)
    return written, drift


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (nonzero on drift in ``--check``)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify generated notebooks are up to date instead of writing them (CI mode).",
    )
    parser.add_argument("--solutions-dir", type=Path, default=SOLUTIONS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    if not args.solutions_dir.exists():
        print(f"No solutions directory at {args.solutions_dir}, nothing to do.")
        return 0

    written, drift = build(args.solutions_dir, args.out_dir, check=args.check)

    if args.check:
        if drift:
            print("Generated notebooks are STALE, run `pixi run build-notebooks`:")
            for rel in drift:
                print(f"  - {rel}")
            return 1
        print("Generated notebooks are up to date.")
        return 0

    for rel in written:
        print(f"wrote {rel}")
    print(f"Generated {len(written)} file(s) from {args.solutions_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
