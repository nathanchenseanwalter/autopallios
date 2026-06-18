"""Generate the *student* notebooks from the *solution* notebooks — one source of truth.

We never hand-maintain two copies of a notebook. A mentor authors the full, working
solution once under ``notebooks/solutions/`` (percent-format ``.py``), marks the cells (or
cell *fragments*) that students should write themselves, and this tool strips those marked
regions down to a ``# TODO`` + ``raise NotImplementedError`` — producing the student copy
under ``notebooks/`` with an identical path tail. Edit the solution, re-run this, done.

Two ways to mark an exercise inside a solution ``.py``:

1. **A fragment of a cell** — wrap the lines students must write::

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

2. **A whole cell** — tag the percent cell header::

       # %% [exercise] write the precision/recall/F1 formulas
       precision = tp / (tp + fp)
       ...

   The whole cell body is replaced with the TODO + raise.

Run it::

    pixi run build-notebooks            # regenerate the student notebooks
    pixi run build-notebooks --check    # CI: fail if a student notebook is stale (drift)

(For centralized autograding with per-student feedback, ``nbgrader`` is the heavier
alternative — overkill for one repo shared by six students, so we don't use it.)
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


def build(solutions_dir: Path, out_dir: Path, *, check: bool) -> tuple[list[Path], list[Path]]:
    """Generate (or, in ``check`` mode, verify) every student notebook.

    Args:
        solutions_dir: Directory of solution ``.py`` files (searched recursively).
        out_dir: Directory the student ``.py`` files are written under (mirroring the path
            tail relative to ``solutions_dir``).
        check: If ``True``, do not write — only report which destinations are missing or
            stale relative to what would be generated.

    Returns:
        ``(written, drift)`` as lists of destination paths relative to ``out_dir``.
    """
    written: list[Path] = []
    drift: list[Path] = []
    for solution in sorted(solutions_dir.rglob("*.py")):
        rel = solution.relative_to(solutions_dir)
        dest = out_dir / rel
        student = strip_exercises(solution.read_text(encoding="utf-8"))
        if check:
            current = dest.read_text(encoding="utf-8") if dest.exists() else None
            if current != student:
                drift.append(rel)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(student, encoding="utf-8")
            written.append(rel)
    return written, drift


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (nonzero on drift in ``--check``)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify student notebooks are up to date instead of writing them (CI mode).",
    )
    parser.add_argument("--solutions-dir", type=Path, default=SOLUTIONS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    if not args.solutions_dir.exists():
        print(f"No solutions directory at {args.solutions_dir} — nothing to do.")
        return 0

    written, drift = build(args.solutions_dir, args.out_dir, check=args.check)

    if args.check:
        if drift:
            print("Student notebooks are STALE — run `pixi run build-notebooks`:")
            for rel in drift:
                print(f"  - {rel}")
            return 1
        print("Student notebooks are up to date.")
        return 0

    for rel in written:
        print(f"wrote notebooks/{rel}")
    print(f"Generated {len(written)} student notebook(s) from {args.solutions_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
