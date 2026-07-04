"""Generate the *student* notebooks from the *solution* notebooks, one source of truth.

We never hand-maintain two copies of a notebook. A mentor authors the full, working
solution once under ``notebooks/solutions/`` (percent-format ``.py``), marks the cells (or
cell *fragments*) that students should write themselves, and this tool strips those marked
regions down to a ``# TODO`` + ``raise NotImplementedError``, producing the student copy
under ``notebooks/`` with an identical path tail. Edit the solution, re-run this, done.

The one source of truth is the **solution** ``.py`` (percent format: clean diffs, lintable,
what mentors review). From it this tool generates, for each notebook:

* a student ``.ipynb`` — what students open and work in (outputs stripped, exercises blanked);
* a student ``.py`` — a runnable, ``main()``-wrapped *script* twin of that notebook (see
  :func:`percent_to_script`): after a student solves the notebook they paste their code into
  the matching blank here and ``python NN_name.py`` runs the whole lesson headless — the
  bridge from "explore in a notebook" to "automate as a script";
* a solution ``.ipynb`` — the worked answer key, rendered beside the solution source.

Both ``.ipynb`` files are generated (outputs stripped, deterministic cell ids), so a fresh
clone has working notebooks with no conversion step. Students still convert by hand as an
exercise, but nobody has to before they can open a notebook.

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
import ast
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


_NOQA_E402 = "# noqa: E402"


def _build_docstring(md_source: str, stem: str) -> str:
    """Turn a notebook's opening markdown cell into a module docstring.

    The markdown H1 becomes the summary line (tagged as the script twin of the notebook)
    and the remaining prose becomes the body. A raw string is used when the text contains
    backslashes (LaTeX like ``\\frac``) so they survive verbatim without escape warnings.

    Args:
        md_source: The first markdown cell's source (jupytext-decoded, no ``# `` prefixes).
        stem: The notebook's file stem, e.g. ``03_implement_iou``.

    Returns:
        A complete triple-quoted docstring literal (its own line(s) of source).
    """
    lines = md_source.splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        title = lines[0].lstrip("#").strip()
        rest = "\n".join(lines[1:]).strip("\n")
    else:
        title = stem
        rest = md_source.strip("\n")
    summary = f"{title} — runnable-script twin of {stem}.ipynb."
    body = summary if not rest.strip() else f"{summary}\n\n{rest}\n"
    if '"""' not in body and not body.endswith("\\"):
        prefix = "r" if "\\" in body else ""
        return f'{prefix}"""{body}"""'
    escaped = body.replace("\\", "\\\\").replace('"""', r"\"\"\"")
    return f'"""{escaped}"""'


def _md_to_comment(md_source: str) -> list[str]:
    """Render a markdown cell as Python comment lines (blank lines kept as bare ``#``)."""
    return ["#" if not line.strip() else f"# {line}" for line in md_source.strip("\n").splitlines()]


def _strip_blank_edges(lines: list[str]) -> list[str]:
    """Drop leading/trailing blank lines from a block, keeping internal spacing."""
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def _import_top_module(line: str) -> str:
    """Return the top-level module name an ``import`` line refers to (``''`` if not one)."""
    line = line.strip()
    if line.startswith("from "):
        module = line[len("from ") :].split(" import", 1)[0].strip()
    elif line.startswith("import "):
        module = line[len("import ") :].split(",", 1)[0].strip().split(" as ", 1)[0].strip()
    else:
        return ""
    return "autopallios" if module.startswith(".") else module.split(".")[0]


def _sort_imports(lines: list[str]) -> list[str]:
    """Group imports (future, stdlib, third-party, first-party) and sort within each group.

    Uses :data:`sys.stdlib_module_names` to tell stdlib from third-party, so the hoisted
    imports read like a normal, isort-style script header (blank line between groups).
    """
    groups: dict[int, list[str]] = {0: [], 1: [], 2: [], 3: []}
    for line in lines:
        top = _import_top_module(line)
        if top == "__future__":
            groups[0].append(line)
        elif top in sys.stdlib_module_names:
            groups[1].append(line)
        elif top == "autopallios":
            groups[3].append(line)
        else:
            groups[2].append(line)
    out: list[str] = []
    for key in (0, 1, 2, 3):
        if groups[key]:
            if out:
                out.append("")
            out += sorted(groups[key], key=lambda s: (_import_top_module(s).lower(), s.lower()))
    return out


def _is_def(node: ast.AST) -> bool:
    """True for a top-level function/class definition (kept at module scope in the script)."""
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))


def _is_import(node: ast.AST) -> bool:
    """True for a top-level ``import`` / ``from ... import`` statement (hoisted to the top)."""
    return isinstance(node, (ast.Import, ast.ImportFrom))


def percent_to_script(py_text: str, *, stem: str) -> str:
    """Render exercise-stripped percent text as a runnable, ``main()``-wrapped script.

    This is the *automation* twin of a notebook: the opening markdown becomes the module
    docstring, imports are hoisted to the top, top-level ``def``\\s stay at module scope
    (so a student pastes their notebook function straight in), every other statement moves
    into ``main()``, and each ``plt.show()`` becomes a ``savefig`` into ``output/`` — the
    concrete "a script saves its results instead of popping a window" lesson. Section
    markdown is preserved as comments above the code it introduced.

    Args:
        py_text: Percent-format source, already exercise-stripped for students.
        stem: The notebook's file stem, used for the docstring and figure filenames.

    Returns:
        The script text, newline-terminated and deterministic (safe for ``--check``).

    Raises:
        SystemExit: If jupytext is not importable (run under a pixi env).
    """
    try:
        import jupytext
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            f"build-notebooks needs '{exc.name}' to render script twins. "
            "Run it through pixi, e.g. `pixi run build-notebooks`."
        ) from exc

    cells = jupytext.reads(py_text, fmt="py:percent").cells

    imports: list[str] = []
    seen_imports: set[str] = set()
    module_blocks: list[list[str]] = []
    main_blocks: list[list[str]] = []
    pending_md: list[str] = []
    docstring: str | None = None
    state = {"figs": 0, "needs_output": False}

    def add_import(line: str) -> None:
        line = line.replace(f"  {_NOQA_E402}", "").replace(f" {_NOQA_E402}", "").rstrip()
        key = line.strip()
        if key and key not in seen_imports:
            seen_imports.add(key)
            imports.append(line)

    def convert_shows(lines: list[str]) -> list[str]:
        out: list[str] = []
        for line in lines:
            if line.strip() == "plt.show()":
                indent = line[: len(line) - len(line.lstrip())]
                state["figs"] += 1
                state["needs_output"] = True
                name = f"{stem}_fig{state['figs']}.png"
                out.append(f'{indent}plt.savefig(OUTPUT_DIR / "{name}", dpi=150, bbox_inches="tight")')
                out.append(f"{indent}plt.close()")
            else:
                out.append(line)
        return out

    def indent4(lines: list[str]) -> list[str]:
        return [f"    {line}" if line.strip() else "" for line in lines]

    for idx, cell in enumerate(cells):
        if cell.cell_type == "markdown":
            if docstring is None and idx == 0:
                docstring = _build_docstring(cell.source, stem)
            else:
                block = _md_to_comment(cell.source)
                if block:
                    pending_md = pending_md + ["#"] + block if pending_md else block
            continue

        src_lines = cell.source.splitlines()
        try:
            body = ast.parse(cell.source).body
        except SyntaxError:  # pragma: no cover - notebooks are valid Python
            block = indent4(convert_shows(src_lines))
            main_blocks.append(indent4(pending_md) + block if pending_md else block)
            pending_md = []
            continue
        if not body:
            if src_lines:
                main_blocks.append(indent4(pending_md + src_lines) if pending_md else indent4(src_lines))
                pending_md = []
            continue

        # A cell's defs stay at module scope; everything else moves into main(). Accumulate
        # per cell (not per statement) so the notebook's own line grouping is preserved.
        primary = "module" if any(_is_def(n) for n in body) else "main"
        cell_module: list[str] = []
        cell_main: list[str] = []
        for i, node in enumerate(body):
            start = 1 if i == 0 else body[i - 1].end_lineno + 1
            end = node.end_lineno if i < len(body) - 1 else len(src_lines)
            chunk = src_lines[start - 1 : end]
            if _is_import(node):
                for line in src_lines[node.lineno - 1 : node.end_lineno]:
                    add_import(line)
            elif _is_def(node):
                cell_module += convert_shows(chunk)
            else:
                cell_main += convert_shows(chunk)

        cell_module = _strip_blank_edges(cell_module)
        cell_main = _strip_blank_edges(cell_main)
        if primary == "module" and pending_md:
            cell_module = pending_md + cell_module
        elif pending_md:  # goes into main(), so comment and code are indented together
            cell_main = pending_md + cell_main
        if cell_module:
            module_blocks.append(cell_module)
        if cell_main:
            main_blocks.append(indent4(cell_main))
        pending_md = []

    out: list[str] = []
    if docstring:
        out += [docstring, ""]
    if state["needs_output"] and "from pathlib import Path" not in seen_imports:
        imports.append("from pathlib import Path")
    out += _sort_imports(imports)
    if state["needs_output"]:
        out += ["", 'OUTPUT_DIR = Path(__file__).resolve().parent / "output"']
    for block in module_blocks:
        out += ["", ""] + block
    out += ["", "", "def main() -> None:"]
    out.append('    """Run this lesson end to end — the notebook, as an automatable script."""')
    if state["needs_output"]:
        out.append("    OUTPUT_DIR.mkdir(exist_ok=True)")
    body_lines: list[str] = []
    for i, block in enumerate(main_blocks):
        if i:
            body_lines.append("")
        body_lines += block
    out += body_lines if any(line.strip() for line in body_lines) else ["    pass"]
    if pending_md:  # a closing markdown cell (e.g. "Next: ...") with no code after it
        out += ["", ""] + pending_md
    out += ["", "", 'if __name__ == "__main__":', "    main()"]
    return "\n".join(out).rstrip("\n") + "\n"


def _artifacts_for(solution: Path, solutions_dir: Path, out_dir: Path) -> dict[Path, str]:
    """Compute every generated file for one solution ``.py`` as ``{dest: content}``.

    Produces the student ``.py`` (a runnable, ``main()``-wrapped *script* twin with the
    exercises blanked), the student ``.ipynb`` twin (the notebook students work in), and the
    solution ``.ipynb`` twin (rendered in place beside the solution source).
    """
    rel = solution.relative_to(solutions_dir)
    source = solution.read_text(encoding="utf-8")
    student_py = strip_exercises(source)
    return {
        out_dir / rel: percent_to_script(student_py, stem=rel.stem),
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
