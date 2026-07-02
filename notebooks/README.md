# Notebooks, where you do the work

The library (`autopallios/`) is the finished tool. These notebooks are where *you* build
understanding: each week is a set of notebooks that runs top to bottom and prints a metric.

## Launch

```bash
pixi run -e teach lab        # opens JupyterLab on this folder
```

Every notebook is committed in **two formats**:

- **`.ipynb`** — the notebook you open and run in JupyterLab, straight from a fresh
  `git clone` with no conversion step.
- **`.py`** (percent format) — the same notebook as a plain script: clean diffs, lintable,
  and what mentors review. This is the **source of truth**.

`pixi run build-notebooks` rebuilds each `.ipynb` from its `.py` (outputs stripped); editing
a notebook in JupyterLab syncs the pair back the other way through jupytext. **Commit both.**

### Convert between the two yourself

Learn to do this by hand as well, it's how a notebook becomes a script other code can
import. The commands:

```bash
jupytext --to py:percent 03_implement_iou.ipynb    # notebook -> script
jupytext --to notebook   03_implement_iou.py       # script   -> notebook
jupyter nbconvert --to script 03_implement_iou.ipynb   # the nbconvert way (-> .py)
jupytext --sync          03_implement_iou.ipynb    # update whichever twin is stale
```

It's also the first move in promoting a notebook function into the shared, tested library,
see [From notebook to library](../docs/from_notebook_to_library.md).

## Work in your own folder

Copy the notebook you're on into your own subfolder so six people never collide:

```bash
cp notebooks/week2_annotate_cv_eval/03_implement_iou.ipynb \
   notebooks/week2_annotate_cv_eval/<your_name>/03_implement_iou.ipynb
```

## How a notebook is graded

Some cells are blanked for you to implement (a `# TODO` and a `raise NotImplementedError`).
Fill them in; the **grader cell** right after imports the real library function and checks
your version against it. Green = correct. That is the whole loop: implement in a cell →
match the library → move on.

## Using AI on this project

AI is a great tutor and a poor ghostwriter. The rule:

- **Encouraged**, ask AI to explain a concept, look up syntax, or decode an error message.
- **Not for the graded blanks**, the formulas you implement (IoU, precision/recall/F1, the
  PR curve & AUC, matching) must be **typed by you**. That cell *is* the lesson, and the grader
  checks it.
- **The test**, *you must be able to walk a mentor through your own notebook, line by line.*

## Layout

```
notebooks/
  week1_foundations/         load a well, images-are-numbers, the assay, visualize
  week2_annotate_cv_eval/    annotate 5 images, classic-CV baseline, implement IoU + P/R/F1 + PR-curve & AUC
  week3_algorithm_hpc/       run the deep model, beat the baseline, the supercomputer, (tracking)
  week4_finish_present/      one-command tool, validation study (vs Agilent), figures, (distill, optional)
  solutions/                 the worked answer keys (mentors edit here; students don't)
```

## For mentors

Author the full, working notebook under `notebooks/solutions/<week>/` as a percent `.py`,
mark the student exercises (`# >>> exercise ... # <<< exercise`, or a `# %% [exercise]`
cell), then run:

```bash
pixi run build-notebooks            # regenerate student .py + all .ipynb twins from solutions/
pixi run build-notebooks --check    # CI: fail if any generated notebook is stale
```

That one command writes the student `.py`, the student `.ipynb`, and the solution `.ipynb`
(all output-stripped and deterministic). You only ever hand-edit the solution `.py`.

See [`tools/make_student_version.py`](../tools/make_student_version.py) for the marker
syntax.
