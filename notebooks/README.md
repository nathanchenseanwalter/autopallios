# Notebooks, where you do the work

The library (`autopallios/`) is the finished tool. These notebooks are where *you* build
understanding: each week is a set of notebooks that runs top to bottom and prints a metric.

## Launch

```bash
pixi run -e teach lab        # opens JupyterLab on this folder
```

Every notebook comes as **two files that share a name**:

- **`.ipynb`** — the notebook you open and run in JupyterLab, straight from a fresh
  `git clone`. **This is where you do the work.**
- **`.py`** — a **runnable-script twin** of that notebook: the same steps, written the way
  you'd *automate* them. `python 03_implement_iou.py` runs the whole lesson in one shot
  (headless) and **saves its figures to `output/`** instead of drawing them inline. After
  you solve a notebook, paste your code into the matching blank in the `.py` and run it —
  that's the leap from *poking at cells* to a script you can schedule, drop in a recipe, or
  send to the supercomputer.

Both files are **generated** from one source of truth — the mentor's worked solution under
`notebooks/solutions/<week>/` (a percent-format `.py`). Don't hand-edit the generated files;
`pixi run build-notebooks` rebuilds them (see *For mentors*). **Commit both.**

### From notebook to script, by hand

Turning a notebook into an automatable script is a real skill — and the committed `.py`
twin is that, done for you. To watch it happen by hand, convert into a *scratch* file so you
never clobber the generated twin:

```bash
jupyter nbconvert --to script 03_implement_iou.ipynb --stdout    # notebook -> script text
jupytext --to py:percent      03_implement_iou.ipynb -o /tmp/nb.py  # the percent-cell form
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
match the library → move on. The `.py` twin carries the *same* blank — paste your finished
cell there and `python NN_name.py` runs the lesson end to end.

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

That one command writes the student **runnable-script** `.py`, the student `.ipynb`, and the
solution `.ipynb` (all output-stripped and deterministic). You only ever hand-edit the
solution `.py`. One authoring tip: **end a figure cell with `plt.show()`** — the notebook
renders it inline, and the script twin turns that line into a `savefig` into `output/`.

See [`tools/make_student_version.py`](../tools/make_student_version.py) for the marker
syntax and the notebook→script renderer (`percent_to_script`).
