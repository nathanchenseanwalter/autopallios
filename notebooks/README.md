# Notebooks — where you do the work

The library (`autopallios/`) is the finished tool. These notebooks are where *you* build
understanding: each week is a set of notebooks that runs top to bottom and prints a metric.

## Launch

```bash
pixi run -e teach lab        # opens JupyterLab on this folder
```

The notebooks are stored as percent-format `.py` files (clean diffs, code review). When
you open one in JupyterLab it pairs to an `.ipynb` automatically (jupytext); the `.ipynb`
is disposable and git-ignored. **Commit the `.py`.**

## Work in your own folder

Copy the notebook you're on into your own subfolder so six people never collide:

```bash
cp notebooks/week2_annotate_cv_eval/03_implement_iou.py \
   notebooks/week2_annotate_cv_eval/<your_name>/03_implement_iou.py
```

## How a notebook is graded

Some cells are blanked for you to implement (a `# TODO` and a `raise NotImplementedError`).
Fill them in; the **grader cell** right after imports the real library function and checks
your version against it. Green ✅ = correct. That is the whole loop: implement in a cell →
match the library → move on.

## Layout

```
notebooks/
  week1_foundations/         load a well, images-are-numbers, the assay, visualize
  week2_annotate_cv_eval/    annotate 5 images, classic-CV baseline, implement IoU + P/R/F1
  week3_algorithm_hpc/       run the deep model, beat the baseline, the supercomputer, (tracking)
  week4_finish_present/      one-command tool, validation study, figures
  solutions/                 the worked answer keys (mentors edit here; students don't)
```

## For mentors

Author the full, working notebook under `notebooks/solutions/<week>/`, mark the student
exercises (`# >>> exercise … # <<< exercise`, or a `# %% [exercise]` cell), then run:

```bash
pixi run build-notebooks            # regenerate the student copies from solutions/
pixi run build-notebooks --check    # CI: fail if a student copy is stale
```

See [`tools/make_student_version.py`](../tools/make_student_version.py) for the marker
syntax.
