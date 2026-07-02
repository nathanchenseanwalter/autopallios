# For Mentors

A quick map of where the load-bearing pieces live, so you can point interns at the right
file and grade by artifact. For program logistics (pre-assessment, AI-usage policy,
escalation path, recordings), see the [Mentor Runbook](mentor_runbook.md).

!!! note "Six students, four roles"
    The cohort is **six** students across the four [role guides](roles/data_lead.md), pair up
    (e.g. two Data, two Model, one HPC, one Viz) so every role has an owner and nobody is a
    single point of failure.

## Where students insert real things

| They want to... | They edit... |
|---|---|
| Use a real model (Cellpose/CellSAM) | `MODEL = "..."` knob in a recipe → `autopallios/core/segmenter.py` backend |
| Add a brand-new model | one `@register_backend("name")` class in `autopallios/core/segmenter.py` |
| Add a new metric | `METRIC_REGISTRY` in `autopallios/modules/evaluation.py` |
| Add a SOTA tracker | `Tracker._link_third_party` adapter in `autopallios/modules/tracking.py` |
| Compare against another tool's exported masks (e.g. Agilent) | `load_agilent_masks` in `autopallios/data/external.py` |
| Point at real data | the `STUDENT KNOBS` block at the top of each recipe |

## Day-1 readiness

Everything runs on synthetic data with no GPU and no real files:

```bash
pixi install && pixi run demo && pixi run test
```

If those three pass, an intern is unblocked regardless of whether the SDSC allocation or
real data is ready yet.

## The minimal bar, by week

Each week's bar is simple and binary: the week's **notebook set runs top-to-bottom
("Restart & Run All", no errors) and prints the metric in the last column.** Polish and the
optional notebooks are upside, not the bar.

| Week | Notebook set | Minimal expectation | The number they show you |
|---|---|---|---|
| 1 · Foundations | `notebooks/week1_foundations/` | load a real well; print its `(T,H,W,C)` shape; plot frames + a pixel histogram | frame shape + a histogram figure |
| 2 · Annotate / CV / Eval | `notebooks/week2_annotate_cv_eval/` | 5 images annotated; baseline runs; **their own** IoU + precision/recall/F1 return numbers | baseline F1 + signed `count_bias` vs their labels |
| 3 · Algorithm / HPC | `notebooks/week3_algorithm_hpc/` | the deep model beats the baseline on their Week-2 metric; one job runs on Expanse | (deep F1 − baseline F1) > 0; an Expanse job |
| 4 · Finish / Present | `notebooks/week4_finish_present/` | folder-in → results-out; a validation comparison produced | consensus / Agilent-comparison result + a poster figure |

Tracking (`week3_algorithm_hpc/04_tracking_advanced`) is optional, above-bar, not required.

## Conventions that keep a cohort sane

See [CONTRIBUTING](https://github.com/nathanchenseanwalter/autopallios/blob/main/CONTRIBUTING.md):
one-way dependencies (`core → modules → recipes`), the `(T,H,W,C)` array contract, lazy
heavy imports, per-student recipe folders, and a daily commit habit.
