# For Mentors

A quick map of where the load-bearing pieces live, so you can point interns at the right
file and grade by artifact.

## Where students insert real things

| They want to... | They edit... |
|---|---|
| Use a real model (Cellpose/CellSAM) | `MODEL = "..."` knob in a recipe → `autopallios/core/segmenter.py` backend |
| Add a brand-new model | one `@register_backend("name")` class in `autopallios/core/segmenter.py` |
| Add a new metric | `METRIC_REGISTRY` in `autopallios/modules/evaluation.py` |
| Add a SOTA tracker | `Tracker._link_third_party` adapter in `autopallios/modules/tracking.py` |
| Point at real data | the `STUDENT KNOBS` block at the top of each recipe |

## Day-1 readiness

Everything runs on synthetic data with no GPU and no real files:

```bash
pixi install && pixi run demo && pixi run test
```

If those three pass, an intern is unblocked regardless of whether the SDSC allocation or
real data is ready yet.

## Grading by artifact (maps to the 4-week arc)

- **Week 1**: a working classic-CV baseline + documented failure cases (merged cells, the
  scratch counted as a cell).
- **Week 2**: a deep-learning segmenter that beats the baseline, with IoU/F1 numbers.
- **Week 3**: the pipeline running on Expanse over a full time-lapse, with tracking and
  scratch/debris rejection + a speed comparison.
- **Week 4**: the one-command tool, a validation study vs the commercial software, and the
  poster.

## Conventions that keep a cohort sane

See [CONTRIBUTING](https://github.com/opals-ucsd/autopallios/blob/main/CONTRIBUTING.md):
one-way dependencies (`core → modules → recipes`), the `(T,H,W,C)` array contract, lazy
heavy imports, per-student recipe folders, and a daily commit habit.
