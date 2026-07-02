# Compare to Agilent's eSight AI (and beat it on reach)

> *"Compare to the new Agilent AI system the company gives us, the SOTA we have on hand,
> and, as a small extra at the very end, see if we can distill a model that runs on the HPC,
> since Agilent can't."*

The tool this whole project replaces is the **Agilent xCELLigence RTCA eSight**. Until 2026
its imaging analysis was rule-based, the same threshold → clean → watershed → size-filter
pipeline you rebuilt as the [classic-CV baseline](../week2_annotate_cv_eval/classic_cv_baseline.md).
On **July 1, 2026** Agilent launched an **AI analysis module** for eSight: one-click,
label-free cell segmentation that removes the manual thresholding and parameter tuning.

That makes it a genuine **state-of-the-art competitor**, not a strawman, so we hold it to
the *same* ruler instead of hand-waving.

## The one thing it can't do

The eSight AI module runs **only on the local workstation** attached to the instrument. It
cannot scale to a cluster. Our pipeline runs the *identical* `Segmenter(...).segment(...)`
call on a laptop **or** across a whole plate on SDSC Expanse (Week 3). That, plus open,
free, and reproducible, is the argument the poster makes when the accuracy is close.

## A fair, same-metric head-to-head

We can't run Agilent here, but we can load the label masks it **exports** and score them with
the exact `SupervisedMetrics` you wrote in Week 2, same wells, same F1/IoU, no home-field
advantage. Ingestion reuses the existing IO layer, so there's no new format to invent:

```python
from autopallios.data.external import load_agilent_masks
from autopallios.modules.evaluation import SupervisedMetrics

agilent = load_agilent_masks("data/agilent_export/E4")   # (T, H, W) int labels
agg = SupervisedMetrics().evaluate(agilent, truth)["aggregate"]
print(agg[["mean_f1", "mean_semantic_iou", "count_bias"]])
```

Until a real export is in hand, `make_agilent_like(truth)` fabricates a plausible stand-in so
`02_validation_study` runs today; a `# TODO` marks exactly where the real masks drop in. The
notebook builds a small leaderboard, our baseline, our deep model, and Agilent, ranked on
the same metric.

!!! note "This trims, it doesn't delete"
    This head-to-head takes the slot the blind-A/B export held in `02_validation_study`; the
    `BlindEvaluationExporter` stays in the library (tested, still available), we just teach
    the Agilent comparison in its place.

## Stretch (optional): distill an HPC model *from* Agilent

If Agilent's AI is strong, use it. Treat its exported masks as a **teacher**: pair each frame
with Agilent's mask, fine-tune an open model (Cellpose) on those pseudo-labels, and run *that*
model at plate scale on Expanse, reach Agilent's local-only tool doesn't have. The optional
`04_distill_for_hpc` notebook prototypes the data-prep (runs anywhere) and guards the GPU
training so it never crashes without `torch`; the real training goes on the cluster via
[`slurm/`](https://github.com/nathanchenseanwalter/autopallios/tree/main/slurm).

## The code behind this chapter

- [`autopallios.data.external`](https://github.com/nathanchenseanwalter/autopallios/blob/main/autopallios/data/external.py)
, `load_agilent_masks` (the real seam) and `make_agilent_like` (today's stand-in).
- [`autopallios.modules.evaluation`](../reference/modules.md), `SupervisedMetrics` and
  `UnsupervisedMetrics`, the same scorers used everywhere else.
