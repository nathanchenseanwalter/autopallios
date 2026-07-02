# Week 4 · Finish & present

> **Goal:** turn the pipeline into a one-command tool, validate it honestly against another
> method, and build the poster and talk.

!!! note "Do the work, this week's notebooks"
    [`notebooks/week4_finish_present/`](https://github.com/nathanchenseanwalter/autopallios/tree/main/notebooks/week4_finish_present):
    `01_one_command_tool` · `02_validation_study` · `03_make_the_figures` ·
    `04_distill_for_hpc` (optional). **Your bar:** a folder-in → results-out run, plus a
    validation figure and one poster figure.

## From notebook to tool

A single command takes a folder of images and outputs results, no per-image babysitting:

```bash
autopallios run data/samples --kind directory --pattern "*_E4_2x2_W.tif" --as-gray --out results/
```

Getting there from a pile of notebook cells is a real skill, see
[From notebook to library](../from_notebook_to_library.md). Reproducibility comes from the
editable install, fixed settings, and the per-run `manifest.json` each recipe writes (which
is what makes the numbers below credible).

## The supervised benchmark (with ground truth)

You already wrote these metrics in Week 2, here you *use* them at scale:

```python
from autopallios.modules.evaluation import SupervisedMetrics

result = SupervisedMetrics(iou_match_threshold=0.5).evaluate(pred_masks, true_masks)
result["per_frame"]   # semantic IoU, pixel Dice, instance F1, count error per frame
result["aggregate"]   # one-row summary incl. signed count_bias (over- vs under-segmenting)
```

## No-reference metrics (no ground truth)

Most real experiments are unlabeled, so we also report proxy-quality metrics:

```python
from autopallios.modules.evaluation import UnsupervisedMetrics

u = UnsupervisedMetrics()
u.temporal_consistency_score(measurements, id_column="track_id")   # stability over time
u.morphological_anomaly_rate(measurements)                          # debris/scratch leakage
```

### Cross-model consensus + the Agilent head-to-head (the validation study)

Where two independent models agree, you can trust the result without a human label. And to
place the tool against the state of the art, we score it and **Agilent's eSight AI** on the
*same* metric, see [Compare to Agilent](compare_to_agilent.md):

```python
from autopallios.core.segmenter import Segmenter
from autopallios.data.external import load_agilent_masks, make_agilent_like
from autopallios.modules.evaluation import SupervisedMetrics, UnsupervisedMetrics

baseline = Segmenter(model="baseline").segment(movie, channel_idx=0)
deep     = Segmenter(model="cellpose_sam").segment(movie, channel_idx=0)

# (1) where do our two models agree? (no labels needed)
consensus = UnsupervisedMetrics().cross_model_consensus_score(
    baseline, deep, model_a="baseline", model_b="cellpose_sam")
print(consensus["summary"])

# (2) same-metric head-to-head vs the commercial AI
agilent = make_agilent_like(true_masks)      # or load_agilent_masks("data/agilent_export/E4")
print(SupervisedMetrics().evaluate(agilent, true_masks)["aggregate"][["mean_f1", "count_bias"]])
```

## The story

Problem → approach → results → impact (accurate, fast, free, accessible). Pair each poster
figure with one sentence of interpretation, and put your validation numbers next to the
quantitative result.

## The code behind this chapter

[`autopallios.modules.evaluation`](../reference/modules.md), `SupervisedMetrics`,
`UnsupervisedMetrics`, `BlindEvaluationExporter`, and the `METRIC_REGISTRY`.
