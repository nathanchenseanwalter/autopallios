# Week 4 · Make it usable & tell the story

> **Goal:** turn the pipeline into a one-command tool, validate it honestly against the
> commercial software, and build the poster and talk.

!!! note "Chapter status"
    Structured stub — outline + working code references + the validation snippets, to be
    expanded during the program.

## Outline

- **From notebook to tool** — a single command/config that takes a folder of images and
  outputs results + figures. Reproducibility (the editable install, fixed settings,
  per-run manifests).
- **A validation study** — how scientists prove a method works: agreement, where each
  method wins, failure cases, honest limitations.
- **The story** — problem → approach → results → impact (accurate, fast, free, accessible).

## The supervised benchmark (with ground truth)

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

### Cross-model consensus + blind scoring (the validation study)

Where two independent models agree, you can trust the result without a human label. And a
blind A/B export lets an expert score two models without knowing which is which:

```python
from autopallios.core.segmenter import Segmenter
from autopallios.modules.evaluation import UnsupervisedMetrics, BlindEvaluationExporter

baseline = Segmenter(model="baseline").segment(movie, channel_idx=0)
model_b  = Segmenter(model="mock").segment(movie, channel_idx=0)   # swap for "cellpose"

# (1) where do the two models agree?
consensus = UnsupervisedMetrics().cross_model_consensus_score(
    baseline, model_b, model_a="baseline", model_b="cellpose")
print(consensus["summary"])

# (2) export randomized, identity-masked overlays for blind human scoring
manifest = BlindEvaluationExporter("blind_eval", seed=0).export(
    movie, {"baseline": baseline, "cellpose": model_b})
# a held-back _KEY_do_not_open.csv un-blinds it afterwards
```

## Packaging

The whole thing is installable (`pip install -e .` / `pixi install`) and exposes a
one-command tool:

```bash
autopallios run data/samples --kind directory --pattern "*_E4_2x2_W.tif" --as-gray --out results/
```

## The code behind this chapter

See the full API in the reference:
[`autopallios.modules.evaluation`](../reference/modules.md) — `SupervisedMetrics`,
`UnsupervisedMetrics`, `BlindEvaluationExporter`, and the `METRIC_REGISTRY`.
