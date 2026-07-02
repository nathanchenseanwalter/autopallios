# The PR curve, Average Precision, and ROC AUC

> *"Precision/recall is one day's lesson. Then AUC and the PR curve, give them the formula
> and let them implement it in Python, on non-image data first, to understand it."*

In [the metrics lesson](implement_the_metrics.md) you scored **one** threshold: a mask was a
cell or it wasn't. But a real detector attaches a **confidence** to every object, and the
honest question becomes *"how good is it across all thresholds?"* That is the
**precision-recall (PR) curve**, summarized by two single numbers you'll implement here:
**Average Precision** (area under the PR curve) and **ROC AUC** (area under the
true-positive-rate vs. false-positive-rate curve).

## 1. Learn it on plain numbers first (no images)

The formula is the lesson, not the pixels, so we start on a toy list of confidence
`scores` in `[0, 1]` and binary `labels` (`1` = really a cell). Predict "positive" when
`score >= threshold`, and slide the threshold from high to low. Only once the numbers make
sense do we point the same functions at cells.

## 2. The formulas (given)

Recall the per-threshold pair, then the two areas:

$$\text{precision} = \frac{TP}{TP+FP}\qquad \text{recall} = \frac{TP}{TP+FN}$$

$$\text{AP} = \sum_k (R_k - R_{k-1})\,P_k \qquad
  \text{ROC AUC} = \int_0^1 \text{TPR}\;d(\text{FPR})$$

- **Average Precision** sweeps the threshold and sums the precision at each new recall step
  (the area under the PR curve).
- **ROC AUC** plots TPR (`TP/P`) against FPR (`FP/N`) and takes the area by the trapezoid
  rule. It equals the probability a random positive outscores a random negative: **1.0** =
  perfect ranking, **0.5** = chance, **0.0** = perfectly wrong.

## 3. Implement, then check against the library

In `05_pr_curve_and_auc` you write two things and a **grader cell** checks each against the
tested library copy:

1. `pr_at_threshold(scores, labels, threshold)`, precision & recall at one cutoff (verified
   on a tiny hand-checkable case).
2. `roc_auc_manual(scores, labels)`, a threshold sweep + trapezoid, asserted to match
   `roc_auc` from the library within a hair.

```python
from autopallios.modules.evaluation import pr_curve, average_precision, roc_auc
# ... your roc_auc_manual(scores, labels) must match roc_auc(scores, labels) ...
```

The library also hands you the whole curve (`pr_curve`) and the summary (`average_precision`)
in one call, the finished version of what you implemented.

## 4. Bring it back to cells

A segmenter finds objects; give each a **confidence** and a **label** (1 if it matches a true
cell at IoU ≥ 0.5, else 0), and the *same* functions draw a PR curve over your detections.
The notebook uses **solidity** (how compact/convex an object is) as the confidence, real
cells are round and solid; merged or streaky false positives are not. That's the same shape
signal the library's `morphological_anomaly_rate` uses, so a confidence cutoff cleans up
exactly the false positives behind the positive `count_bias` you found in the last lesson.

Write down this AP, in Week 3 the deep model should push it up, separating real cells from
junk with full shape *and* texture rather than a single number.

## The code behind this chapter

- [`autopallios.modules.evaluation`](../reference/modules.md), `pr_curve`,
  `average_precision`, `roc_auc` (the finished versions of what you implement), all registered
  in `METRIC_REGISTRY`.
- [`autopallios.modules`](../reference/modules.md), the shared `iou_matrix` / `greedy_match`
  primitives that turn detections into the TP/FP labels these curves consume.
