# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Week 2 · The PR curve, Average Precision, and ROC AUC
#
# **Reading:** *The PR curve, AP, and AUC* chapter.
# **Deliverable:** your own precision/recall at a threshold and your own ROC-AUC by a
# threshold sweep, both proven against the library, then the same curve drawn on real cells.
#
# In `04_precision_recall_f1` you scored **one** threshold: a mask was a cell or it wasn't.
# But a real detector gives every object a **confidence**, and the honest question is not
# "how good is it at one cutoff?" but **"how good is it across all cutoffs?"** That is the
# **precision-recall (PR) curve**, and its two one-number summaries: **Average Precision**
# (area under the PR curve) and **ROC AUC** (area under the true-positive-rate vs.
# false-positive-rate curve).
#
# We learn it first on **plain numbers, no images**, so the *formula* is the lesson, not
# the pixels. Then we point the exact same functions at your cells.

# %%
import numpy as np

# %% [markdown]
# ## Part A, on plain numbers (no images)
#
# Twelve objects. A detector gave each a confidence `score` in `[0, 1]`; `labels` is the
# truth (`1` = really a cell, `0` = not). Predict "positive" when `score >= threshold`.

# %%
scores = np.array([0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.55, 0.40, 0.35, 0.30, 0.20, 0.10])
labels = np.array([1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0])

# %% [markdown]
# ### Your turn 1, precision and recall at one threshold
#
# Count TP/FP/FN at the given threshold, then apply the Week-2 formulas:
#
# $$\text{precision} = \frac{TP}{TP+FP}\qquad \text{recall} = \frac{TP}{TP+FN}$$

# %%
def pr_at_threshold(scores, labels, threshold):
    """Return (precision, recall) when predicting positive at `score >= threshold`."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    predicted_positive = scores >= threshold
    tp = int(np.sum(predicted_positive & (labels == 1)))
    fp = int(np.sum(predicted_positive & (labels == 0)))
    fn = int(np.sum(~predicted_positive & (labels == 1)))
    # TODO(you): precision=tp/(tp+fp); recall=tp/(tp+fn), guard each zero denominator
    raise NotImplementedError("Exercise: precision=tp/(tp+fp); recall=tp/(tp+fn), guard each zero denominator")


# %% [markdown]
# #### Grader, a tiny hand-checkable case
#
# Scores `[0.9, 0.6, 0.6, 0.2]`, truth `[1, 0, 1, 0]`, threshold `0.5`: three are predicted
# positive (`0.9, 0.6, 0.6`); two are real (`0.9, 0.6`) → TP=2, FP=1, FN=0 →
# precision `2/3`, recall `1.0`.

# %%
p, r = pr_at_threshold([0.9, 0.6, 0.6, 0.2], [1, 0, 1, 0], 0.5)
assert abs(p - 2 / 3) < 1e-9 and abs(r - 1.0) < 1e-9, f"got precision={p}, recall={r}"
print(f"precision/recall at a threshold: P={p:.3f} R={r:.3f}")

# %% [markdown]
# ### Your turn 2, ROC AUC by a threshold sweep
#
# Slide the threshold from 1 down to 0. At each step compute the **true-positive rate**
# (`TP / P`, P = total positives) and **false-positive rate** (`FP / N`, N = total
# negatives), then take the **area** under the `(FPR, TPR)` curve by the trapezoid rule.
# A perfect ranking gives AUC = 1.0; random gives ≈ 0.5.

# %%
def roc_auc_manual(scores, labels, n_grid=1001):
    """ROC AUC by sweeping `n_grid` thresholds across [0, 1] and integrating."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    tpr_list, fpr_list = [], []
    for threshold in np.linspace(0.0, 1.0, n_grid):
        predicted_positive = scores >= threshold
        tp = int(np.sum(predicted_positive & (labels == 1)))
        fp = int(np.sum(predicted_positive & (labels == 0)))
        # TODO(you): tpr = tp/n_pos ; fpr = fp/n_neg (guard zero denominators)
        raise NotImplementedError("Exercise: tpr = tp/n_pos ; fpr = fp/n_neg (guard zero denominators)")
    # Sort by ascending FPR so the curve runs left→right for integration.
    fpr = np.array(fpr_list)[::-1]
    tpr = np.array(tpr_list)[::-1]
    # TODO(you): trapezoid area under (fpr, tpr): sum of dx * (y_left + y_right) / 2
    raise NotImplementedError("Exercise: trapezoid area under (fpr, tpr): sum of dx * (y_left + y_right) / 2")


# %% [markdown]
# #### Grader, match the library
#
# The library ships the tested versions (`roc_auc`, `average_precision`, `pr_curve`). Your
# swept AUC should land within a hair of the exact one.

# %%
from autopallios.modules.evaluation import average_precision, pr_curve, roc_auc

mine = roc_auc_manual(scores, labels)
theirs = roc_auc(scores, labels)
assert abs(mine - theirs) < 0.02, f"your AUC {mine:.3f} vs library {theirs:.3f}"
print(f"your ROC AUC matches the library: {mine:.3f} ≈ {theirs:.3f}")

# %% [markdown]
# The library also hands you the **whole curve** and the AP in one call:

# %%
curve = pr_curve(scores, labels)
print(curve.round(3).to_string(index=False))
print(f"\nAverage Precision = {average_precision(scores, labels):.3f}")

# %% [markdown]
# ## Part B, the same idea, now on your cells
#
# A segmenter finds objects; give each a **confidence** (here: its **solidity**, how compact
# and convex it is, real cells are round and solid; merged or streaky false positives are
# not) and a **label** (1 if it matches a true cell at IoU ≥ 0.5, else 0). Feed those two
# arrays to the *identical* functions and you get a PR curve that tells you: *if I only kept
# my most cell-shaped detections, how clean would they be?* (Solidity is one of the shape
# priors the library's `morphological_anomaly_rate` already uses.)

# %%
from skimage.measure import regionprops

from autopallios.core.baseline import BaselineSegmenter
from autopallios.data import synthetic
from autopallios.modules._common import iou_matrix

movie, truth = synthetic.make_movie_with_labels(
    n_frames=4, size=(160, 160), n_cells=16, motion="migration", with_scratch=True, n_debris=10, seed=7
)
pred = BaselineSegmenter().segment(movie, channel_idx=0)

det_scores, det_labels = [], []
for t in range(pred.shape[0]):
    iou, ids_pred, _ = iou_matrix(pred[t], truth[t])
    if ids_pred.size == 0:
        continue
    best_iou = iou.max(axis=1) if iou.size else np.zeros(ids_pred.size)
    solidity = {prop.label: prop.solidity for prop in regionprops(pred[t])}
    for k, pid in enumerate(ids_pred):
        det_scores.append(solidity[pid])  # solidity ∈ (0, 1]: compact/convex = cell-like
        det_labels.append(int(best_iou[k] >= 0.5))

det_scores = np.array(det_scores)
det_labels = np.array(det_labels)
print(f"{len(det_labels)} detections | real cells: {det_labels.sum()} | merged/streaky FP: {(det_labels == 0).sum()}")

# %%
if det_labels.min() != det_labels.max():  # need both classes for a curve
    import matplotlib.pyplot as plt

    dcurve = pr_curve(det_scores, det_labels)
    ap = average_precision(det_scores, det_labels)
    auc = roc_auc(det_scores, det_labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(dcurve["recall"], dcurve["precision"], marker="o")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"Baseline detections, AP={ap:.2f}, ROC-AUC={auc:.2f}")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    plt.show()
    print(f"AP={ap:.3f}  ROC-AUC={auc:.3f}")
else:
    print("All detections fell in one class, bump n_debris or the scene to get both.")

# %% [markdown]
# **Interpret it:** where the curve stays high then falls, solidity *ranks* real cells above
# the merged/streaky false positives, the same shape signal the morphological-anomaly metric
# uses, so a solidity cutoff would clean up the false positives behind the positive
# `count_bias` you saw in `04`. In Week 3 the deep model should push this AP up: it separates
# real cells from junk using full shape *and* texture, not one number.
