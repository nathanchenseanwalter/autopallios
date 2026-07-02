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
# # Week 2 · Precision, recall, and F1
#
# **Reading:** *Implement the metrics (IoU, precision/recall/F1)* chapter.
# **Deliverable:** your own precision/recall/F1, proven against the library, used to score
# the baseline on your labels.
#
# In `03_implement_iou` you matched predicted cells to true cells (IoU ≥ 0.5). That splits
# every object into one of three buckets:
#
# - **TP** (true positive): a predicted cell that matched a true cell, *found it*.
# - **FP** (false positive): a predicted cell with no match, *invented it* (debris, a scratch).
# - **FN** (false negative): a true cell with no match, *missed it* (merged, too faint).
#
# From those three counts come the three scores:
#
# $$\text{precision} = \frac{TP}{TP+FP}\qquad \text{recall} = \frac{TP}{TP+FN}\qquad
#   F_1 = \frac{2\,TP}{2\,TP+FP+FN}$$
#
# Precision asks "of what I called a cell, how much was real?"; recall asks "of the real
# cells, how many did I find?"; F1 balances the two.

# %%
import numpy as np

# %% [markdown]
# ## Your turn
#
# Implement the three formulas above. Guard every denominator: if there are no predictions
# at all, precision is 0.0; if there are no true cells, recall is 0.0; and F1 is 1.0 only
# when there is nothing to find *and* nothing was predicted (the library's convention).

# %%
def prf1(tp, fp, fn):
    """Return (precision, recall, f1) from the TP / FP / FN counts."""
    # >>> exercise: precision=tp/(tp+fp); recall=tp/(tp+fn); f1=2*tp/(2*tp+fp+fn), guard each zero denominator
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return precision, recall, f1
    # <<< exercise


# %% [markdown]
# ## Grader, do not edit
#
# We run the real baseline on a synthetic well, ask the library for the TP/FP/FN of one
# frame, and check that *your* formulas reproduce the library's precision/recall/F1.

# %%
from autopallios.core.baseline import BaselineSegmenter
from autopallios.data import synthetic
from autopallios.modules.evaluation import SupervisedMetrics

movie, truth = synthetic.make_movie_with_labels(
    n_frames=3, size=(160, 160), n_cells=18, motion="migration", with_scratch=True, seed=7
)
pred = BaselineSegmenter().segment(movie, channel_idx=0)

row = SupervisedMetrics()._frame_row(pred[0], truth[0], 0)
p, r, f = prf1(row["tp"], row["fp"], row["fn"])
assert abs(p - row["precision"]) < 1e-9, f"precision: got {p}, library {row['precision']}"
assert abs(r - row["recall"]) < 1e-9, f"recall: got {r}, library {row['recall']}"
assert abs(f - row["f1"]) < 1e-9, f"f1: got {f}, library {row['f1']}"
print(f"your precision/recall/F1 match the library: P={p:.2f} R={r:.2f} F1={f:.2f}")

# %% [markdown]
# ## Score the baseline on your data, and read the bias
#
# `SupervisedMetrics().evaluate` runs your formulas (the library's identical copy) over
# every frame and reports a signed `count_bias`: **positive** = the baseline over-segments
# (debris/scratch counted as cells), **negative** = it merges cells (weak boundaries).

# %%
from autopallios import viz

agg = SupervisedMetrics().evaluate(pred, truth)["aggregate"]
print(agg[["mean_f1", "count_bias", "mean_abs_count_error"]].round(3))

viz.compare(pred, truth, frame=0)

# %% [markdown]
# **Interpret it:** is your `count_bias` positive or negative? Which failure is your
# baseline making? Write down this F1, in Week 3, the deep-learning model has to beat it
# on this exact metric, and push `count_bias` toward 0.
