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
# # Week 2 · The classic-CV baseline
#
# **Reading:** *The classic-CV baseline* chapter.
# **Deliverable:** run the rule-based segmenter on a well, *see* where it breaks, and
# capture its three failure modes.
#
# Before we teach a neural network anything, we build the "old way" — the same
# **threshold → clean → watershed → size-filter** pipeline the commercial tool uses. It
# has a knob for everything and it fails in instructive ways: it merges touching cells and
# it happily calls a plate scratch a cell. *That failure is the point* — it is what the
# deep model in Week 3 has to beat, measured on the metrics you write next.

# %%
import matplotlib.pyplot as plt

from autopallios import viz
from autopallios.core.baseline import BaselineSegmenter
from autopallios.data import synthetic
from autopallios.modules.evaluation import SupervisedMetrics

# %% [markdown]
# ## 1. Make a well with known answers
#
# We fabricate a migration-assay well: cells crawling into a wound band, plus some debris
# specks and a thin bright "scratch" line. Because it is synthetic, we also get the exact
# ground-truth labels for free — so later we can score the baseline honestly.

# %%
movie, truth = synthetic.make_movie_with_labels(
    n_frames=4,
    size=(160, 160),
    n_cells=18,
    motion="migration",
    with_scratch=True,
    n_debris=8,
    with_artifact_line=True,
    seed=7,
)
print("movie:", movie.shape, movie.dtype)  # (T, H, W, C)
print("truth:", truth.shape, truth.dtype)  # (T, H, W) int labels

# %% [markdown]
# ## 2. Run the baseline and look at it
#
# `BaselineSegmenter` is the classic pipeline. We segment on channel 0 (the "all cells"
# stain) and overlay the result on the raw frame.

# %%
baseline = BaselineSegmenter()
pred = baseline.segment(movie, channel_idx=0)

viz.show_overlay(movie, pred, frame=0, title="baseline masks (frame 0)")
plt.show()

viz.montage(movie, labels=pred, max_cols=4)
plt.show()

# %% [markdown]
# ## 3. Where does it break?
#
# Compare the predicted object count to the truth. The debris specks and the scratch line
# get counted as "cells," and weak-boundary neighbours get merged — the classic failures.

# %%
for t in range(movie.shape[0]):
    n_pred = int(pred[t].max())
    n_true = int(truth[t].max())
    print(f"frame {t}: baseline found {n_pred:3d} objects, truth has {n_true:3d}")

# %% [markdown]
# ## 4. Score it honestly (preview of the metrics lesson)
#
# `SupervisedMetrics` matches predicted cells to true cells and reports F1 and a signed
# `count_bias` (positive = over-segmenting, negative = merging). In the next notebook you
# implement these numbers yourself; here you just see what they say about the baseline.

# %%
result = SupervisedMetrics().evaluate(pred, truth)
print(result["aggregate"][["mean_f1", "count_bias", "mean_abs_count_error"]].round(3))

# The TP / FP / FN picture for one frame — green = found, red = missed, blue = invented.
viz.compare(pred, truth, frame=0)
plt.show()

# %% [markdown]
# **Capture for your write-up:** which failures do you see — over-segmentation (debris /
# scratch counted, `count_bias` > 0) or merging (weak boundaries, `count_bias` < 0)? Keep
# this baseline F1; in Week 3 the deep model has to beat it on the *same* metric.
