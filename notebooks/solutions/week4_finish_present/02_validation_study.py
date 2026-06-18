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
# # Week 4 · The validation study
#
# **Reading:** *Finish & present* chapter.
# **Deliverable:** an honest comparison — your tool vs. another method — with numbers.
#
# "It looks good" is not science. We validate three ways, each already built into the
# library (you do **not** write new metric code here — you *use* what's there):
#
# 1. **Supervised** — score against hand labels (IoU, F1) when you have ground truth.
# 2. **Unsupervised consensus** — where two independent models agree, you can trust the
#    result with *no* labels.
# 3. **Blind A/B** — a human scores two methods without knowing which is which.

# %%
from autopallios.core.baseline import BaselineParams, BaselineSegmenter
from autopallios.data import synthetic
from autopallios.modules.evaluation import (
    BlindEvaluationExporter,
    SupervisedMetrics,
    UnsupervisedMetrics,
)

movie, truth = synthetic.make_movie_with_labels(
    n_frames=4, size=(160, 160), n_cells=16, motion="migration", with_scratch=True, seed=11
)

# Two genuinely different "methods" to compare (here: watershed on vs. off; in your study,
# model B is cellpose_sam and the reference is the commercial tool).
method_a = BaselineSegmenter(BaselineParams(use_watershed=True)).segment(movie, channel_idx=0)
method_b = BaselineSegmenter(BaselineParams(use_watershed=False)).segment(movie, channel_idx=0)

# %% [markdown]
# ## 1. Supervised — against ground truth

# %%
agg = SupervisedMetrics().evaluate(method_a, truth)["aggregate"]
print(agg[["mean_f1", "mean_semantic_iou", "count_bias"]].round(3))

# %% [markdown]
# ## 2. Unsupervised consensus — no labels needed

# %%
consensus = UnsupervisedMetrics().cross_model_consensus_score(
    method_a, method_b, model_a="watershed", model_b="no_watershed"
)
print(consensus["summary"][["consensus_score", "total_agree", "total_a_only", "total_b_only"]])

# %% [markdown]
# ## 3. Blind A/B — randomized, identity-hidden overlays for a human to score

# %%
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as out:
    manifest = BlindEvaluationExporter(out, seed=0).export(
        movie, {"watershed": method_a, "no_watershed": method_b}, frames=[0, 1]
    )
    n_png = len(list(Path(out).glob("*.png")))
    has_key = (Path(out) / "_KEY_do_not_open.csv").exists()
print(f"wrote {n_png} blind A/B panels; un-blinding key present: {has_key}")
print("✅ validated three ways — supervised, consensus, and blind A/B")
