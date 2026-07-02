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
# **Deliverable:** an honest comparison, your tool vs. another method, with numbers.
#
# "It looks good" is not science. We validate three ways, each already built into the
# library (you do **not** write new metric code here, you *use* what's there):
#
# 1. **Supervised**, score against hand labels (IoU, F1) when you have ground truth.
# 2. **Unsupervised consensus**, where two independent models agree, you can trust the
#    result with *no* labels.
# 3. **Head-to-head vs. Agilent**, hold our tool and the commercial AI to the *same* metric,
#    on the same wells.

# %%
import pandas as pd

from autopallios.core.baseline import BaselineParams, BaselineSegmenter
from autopallios.data import synthetic
from autopallios.modules.evaluation import SupervisedMetrics, UnsupervisedMetrics

movie, truth = synthetic.make_movie_with_labels(
    n_frames=4, size=(160, 160), n_cells=16, motion="migration", with_scratch=True, seed=11
)

# Two of *our* models to compare (here: watershed on vs. off; in your study these are the
# Week-2 baseline and the Week-3 cellpose_sam). Agilent enters as the reference in section 3.
method_a = BaselineSegmenter(BaselineParams(use_watershed=True)).segment(movie, channel_idx=0)
method_b = BaselineSegmenter(BaselineParams(use_watershed=False)).segment(movie, channel_idx=0)

# %% [markdown]
# ## 1. Supervised, against ground truth

# %%
agg = SupervisedMetrics().evaluate(method_a, truth)["aggregate"]
print(agg[["mean_f1", "mean_semantic_iou", "count_bias"]].round(3))

# %% [markdown]
# ## 2. Unsupervised consensus, no labels needed

# %%
consensus = UnsupervisedMetrics().cross_model_consensus_score(
    method_a, method_b, model_a="watershed", model_b="no_watershed"
)
print(consensus["summary"][["consensus_score", "total_agree", "total_a_only", "total_b_only"]])

# %% [markdown]
# ## 3. Head-to-head vs. Agilent's eSight AI
#
# The tool we're replacing, **Agilent xCELLigence RTCA eSight AI**, is a strong one-click
# segmenter, but it runs **only on a local workstation**; it cannot scale to the
# supercomputer. We can still hold it to the *same* ruler. Drop the masks Agilent exports into
# `load_agilent_masks(...)`; until we have them, `make_agilent_like` fabricates a plausible
# stand-in so this runs today.

# %%
from autopallios.data.external import load_agilent_masks, make_agilent_like  # noqa: F401

# Real study: agilent = load_agilent_masks("data/agilent_export/<well>")
# TODO: point at a real Agilent export when available. Until then, a synthetic stand-in:
agilent = make_agilent_like(truth, seed=0)

rows = []
for name, masks in {
    "autopallios (baseline)": method_a,
    "autopallios (watershed off)": method_b,
    "Agilent eSight AI": agilent,
}.items():
    agg = SupervisedMetrics().evaluate(masks, truth)["aggregate"]
    rows.append(
        {
            "method": name,
            "mean_f1": round(float(agg["mean_f1"].iloc[0]), 3),
            "mean_semantic_iou": round(float(agg["mean_semantic_iou"].iloc[0]), 3),
            "count_bias": round(float(agg["count_bias"].iloc[0]), 3),
        }
    )
leaderboard = pd.DataFrame(rows).sort_values("mean_f1", ascending=False)
print(leaderboard.to_string(index=False))

# %% [markdown]
# **The differentiator isn't only the score.** Where methods tie on F1, ours still wins on
# *reach*: the identical `Segmenter(...).segment(...)` call runs on one laptop **or** across a
# whole plate on Expanse (`sbatch slurm/segment_array.sbatch`, Week 3), Agilent's AI is
# local-only. Open, free, reproducible, *and* HPC-scalable is the argument the poster makes.
# (Then see the optional `04_distill_for_hpc`: turn Agilent's own outputs into an HPC model.)

# %%
print("validated three ways, supervised, consensus, and a same-metric head-to-head vs Agilent")
