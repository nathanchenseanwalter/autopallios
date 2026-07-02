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
# # Week 2 · Annotate 5 images
#
# **Reading:** *Annotate 5 images* chapter.
# **Deliverable:** five hand-label masks saved in the gold format, ready to score against.
#
# The model can only be *scored* against cells a human called real. This notebook is about
# making that ground truth. The key idea is the **format**, not the drawing tool:
#
# > A hand label is a single `(H, W)` **integer** image: `0` is background, `1, 2, 3, ...`
# > are the cells you outlined. That is the *exact* same thing `SupervisedMetrics` and
# > `iou_matrix` consume, so your labels drop straight into the scoring code.
#
# `autopallios.data.annotations` is the validated load/save layer for these files.

# %%
import numpy as np

from autopallios import viz
from autopallios.data import synthetic
from autopallios.data.annotations import load_annotation, save_annotation, validate_annotation

# %% [markdown]
# ## What a label looks like
#
# Until the real well + tool are chosen (see the runbook), we use a synthetic frame so this
# notebook runs for everyone today. A label is just a labeled image, here, one integer per
# cell. `validate_annotation` enforces the contract (2D, integer, `0` = background).

# %%
movie, example_label = synthetic.make_movie_with_labels(
    n_frames=5, size=(160, 160), n_cells=14, motion="migration", with_scratch=True, seed=7
)
frame0_label = example_label[0]  # one (H, W) label image
print("label dtype:", frame0_label.dtype, "| shape:", frame0_label.shape)
print("cell ids in frame 0:", np.unique(frame0_label))  # 0 (background) + one id per cell

validate_annotation(frame0_label, image=movie[0])  # raises if it breaks the contract
viz.show_overlay(movie, example_label, frame=0, title="a hand label = colored cell outlines")

# %% [markdown]
# ## Annotating in your tool (recorded tutorial)
#
# > **TODO (pending tool decision, see the Mentor Runbook):** the click-by-click steps for
# > **napari** (recommended) or Fiji go here, with the recorded walkthrough link. The
# > workflow is the same regardless of tool: open a frame → paint each cell a different
# > integer id → export a label image → save it with the cell below.

# %% [markdown]
# ## Save your five labels
#
# Save each annotated frame as `data/gold/labels/<well>_f<NN>.tif`. (Here we save the five
# synthetic frames as a stand-in so you can see the files appear; you'll replace them with
# your real hand labels.) These feed directly into `04_precision_recall_f1`.

# %%
out_dir = "output/my_labels"  # your own folder; real gold labels live under data/gold/labels/
saved = []
for t in range(5):
    path = save_annotation(example_label[t], f"{out_dir}/well_f{t:02d}.tif")
    saved.append(path)
print(f"saved {len(saved)} labels:")
for p in saved:
    print("  ", p)

# Reload one to prove the round-trip and the contract hold.
reloaded = load_annotation(saved[0])
assert np.array_equal(reloaded, example_label[0].astype(np.int32))
print("labels saved and reload cleanly in the gold format")

# %% [markdown]
# **Next:** in `02_classic_cv_baseline` you build the segmenter, then in
# `03_implement_iou` / `04_precision_recall_f1` you score it against the labels you just
# made, the whole point of annotating.
