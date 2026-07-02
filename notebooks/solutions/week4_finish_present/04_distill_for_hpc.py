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
# # Week 4 · (Optional, bonus) Distill an HPC model from Agilent
#
# **Optional, above the bar** (like `04_tracking_advanced`). **Reading:** *Compare to
# Agilent* chapter, the "Stretch" section. **Deliverable:** the *idea*, prototyped, turn a
# strong-but-local model's outputs into training labels for a model you own and can run on the
# supercomputer.
#
# Agilent's eSight AI is accurate but **local-only**. The move: treat its exported masks as a
# **teacher**, use them as pseudo-labels to **fine-tune** an open model (Cellpose), and run
# *that* across a whole plate on Expanse, reach Agilent can't. (Any strong teacher works:
# Agilent, `cellpose_sam`, or the cross-model consensus from `02`.)

# %% [markdown]
# ## 1. Build the training set from the teacher's outputs (runs anywhere)
#
# Pair each raw frame with the teacher's mask. Here the teacher is a stand-in for Agilent's
# export; swap in `load_agilent_masks(...)` for the real thing.

# %%
from autopallios.data import synthetic
from autopallios.data.external import load_agilent_masks, make_agilent_like  # noqa: F401

movie, truth = synthetic.make_movie_with_labels(
    n_frames=4, size=(160, 160), n_cells=16, motion="migration", with_scratch=True, seed=11
)
# teacher = load_agilent_masks("data/agilent_export/<well>")   # <- real Agilent, when we have it
teacher = make_agilent_like(truth, seed=0)                     # <- stand-in for today

images = [movie[t, :, :, 0] for t in range(movie.shape[0])]    # raw frames  (H, W)
labels = [teacher[t] for t in range(teacher.shape[0])]         # teacher masks (H, W) int
print(f"prepared {len(images)} (image, teacher-label) training pairs")
print(f"example shapes, image {images[0].shape}, label {labels[0].shape}, ids {labels[0].max()}")

# %% [markdown]
# ## 2. Fine-tune an open model on those labels (GPU / HPC)
#
# Training needs the `dl` extra and a GPU, so it belongs on Expanse, we guard it here so the
# notebook never crashes without torch. Locally you can prototype on a crop; at scale you run
# it as a Slurm job.

# %%
try:
    from cellpose import models  # noqa: F401

    print("cellpose is available, on a GPU node you would now train, e.g.:")
    print("    model = models.CellposeModel(gpu=True)")
    print("    model.train(train_data=images, train_labels=labels, n_epochs=100, ...)")
    print("then load the distilled weights back through the same registry:")
    print("    Segmenter(model='cellpose', pretrained_model='runs/distilled.pt')")
except ImportError:
    print("[distill] training needs the deep-learning extra (torch + cellpose).")
    print("    pixi run -e teach lab        (or: pixi add --feature dl cellpose)")
    print("Run the real training on Expanse, see slurm/ and the Week-3 HPC lesson.")

# %% [markdown]
# ## 3. Run your distilled model at plate scale (where Agilent can't)
#
# Once trained it's the same one-line swap and the same Slurm array as Week 3, no new code:
#
# ```bash
# sbatch slurm/segment_array.sbatch   # with MODEL=cellpose, PRETRAINED=runs/distilled.pt
# squeue -u $USER
# ```

# %%
print("prototyped distillation: teacher masks → training pairs → an HPC-ready model")
