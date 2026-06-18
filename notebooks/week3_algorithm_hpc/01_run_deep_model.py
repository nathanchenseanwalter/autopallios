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
# # Week 3 · Run the deep model
#
# **Reading:** *The algorithm → the supercomputer* chapter.
# **Deliverable:** run a deep-learning segmenter and see its masks next to the baseline's.
#
# The classic baseline needs a knob for everything. A modern generalist model learned cell
# shapes from huge datasets and needs none. Our recommended model is **Cellpose-SAM**
# (`cellpose_sam`, the `cpsam` generalist) — no fine-tuning required to get strong results.
#
# Swapping models is *one line*, thanks to the backend registry: `Segmenter(model="...")`.

# %%
import matplotlib.pyplot as plt

from autopallios import viz
from autopallios.core.segmenter import Segmenter
from autopallios.data import synthetic

print("available backends:", Segmenter.available_backends())
movie = synthetic.make_scene("mock_migration", n_frames=4, size=(160, 160))

# %% [markdown]
# ## The always-available baseline (Day-1 mock)

# %%
mock = Segmenter(model="mock").segment(movie, channel_idx=0)
viz.show_overlay(movie, mock, frame=0, title="mock baseline")
plt.show()

# %% [markdown]
# ## The deep model — one line to swap
#
# `cellpose_sam` needs the deep-learning extra (`pixi run -e teach …`, or
# `pixi add --feature dl cellpose`) and downloads its weights on first use. If it isn't
# installed, this cell tells you how to get it instead of crashing the notebook.

# %%
DEEP_MODEL = "cellpose_sam"
try:
    deep = Segmenter(model=DEEP_MODEL).segment(movie, channel_idx=0)
    viz.show_overlay(movie, deep, frame=0, title=DEEP_MODEL)
    plt.show()
    print(f"{DEEP_MODEL} found {int(deep[0].max())} objects in frame 0")
except ImportError as exc:
    print(f"[{DEEP_MODEL}] needs the deep-learning extra — install it, then re-run:")
    print("    pixi run -e teach lab        (or: pixi add --feature dl cellpose)")
    print("details:", exc)
