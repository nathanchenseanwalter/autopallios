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
# # Week 1 · Images are just numbers
#
# **Reading:** *Foundations* chapter.
# **Deliverable:** open a frame, prove it's an array, and read a pixel value.
#
# A microscopy image is not a picture — it's a grid of numbers. In autopallios every image
# is a 4D array shaped **`(T, H, W, C)`**: time, height, width, channels (see
# `autopallios._typing`). A single grayscale frame is `(1, H, W, 1)`. Let's see that.

# %%
import numpy as np

from autopallios.data import synthetic

# A synthetic well so this runs for everyone today. (To use a real well later, load it
# with `autopallios.core.io.load(...)` or the `autopallios run` CLI — same array shape.)
movie = synthetic.make_scene("mock_migration", n_frames=6, size=(160, 160))
print("type:", type(movie).__name__)
print("shape (T, H, W, C):", movie.shape)
print("dtype:", movie.dtype, "| min:", int(movie.min()), "max:", int(movie.max()))

# %% [markdown]
# ## A frame is a 2D array; a pixel is one number
#
# Slice out frame 0, channel 0. It's an `(H, W)` array. Index a single pixel — it's just an
# integer brightness.

# %%
frame = movie[0, :, :, 0]  # (H, W)
print("one frame shape:", frame.shape)
print("pixel at (80, 80):", int(frame[80, 80]))
print("brightest pixel value:", int(frame.max()))

# A 5x5 corner of raw numbers — this is all an image ever is.
print("top-left 5x5 block:\n", frame[:5, :5])

# %% [markdown]
# ## Draw the numbers
#
# `imshow` maps those numbers to brightness. Same data, shown as a picture.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(frame, cmap="gray")
ax.set_title("frame 0 — the same numbers, drawn")
ax.axis("off")
plt.show()
