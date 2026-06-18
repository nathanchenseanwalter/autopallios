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
# # Week 1 · The biology & the assay
#
# **Reading:** *Foundations* chapter.
# **Deliverable:** explain, in your own words, what the assay measures — and find it in the
# channels.
#
# Our lab images living cells over time. To get science out — how many cells, how big, how
# fast they move, how many die — every frame must be **segmented** (each cell outlined) and
# **tracked** (followed over time). We work with two kinds of data:
#
# - **Brightfield** (the wound-healing / migration assay): grayscale, **1 channel**. Cells
#   crawl to close a "wound" gap; we measure how fast.
# - **Live/Dead fluorescence** (the kill-curve assay): **3 channels**. Channel 0 is the
#   "all cells" stain (what we segment on); the others light up live vs. dead cells.
#
# The shape tells you which is which: `(T, H, W, 1)` vs `(T, H, W, 3)`.

# %%
from autopallios.data import synthetic

brightfield = synthetic.make_scene("mock_migration", n_frames=6, size=(160, 160))   # 1-channel
livedead = synthetic.make_scene("mock_killcurve", n_frames=6, size=(160, 160))       # 3-channel
print("brightfield (migration):", brightfield.shape, "-> 1 channel")
print("live/dead  (kill curve):", livedead.shape, "-> 3 channels")

# %% [markdown]
# ## The channels of a Live/Dead frame
#
# Channel 0 = all cells (segment on this). Channel 1 = live signal. Channel 2 = dead signal.
# As cells die, channel 1 fades and channel 2 brightens — that's the whole kill-curve readout.

# %%
import matplotlib.pyplot as plt

last = livedead[-1]  # final frame, (H, W, 3)
titles = ["channel 0 — all cells (segment here)", "channel 1 — live", "channel 2 — dead"]
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
for ax, c, title in zip(axes, range(3), titles):
    ax.imshow(last[:, :, c], cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
plt.show()

# %% [markdown]
# **Write it down:** in one or two sentences, what does *your* assay measure, and which
# channel would you segment on? (For real Live/Dead AVIs, verify the channel order before
# trusting it — microscopes differ.)
