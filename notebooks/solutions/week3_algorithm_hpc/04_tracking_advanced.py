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
# # Week 3 · Tracking (advanced / optional)
#
# **Reading:** *Tracking (advanced / optional)* chapter.
# **Deliverable (optional):** follow cells across time and compare matching strategies.
#
# In Week 2 you matched predicted cells to *true* cells in one frame (greedy IoU). Tracking
# is the same matching idea on a new axis: match a cell in frame *t* to the closest cell in
# frame *t−1* (nearest-centroid + a distance "gate"). This is the optional extension, strong
# students can compare greedy **nearest** matching to optimal **Hungarian** assignment by
# flipping one flag.

# %%
import matplotlib.pyplot as plt

from autopallios import viz
from autopallios.core.baseline import BaselineSegmenter
from autopallios.core.filter import ArtifactFilter
from autopallios.data import synthetic
from autopallios.modules import tracking

movie = synthetic.make_scene("mock_migration", n_frames=6, size=(160, 160))
pred = BaselineSegmenter().segment(movie, channel_idx=0)

# %% [markdown]
# ## Reject artifacts first (debris, the scratch line)

# %%
filtered, report = ArtifactFilter(min_area=30, max_aspect_ratio=8.0).apply(pred)
removed = int((~report["kept"]).sum())
print(f"artifact filter removed {removed} objects (debris / scratch fragments)")

# %% [markdown]
# ## Track: nearest vs. Hungarian
#
# Both use a distance gate; Hungarian (via SciPy, no new dependency) finds the globally
# optimal assignment instead of greedily taking the closest pair first.

# %%
near = tracking.track(filtered, max_distance=25, backend="nearest")
hung = tracking.track(filtered, max_distance=25, backend="hungarian")
print(f"tracks, nearest: {near.n_tracks} | hungarian: {hung.n_tracks}")

viz.plot_tracks(near)
plt.show()

# %% [markdown]
# **Compare:** do the two backends find the same number of tracks here? On crowded, fast
# movies Hungarian usually wins (greedy can "steal" a match). That trade-off, simple vs.
# optimal, is the lesson.
