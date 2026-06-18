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
# # Week 3 · To the supercomputer
#
# **Reading:** *The algorithm → the supercomputer* chapter.
# **Deliverable:** the same job that ran on your laptop runs on SDSC Expanse via Slurm.
#
# One well on a laptop is fine. A whole plate × many timepoints with a deep model is not —
# that's what the **supercomputer** is for. The idea is **embarrassingly parallel**: each
# well is independent, so each becomes one task in a Slurm *array* and they all run at once.
#
# On Expanse you'd submit the batch script (no code change — same `autopallios`):
#
# ```bash
# pixi install                       # build the env on the cluster
# sbatch slurm/segment_array.sbatch  # one array task per well; results land in scratch
# squeue -u $USER                    # watch them run in parallel
# ```

# %%
import time

from autopallios.core.segmenter import Segmenter
from autopallios.data import synthetic

# Stand in for "a plate of wells" locally so you can see the batch shape.
wells = [synthetic.make_scene("mock_migration", n_frames=3, size=(120, 120), seed=s) for s in range(4)]
segmenter = Segmenter(model="mock")

start = time.perf_counter()
results = [segmenter.segment(well, channel_idx=0) for well in wells]
elapsed = time.perf_counter() - start

print(f"segmented {len(wells)} wells sequentially in {elapsed:.2f}s on this machine")
print("on Expanse: each well is one array task -> they finish in ~the time of a single well")
print("✅ same segmentation call; the cluster just runs many copies of it at once")
