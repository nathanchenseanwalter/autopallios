"""Week 1 · Visualize a well — runnable-script twin of 04_visualize_a_well.ipynb.

**Reading:** *Foundations* chapter.
**Deliverable:** plot every frame of a well **and** its per-frame pixel-intensity
histograms. This is your Week-1 bar: load a well, show it, describe it, no segmentation yet.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autopallios import viz
from autopallios.data import synthetic

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    """Run this lesson end to end — the notebook, as an automatable script."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    well = synthetic.make_scene("mock_migration", n_frames=6, size=(180, 180))
    print("well shape (T, H, W, C):", well.shape)

    # ## Every frame at a glance
    #
    # `viz.montage` tiles the whole time-series so you can scan it in one look.
    viz.montage(well, max_cols=3)
    plt.savefig(OUTPUT_DIR / "04_visualize_a_well_fig1.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ## Pixel histograms, the well as a distribution
    #
    # A histogram of pixel brightness is the simplest "metric" before segmentation: it tells
    # you the background level, how bright the cells are, and whether that drifts over time.
    fig, ax = plt.subplots(figsize=(6, 4))
    for t in range(well.shape[0]):
        values = well[t, :, :, 0].ravel()
        ax.hist(values, bins=50, histtype="step", label=f"t={t}")
    ax.set_xlabel("pixel intensity")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.set_title("per-frame pixel-intensity histograms")
    ax.legend(fontsize=8)
    plt.savefig(OUTPUT_DIR / "04_visualize_a_well_fig2.png", dpi=150, bbox_inches="tight")
    plt.close()


# **Describe what you see:** where is the background peak? Where are the cell pixels? Does
# the distribution shift across frames (cells moving in / dying)? That intuition is what the
# segmenter has to capture in Week 2.


if __name__ == "__main__":
    main()
