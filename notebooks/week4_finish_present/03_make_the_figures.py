"""Week 4 · Make the figures — runnable-script twin of 03_make_the_figures.ipynb.

**Reading:** *Finish & present* chapter.
**Deliverable:** the poster figures, a result-over-time curve, tracks, and an error map.

A poster is three or four figures that tell the story. We make them from one synthetic
migration well; swap in your real run and the code is unchanged.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from autopallios import viz
from autopallios.core.baseline import BaselineSegmenter
from autopallios.data import synthetic
from autopallios.modules import tracking

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    """Run this lesson end to end — the notebook, as an automatable script."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    movie, truth = synthetic.make_movie_with_labels(
        n_frames=8, size=(180, 180), n_cells=18, motion="migration", with_scratch=True, seed=13
    )
    pred = BaselineSegmenter().segment(movie, channel_idx=0)

    # ## Figure 1, cell count over time (the quantitative result)
    counts = [int(pred[t].max()) for t in range(pred.shape[0])]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(counts)), counts, marker="o")
    ax.set_xlabel("frame")
    ax.set_ylabel("cells detected")
    ax.set_title("detected cell count over time")
    plt.savefig(OUTPUT_DIR / "03_make_the_figures_fig1.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ## Figure 2, tracks (cells followed across frames)
    result = tracking.track(pred, max_distance=25)
    print("tracks found:", result.n_tracks)
    viz.plot_tracks(result)
    plt.savefig(OUTPUT_DIR / "03_make_the_figures_fig2.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ## Figure 3, where we're right and wrong (TP / FP / FN)
    viz.compare(pred, truth, frame=0)
    plt.savefig(OUTPUT_DIR / "03_make_the_figures_fig3.png", dpi=150, bbox_inches="tight")
    plt.close()


# **For the poster:** pair each figure with one sentence of interpretation, and put your
# validation numbers (Week-4 study) next to Figure 1. That's the story, from "images are
# numbers" to a measured, validated result.


if __name__ == "__main__":
    main()
