"""Week 1 · The biology & the assay — runnable-script twin of 03_the_biology_and_assay.ipynb.

**Reading:** *Foundations* chapter.
**Deliverable:** explain, in your own words, what the assay measures, and find it in the
channels.

Our lab images living cells over time. To get science out, how many cells, how big, how
fast they move, how many die, every frame must be **segmented** (each cell outlined) and
**tracked** (followed over time). We work with two kinds of data:

- **Brightfield** (the wound-healing / migration assay): grayscale, **1 channel**. Cells
  crawl to close a "wound" gap; we measure how fast.
- **Live/Dead fluorescence** (the kill-curve assay): **3 channels**. Channel 0 is the
  "all cells" stain (what we segment on); the others light up live vs. dead cells.

The shape tells you which is which: `(T, H, W, 1)` vs `(T, H, W, 3)`.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from autopallios.data import synthetic

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    """Run this lesson end to end — the notebook, as an automatable script."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    brightfield = synthetic.make_scene("mock_migration", n_frames=6, size=(160, 160))   # 1-channel
    livedead = synthetic.make_scene("mock_killcurve", n_frames=6, size=(160, 160))       # 3-channel
    print("brightfield (migration):", brightfield.shape, "-> 1 channel")
    print("live/dead  (kill curve):", livedead.shape, "-> 3 channels")

    # ## The channels of a Live/Dead frame
    #
    # Channel 0 = all cells (segment on this). Channel 1 = live signal. Channel 2 = dead signal.
    # As cells die, channel 1 fades and channel 2 brightens, that's the whole kill-curve readout.
    last = livedead[-1]  # final frame, (H, W, 3)
    titles = ["channel 0, all cells (segment here)", "channel 1, live", "channel 2, dead"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, c, title in zip(axes, range(3), titles):
        ax.imshow(last[:, :, c], cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.savefig(OUTPUT_DIR / "03_the_biology_and_assay_fig1.png", dpi=150, bbox_inches="tight")
    plt.close()


# **Write it down:** in one or two sentences, what does *your* assay measure, and which
# channel would you segment on? (For real Live/Dead AVIs, verify the channel order before
# trusting it, microscopes differ.)


if __name__ == "__main__":
    main()
