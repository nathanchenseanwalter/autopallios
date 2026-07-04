"""Week 3 · Run the deep model — runnable-script twin of 01_run_deep_model.ipynb.

**Reading:** *The algorithm → the supercomputer* chapter.
**Deliverable:** run a deep-learning segmenter and see its masks next to the baseline's.

The classic baseline needs a knob for everything. A modern generalist model learned cell
shapes from huge datasets and needs none. Our recommended model is **Cellpose-SAM**
(`cellpose_sam`, the `cpsam` generalist), no fine-tuning required to get strong results.

Swapping models is *one line*, thanks to the backend registry: `Segmenter(model="...")`.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from autopallios import viz
from autopallios.core.segmenter import Segmenter
from autopallios.data import synthetic

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    """Run this lesson end to end — the notebook, as an automatable script."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("available backends:", Segmenter.available_backends())
    movie = synthetic.make_scene("mock_migration", n_frames=4, size=(160, 160))

    # ## The always-available baseline (Day-1 mock)
    mock = Segmenter(model="mock").segment(movie, channel_idx=0)
    viz.show_overlay(movie, mock, frame=0, title="mock baseline")
    plt.savefig(OUTPUT_DIR / "01_run_deep_model_fig1.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ## The deep model, one line to swap
    #
    # `cellpose_sam` needs the deep-learning extra (`pixi run -e teach ...`, or
    # `pixi add --feature dl cellpose`) and downloads its weights on first use. If it isn't
    # installed, this cell tells you how to get it instead of crashing the notebook.
    DEEP_MODEL = "cellpose_sam"
    try:
        deep = Segmenter(model=DEEP_MODEL).segment(movie, channel_idx=0)
        viz.show_overlay(movie, deep, frame=0, title=DEEP_MODEL)
        plt.savefig(OUTPUT_DIR / "01_run_deep_model_fig2.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"{DEEP_MODEL} found {int(deep[0].max())} objects in frame 0")
    except ImportError as exc:
        print(f"[{DEEP_MODEL}] needs the deep-learning extra, install it, then re-run:")
        print("    pixi run -e teach lab        (or: pixi add --feature dl cellpose)")
        print("details:", exc)


if __name__ == "__main__":
    main()
