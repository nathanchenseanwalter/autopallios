"""Week 3 · Beat the baseline — runnable-script twin of 02_beat_the_baseline.ipynb.

**Reading:** *The algorithm → the supercomputer* chapter.
**Deliverable:** the deep model scores **higher** than your Week-2 baseline on the *same*
metric you implemented.

This is the whole point of the deep model: not "it looks nicer," but a higher F1 on the
exact metric functions you wrote in Week 2. We score both models with
`SupervisedMetrics` against synthetic ground truth.
"""

from autopallios.core.baseline import BaselineSegmenter
from autopallios.core.segmenter import Segmenter
from autopallios.data import synthetic
from autopallios.modules.evaluation import SupervisedMetrics


def main() -> None:
    """Run this lesson end to end — the notebook, as an automatable script."""
    movie, truth = synthetic.make_movie_with_labels(
        n_frames=4, size=(160, 160), n_cells=18, motion="migration", with_scratch=True, seed=7
    )

    baseline = BaselineSegmenter().segment(movie, channel_idx=0)
    f1_baseline = SupervisedMetrics().evaluate(baseline, truth)["aggregate"]["mean_f1"].iloc[0]
    print(f"baseline F1 = {f1_baseline:.3f}  (your Week-2 number to beat)")

    # ## Score the deep model and compare
    #
    # (Runs only where the `dl` extra is installed; otherwise it prints how to get it.)
    try:
        deep = Segmenter(model="cellpose_sam").segment(movie, channel_idx=0)
        f1_deep = SupervisedMetrics().evaluate(deep, truth)["aggregate"]["mean_f1"].iloc[0]
        print(f"cellpose_sam F1 = {f1_deep:.3f}")
        if f1_deep > f1_baseline:
            print(f"the deep model beats the baseline by {f1_deep - f1_baseline:+.3f} F1")
        else:
            print("the deep model did not win here, try a different channel or scene")
    except ImportError:
        print("Install the deep-learning extra to run cellpose_sam (see 01_run_deep_model),")
        print("then the deep F1 should clear the baseline number above.")


if __name__ == "__main__":
    main()
