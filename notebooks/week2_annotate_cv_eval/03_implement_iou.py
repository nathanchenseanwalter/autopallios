r"""Week 2 · Implement IoU — runnable-script twin of 03_implement_iou.ipynb.

**Reading:** *Implement the metrics (IoU, precision/recall/F1)* chapter.
**Deliverable:** your own `iou_two_masks` function, proven correct against the library.

To score segmentation we first need one number for "how much do two blobs overlap?" ,
the **Intersection over Union**:

$$\mathrm{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{pixels in both}}{\text{pixels in either}}$$

IoU is 1.0 for a perfect overlap and 0.0 for no overlap. It is the building block of the
precision/recall/F1 you implement in the next notebook: two cells "match" when their IoU
is at least 0.5.

The whole library shares **one** IoU implementation
(`autopallios.modules._common.iou_matrix`). You are about to write your own from
scratch, then check it against that shared one, your first taste of "implement it in a
notebook, then trust the tested library version."
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autopallios import viz
from autopallios.core.baseline import BaselineSegmenter
from autopallios.data import synthetic
from autopallios.modules._common import iou_matrix
from autopallios.modules.evaluation import SupervisedMetrics

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


# ## Your turn
#
# Fill in the body below. `mask_a` and `mask_b` are boolean images (`True` where the cell
# is). You need the count of pixels that are in **both** (intersection) and in **either**
# (union), then their ratio, guarding against an empty union so you never divide by zero.
def iou_two_masks(mask_a, mask_b):
    """IoU of two single-object masks: |A ∩ B| / |A ∪ B|."""
    mask_a = np.asarray(mask_a) > 0
    mask_b = np.asarray(mask_b) > 0
    # TODO(you): intersection = pixels in BOTH; union = pixels in EITHER; return inter/union (0.0 if union is empty)
    raise NotImplementedError("Exercise: intersection = pixels in BOTH; union = pixels in EITHER; return inter/union (0.0 if union is empty)")


def main() -> None:
    """Run this lesson end to end — the notebook, as an automatable script."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    # ## Grader, do not edit
    #
    # This cell builds two overlapping squares and checks your `iou_two_masks` against the
    # library's `iou_matrix`. A green means your function is correct.
    a = np.zeros((20, 20), dtype=int)
    a[5:15, 5:15] = 1  # a 10x10 square (area 100)
    b = np.zeros((20, 20), dtype=int)
    b[8:18, 8:18] = 1  # another 10x10 square, shifted by (3, 3)

    mine = iou_two_masks(a, b)
    iou, _, _ = iou_matrix(a, b)  # the library, on the same two objects
    reference = float(iou[0, 0])

    assert abs(mine - reference) < 1e-9, f"got {mine:.4f}, the library says {reference:.4f}"
    print(f"your IoU matches the library: {mine:.3f}")

    # ## See it on real masks
    #
    # Now use the library's vectorized `iou_matrix` (it does what you just wrote, for *every*
    # pair of objects at once) inside `SupervisedMetrics`, and look at the picture: green cells
    # were found, red were missed (false negatives), blue were invented (false positives).
    movie, truth = synthetic.make_movie_with_labels(
        n_frames=3, size=(160, 160), n_cells=18, motion="migration", with_scratch=True, seed=7
    )
    pred = BaselineSegmenter().segment(movie, channel_idx=0)

    agg = SupervisedMetrics().evaluate(pred, truth)["aggregate"]
    print(agg[["mean_f1", "count_bias"]].round(3))

    viz.compare(pred, truth, frame=0)
    plt.savefig(OUTPUT_DIR / "03_implement_iou_fig1.png", dpi=150, bbox_inches="tight")
    plt.close()


# **Next:** in `04_precision_recall_f1` you turn matched/unmatched cells (TP/FP/FN) into
# precision, recall, and F1, the score the deep model in Week 3 must beat.


if __name__ == "__main__":
    main()
