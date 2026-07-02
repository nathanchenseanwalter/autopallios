# Implement the metrics (IoU, precision/recall/F1)

> *"The most interesting part: give them the formula and let them implement it in Python."*

This is the flagship lesson. The library already computes these scores, but a number you
didn't compute yourself teaches you nothing, so here you **un-finish** them: given the
formulas, you write them, check against the library, and interpret the result on your own
labels.

## 1. The picture: TP, FP, FN

Match each predicted cell to a true cell. Every object lands in one bucket:

- **TP** (true positive): a prediction that matched a true cell, *found it*.
- **FP** (false positive): a prediction with no match, *invented it* (debris, a scratch).
- **FN** (false negative): a true cell with no match, *missed it* (merged, too faint).

`autopallios.viz.compare(pred, true)` draws exactly this, green found, red missed, blue
invented, so you can *see* a false positive before you score it.

## 2. The formulas (given)

$$\mathrm{IoU}(A,B) = \frac{|A \cap B|}{|A \cup B|}\qquad
  \text{precision} = \frac{TP}{TP+FP}\qquad
  \text{recall} = \frac{TP}{TP+FN}\qquad
  F_1 = \frac{2\,TP}{2\,TP+FP+FN}$$

Two cells "match" when their IoU ≥ 0.5. At that threshold each cell matches at most one
partner, so simple **greedy** matching is provably optimal, which is why we teach it
instead of the heavier Hungarian algorithm.

## 3. Implement, then check against the library

In `03_implement_iou` you write IoU for two masks; in `04_precision_recall_f1` you write
the three scores from TP/FP/FN. Each notebook has a **grader cell** that imports the real
library function and asserts your version matches:

```python
from autopallios.modules._common import iou_matrix          # the one shared, tested copy
# ... your iou_two_masks(a, b) must equal iou_matrix(a, b) ...
```

The invariant the grader leans on is the same one the test suite checks
(`tests/test_evaluation.py`): score identical masks and you must get F1 = 1.0, `count_bias`
= 0. Green means correct.

## 4. Score the baseline, and read the bias

Run *your* metrics on the Week-2 baseline against *your* 5 labels. The aggregate reports a
signed `count_bias`:

- **positive** → the baseline **over-segments** (debris / the scratch counted as cells);
- **negative** → it **merges** cells (weak boundaries).

Which failure does your data show? Write down this F1, in Week 3 the deep model has to beat
it on this exact metric, and push `count_bias` toward 0. Once your cell-version matches the
library, you've earned the [notebook → library](../from_notebook_to_library.md) lesson:
stop maintaining your copy and import the tested one.

## The code behind this chapter

- [`autopallios.modules.evaluation`](../reference/modules.md), `SupervisedMetrics` (the
  finished version of what you implemented), `METRIC_REGISTRY`.
- [`autopallios.modules`](../reference/modules.md), the shared `iou_matrix` / `greedy_match`
  primitives.
