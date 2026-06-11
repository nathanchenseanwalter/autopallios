# Role guide · Data Lead

> *"How we taught the computer what a cell looks like."*

**You own:** organizing the image dataset, hand-annotation / ground truth, train/val/test
splits, and data augmentation.

**Your code touch-points:**

- `autopallios/core/io.py` — loading directories, multipage TIFFs, and videos into
  `(T, H, W, C)`. Extend the well-id parsing in `autopallios/_utils.py` if filenames change.
- `autopallios/data/synthetic.py` — the synthetic generator with matching ground truth;
  great for sanity-checking metrics before real labels exist.
- Ground-truth label masks feed `SupervisedMetrics` in `autopallios/modules/evaluation.py`.

**First task:** load one real brightfield well and one Live/Dead AVI, confirm the shapes
(`(T,H,W,1)` and `(T,H,W,3)`), and verify the Live/Dead channel mapping before the
validation study relies on it.
