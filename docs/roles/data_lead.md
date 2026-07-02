# Role guide · Data Lead

> *"How we taught the computer what a cell looks like."*

**You own:** organizing the image dataset, hand-annotation / ground truth, train/val/test
splits, and data augmentation.

**Your code touch-points:**

- `autopallios/core/io.py`, loading directories, multipage TIFFs, and videos into
  `(T, H, W, C)`. Extend the well-id parsing in `autopallios/_utils.py` if filenames change.
- `autopallios/data/synthetic.py`, the synthetic generator with matching ground truth;
  great for sanity-checking metrics before real labels exist.
- `autopallios/data/annotations.py`, load/save/validate the hand-label gold masks under
  `data/gold/`; these feed `SupervisedMetrics` in `autopallios/modules/evaluation.py`.

**First task:** confirm the two data shapes (`(T,H,W,1)` and `(T,H,W,3)`) and the Live/Dead
channel mapping (a Week-1 warm-up), then in Week 2 hand-label 5 frames and save them as the
gold set with `annotations.save_annotation`.
