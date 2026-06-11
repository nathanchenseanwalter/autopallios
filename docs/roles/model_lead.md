# Role guide · Model Lead

> *"How the model learns to outline cells."*

**You own:** the segmentation model — running Cellpose-SAM / CellSAM, fine-tuning, the
training loop, and accuracy metrics.

**Your code touch-points:**

- `autopallios/core/segmenter.py` — the backend registry and the plug-and-play menu. Swap
  the mock for a real model with `Segmenter(model="cellpose")`, or add your own with one
  `@register_backend("name")` class.
- `autopallios/core/baseline.py` — the classic-CV baseline you must beat.
- `autopallios/modules/evaluation.py` — `SupervisedMetrics` (IoU, F1) to prove it.

**First task:** run `Segmenter(model="baseline")` and `Segmenter(model="mock")` on a well
and compare object counts; then wire up one real backend behind the `dl` extra.
