# Role guide · Model Lead

> *"How the model learns to outline cells."*

**You own:** the segmentation model, running Cellpose-SAM / CellSAM, fine-tuning, the
training loop, and accuracy metrics.

**Your code touch-points:**

- `autopallios/core/segmenter.py`, the backend registry and the plug-and-play menu. Swap
  the mock for a real model with `Segmenter(model="cellpose")`, or add your own with one
  `@register_backend("name")` class.
- `autopallios/core/baseline.py`, the classic-CV baseline you must beat.
- `autopallios/modules/evaluation.py`, `SupervisedMetrics` (IoU, F1) to prove it, scored
  with the metric functions the cohort implements in Week 2.

**First task:** run `Segmenter(model="baseline")` on a well, then
`Segmenter(model="cellpose_sam")` (the recommended Week-3 generalist, behind the `dl`
extra) and compare F1 on the same labels.
