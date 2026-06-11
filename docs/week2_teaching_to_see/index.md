# Week 2 · Teaching a computer to see cells

> **Goal:** understand supervised learning and build a deep-learning segmenter
> (Cellpose-SAM / CellSAM) that beats the Week-1 baseline on weak-boundary cells — with
> real accuracy numbers.

!!! note "Chapter status"
    This is a structured stub. The Week-1 chapter is the fully-written exemplar; this one
    has the outline, the key idea, and working code references, to be expanded during the
    program.

## Outline

- **Supervised learning, told as a story** — training data, labels, a loss function, and
  why a *learned* model beats hand-tuned thresholds on faint boundaries.
- **Run two generalist models out of the box** — Cellpose-SAM and CellSAM, with no tuning.
  Compare them to the Week-1 baseline. The "wow" moment: they outline weak cells you
  couldn't threshold.
- **Decide whether to fine-tune**, and how a few good labels plus correction beats
  thousands of bad ones.

## The one place you plug in a real model

Swapping the mock baseline for a real network is a one-line change — pick a backend by
name. Every backend is lazily imported, so this only requires the model you actually use:

```python
from autopallios.core.segmenter import Segmenter

seg = Segmenter(model="cellpose")          # needs: pixi add --feature dl cellpose
labels = seg.segment(movie, channel_idx=0)
```

To add *your own* model, write one class with a `run(frame_gray) -> labels` method and a
`@register_backend("name")` line — that decorator is the entire extension point.

## How we score it (preview of Week 4)

Once you have ground-truth labels, use `SupervisedMetrics` for IoU and F1; see
[Week 4](../week4_make_it_usable/index.md).

## The code behind this chapter

See the full API in the reference: [`autopallios.core.segmenter`](../reference/core.md) —
`Segmenter`, the `SegmentationBackend` protocol, and the `register_backend` decorator.
