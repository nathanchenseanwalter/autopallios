# Annotate 5 images

> *"Simple annotation — do 5 images. Teach them how to annotate."*

A model is only as good as the ground truth you score it against, and that ground truth is
made by hand. This lesson is about **the format**, which matters more than the tool.

## What a hand label is

A hand annotation is a single `(H, W)` **integer** image: `0` is background, and `1, 2, 3,
…` are the cells you outlined. That is the *exact* contract the rest of the library already
consumes — the same `(T, H, W)` masks `SupervisedMetrics` and `iou_matrix` use. There is no
new format to invent: a label you draw drops straight into the scoring code.

`autopallios.data.annotations` is the validated load/save layer:

```python
from autopallios.data.annotations import save_annotation, load_annotation, validate_annotation

validate_annotation(my_label, image=my_frame)        # enforces the contract (raises if broken)
save_annotation(my_label, "data/gold/labels/well_f00.tif")
labels = load_annotation("data/gold/labels/well_f00.tif")
```

## Where the gold set lives

`/data/` is git-ignored, but `data/gold/` is a small **tracked** exception so a fresh clone
can score against the same gold labels:

```
data/gold/images/<name>.tif    # the raw frame
data/gold/labels/<name>.tif    # your hand label, (H, W) int
```

Keep them tiny — a handful of small crops. Load a pair with
`annotations.load_gold_pair("<name>")`.

## Annotating in your tool

!!! note "Pending the tool decision (see the Mentor Runbook)"
    The recommended tool is **napari** (pure-Python, in the `teach` env); Fiji ROI export is
    the zero-install fallback. The click-by-click steps and the recorded walkthrough go here
    once chosen. The workflow is the same regardless of tool: open a frame → paint each cell
    a different integer id → export a label image → save it with `save_annotation`.

## The deliverable

Five label files under `data/gold/labels/`, validated, ready to feed
[the metrics lesson](implement_the_metrics.md). Bio people help sanity-check that what you
called a cell really is one.
