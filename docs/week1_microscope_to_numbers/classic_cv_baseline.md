# The classic-CV baseline

This is the "old way" — the same threshold → clean → watershed → size-filter recipe the
commercial tool uses. You build it, run it on real fibroblasts, and document where it
breaks. The same class is reused as the `mock` backend of the segmenter, so the whole
pipeline runs on Day 1 with no neural network.

## Walk through the steps

```python
from autopallios.core.baseline import BaselineParams, BaselineSegmenter

params = BaselineParams(
    threshold_method="otsu",   # try "adaptive" on uneven illumination
    min_object_area=50,        # raise this to ignore debris...
    use_watershed=True,        # ...but watch it merge weak-boundary neighbors anyway
)
seg = BaselineSegmenter(params)
labels = seg.segment(movie, channel_idx=0)   # (T, H, W, C) -> (T, H, W)
```

Each step maps to a method you can read:

- `_threshold` — `skimage.filters.threshold_otsu` (or `threshold_local`), with an automatic
  polarity check so dark-on-light and light-on-dark both work.
- `_clean` — `skimage.morphology` opening/closing + `remove_small_objects/holes`.
- `_split` — `scipy.ndimage.distance_transform_edt` + `skimage.feature.peak_local_max` +
  `skimage.segmentation.watershed` to separate touching cells.
- `_size_filter` — drop objects outside the allowed area band and renumber.

## The experiment to run

1. Load one real brightfield well and segment it.
2. Sweep `min_object_area` and the threshold method; record how the cell count changes.
3. Find the plate scratch and a pair of merged cells; screenshot both. These are your
   "failure cases" — the motivation for Week 2.

## The code behind this chapter

The full, type-hinted docstrings (rendered straight from the source, so they never drift)
live in the API reference:

- [`autopallios.core.baseline`](../reference/core.md) — `BaselineSegmenter` and
  `BaselineParams`, the class you built above.
- [`autopallios.core.io`](../reference/core.md) — `load`, `ImageSequence`, and
  `save_mask_as_tiff`.
