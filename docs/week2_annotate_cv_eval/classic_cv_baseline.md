# The classic-CV baseline

This is the "old way", the same threshold → clean → watershed → size-filter recipe the
commercial tool uses. You build it, run it on a real well, and document where it breaks.
The same class is reused as the `mock` backend of the segmenter, so the whole pipeline runs
on Day 1 with no neural network.

!!! note "Agilent's classic mode, and its new AI mode"
    This baseline mirrors Agilent eSight's **rule-based** analysis. In 2026 Agilent added an
    **AI** module; the Week-3 deep model (and the Week-4
    [head-to-head](../week4_finish_present/compare_to_agilent.md)) is what meets *that*.

## The traditional way (and where it breaks)

The commercial tool segments with **rule-based** computer vision:

1. **Threshold**, pick a brightness cutoff; pixels above it are "cell," below are
   "background."
2. **Morphological cleanup**, open/close to remove specks and fill holes.
3. **Watershed**, split touching cells using a distance transform.
4. **Size filter**, drop objects too small (debris) or too big (merged blobs).

It works on easy images and **fails** on ours in three ways you'll see for yourself:

- it needs the threshold **hand-tuned for every experiment**;
- it **merges** cells whose boundaries are faint (very common in brightfield);
- it counts the **plate scratch** as a cell.

That failure is the point, it's what the deep model in Week 3 has to beat, measured on the
metrics you implement next.

## Walk through the steps

```python
from autopallios.core.baseline import BaselineParams, BaselineSegmenter

params = BaselineParams(
    threshold_method="otsu", # try "adaptive" on uneven illumination
    min_object_area=50, # raise this to ignore debris...
    use_watershed=True, # ...but watch it merge weak-boundary neighbors anyway
)
seg = BaselineSegmenter(params)
labels = seg.segment(movie, channel_idx=0)   # (T, H, W, C) -> (T, H, W)
```

Each step maps to a method you can read:

- `_threshold`, `skimage.filters.threshold_otsu` (or `threshold_local`), with an automatic
  polarity check so dark-on-light and light-on-dark both work.
- `_clean`, `skimage.morphology` opening/closing + `scipy.ndimage.binary_fill_holes`.
- `_split`, `scipy.ndimage.distance_transform_edt` + `skimage.feature.peak_local_max` +
  `skimage.segmentation.watershed` to separate touching cells.
- `_size_filter`, drop objects outside the allowed area band and renumber.

## The experiment to run

1. Build and run the baseline on a well (notebook `02_classic_cv_baseline`).
2. Sweep `min_object_area` and the threshold method; record how the cell count changes.
3. Find the plate scratch and a pair of merged cells. These are your "failure cases", and
   in [the metrics lesson](implement_the_metrics.md) you'll *quantify* them with `count_bias`.

## The code behind this chapter

The full, type-hinted docstrings (rendered straight from the source, so they never drift)
live in the API reference:

- [`autopallios.core.baseline`](../reference/core.md), `BaselineSegmenter` and
  `BaselineParams`, the class you built above.
- [`autopallios.core.io`](../reference/core.md), `load`, `ImageSequence`, and
  `save_mask_as_tiff`.
