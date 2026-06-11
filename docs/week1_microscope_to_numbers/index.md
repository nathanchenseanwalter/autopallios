# Week 1 · From microscope to numbers

> **Goal:** become fluent with images as data, understand the biology and the problem,
> build the classic computer-vision baseline — and *feel* why it fails. That failure is
> what motivates machine learning in Week 2.

## Images are just numbers

A microscope image is a grid of numbers. A grayscale frame is a 2D array of shape
`(H, W)` — each number is how bright that pixel is. A color image adds a third axis for
channels: `(H, W, C)`, e.g. `C=3` for red/green/blue. A movie adds time on the front:
`(T, H, W, C)`.

`autopallios` standardizes **everything** to that 4D shape `(T, H, W, C)`, with grayscale
keeping a channel axis of size 1. This one rule removes a whole class of "is this 2D or
3D right now?" bugs. The contract is written down once, here:

::: autopallios._typing

## Our two kinds of data

The lab gives us two very different files, and the loader makes them look the same:

- **Brightfield fibroblasts** (`.tif`): one file per timepoint per well, so a *time series
  is a sorted folder of single-frame TIFFs* → `(T, H, W, 1)`.
- **Live/Dead fluorescence** (`.avi`): one video file *is* 53 frames → `(T, H, W, 3)`.

You load either with the same call:

```python
from autopallios.core import io

# a folder of brightfield frames for one well:
movie = io.load("data/samples", kind="directory", pattern="*_E4_2x2_W.tif", as_gray=True)
print(movie.shape)   # (T, H, W, 1)
```

## The traditional way (and where it breaks)

The commercial tool segments with **rule-based** computer vision:

1. **Threshold** — pick a brightness cutoff; pixels above it are "cell," below are
   "background."
2. **Morphological cleanup** — erode/dilate to remove specks and fill holes.
3. **Watershed** — split touching cells using a distance transform.
4. **Size filter** — drop objects too small (debris) or too big (merged blobs).

It works on easy images and **fails** on ours in three ways you'll see for yourself:

- it needs the threshold **hand-tuned for every experiment**;
- it **merges** cells whose boundaries are faint (very common in brightfield);
- it counts the **plate scratch** as a cell.

You build exactly this pipeline in [The classic-CV baseline](classic_cv_baseline.md), then
run it and document the failures — your Week-1 deliverable.

## Try it now

```python
from autopallios.core.baseline import BaselineSegmenter
from autopallios.data import synthetic

movie = synthetic.make_cell_movie(n_frames=4, channels=1, with_scratch=True, n_debris=8)
labels = BaselineSegmenter().segment(movie, channel_idx=0)
print("objects in frame 0:", labels[0].max())
```

Change the `BaselineParams` (threshold method, sizes) and watch the object count swing —
that sensitivity *is* the lesson.
