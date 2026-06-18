# Week 1 · Foundations & first look

> **Goal:** get fluent with images as data, meet the biology and the assay, and *see* a
> real well. No segmentation yet — just read the data clearly and set up your tools.

!!! note "Do the work — this week's notebooks"
    [`notebooks/week1_foundations/`](https://github.com/nathanchenseanwalter/autopallios/tree/main/notebooks/week1_foundations):
    `01_command_line_and_setup` · `02_images_are_numbers` · `03_the_biology_and_assay` ·
    `04_visualize_a_well`. **Your bar:** load one well, print its `(T, H, W, C)` shape, and
    plot its frames + a pixel-intensity histogram. Open them with `pixi run -e teach lab`.

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

## The biology & the assay

Our lab images living cells over time and asks quantitative questions: how many cells, how
big, how fast they move, how many die. Two assays:

- **Wound-healing / migration** (brightfield, 1 channel): cells crawl to close a "wound"
  gap; we measure the closure rate.
- **Live/Dead kill-curve** (fluorescence, 3 channels): **channel 0 is the "all cells" stain
  we segment on**; the others light up live vs. dead cells as a drug takes effect.

Every later step — segment, track, measure, validate — exists to turn those pixels into
those numbers.

## Visualize before you compute

The fastest way to understand a well is to look at it. `autopallios.viz` gives you
notebook-friendly helpers (matplotlib only):

```python
from autopallios import viz
from autopallios.data import synthetic

well = synthetic.make_scene("mock_migration", n_frames=6)
viz.montage(well)                    # every frame in a grid
```

A pixel-intensity **histogram** is the simplest "metric" before segmentation — it shows the
background level, how bright the cells are, and whether that drifts over time. That
intuition is exactly what the segmenter has to capture in Week 2.
