# autopallios

**Automated cell segmentation, tracking & analysis for fluorescence + brightfield microscopy.**

Built by the [OPALS @ UC San Diego](https://iem.ucsd.edu/programs/opals) high-school
research internship. `autopallios` replaces slow, hand-tuned, single-license commercial
software (the Agilent xCELLigence RTCA eSight) with an open, automated, deep-learning
pipeline, and doubles as a teaching codebase for clean OOP, machine learning, and
high-performance computing.

Agilent added an AI segmentation module in 2026; autopallios's edge over it is **scale**,
the same pipeline runs on a laptop *or* the supercomputer, plus being open and free. Week 4
proves it with a same-metric head-to-head.

> **The product goal, in one sentence:** import everything, press run, get the best
> results, no per-image babysitting, no single-seat license.

## Quickstart

```bash
pixi install        # build the default environment (no GPU, no heavy downloads)
pixi run demo       # run the wound-healing recipe end-to-end on SYNTHETIC data
pixi run test       # run the test suite
pixi run docs       # open the textbook docs locally
```

`pixi run demo` works on Day 1 with **no real data and no GPU**, it fabricates a
synthetic cell movie, segments it, tracks cells, and reports a stability score.

In Python:

```python
from autopallios import Pipeline
result = Pipeline(model="mock").run("data/samples", kind="directory",
                                    pattern="*_E4_2x2_W.tif", as_gray=True)
print(result.labels.shape)   # (T, H, W) instance labels
```

Or from the command line (the one-command tool):

```bash
autopallios run movie.avi --model mock --channel 0 --out results/
```

## How it's organized

```
autopallios/        the installable library
  core/             load data · segment cells · reject artifacts   (io, baseline, segmenter, filter)
  modules/          track · measure intensity · evaluate           (tracking, intensity, evaluation)
  data/             synthetic movies + sample-data paths + external (Agilent) mask ingestion
recipes/            per-student experiment scripts (your workspace)
notebooks/          the weekly notebooks (where students work) + solution answer keys
docs/               the textbook (mkdocs), mirrors the 4-week program
slurm/              SDSC Expanse batch templates
tests/              CPU-only, mock-data tests
```

- **Heavy models are optional.** Cellpose, CellSAM, trackpy, btrack, PyTorch are opt-in
  extras, imported lazily. The baseline + the full mock pipeline run with only
  numpy/scipy/scikit-image/tifffile/imageio/pandas/matplotlib.
- **One data contract.** Every image is `(Time, Height, Width, Channels)`; grayscale is
  `C=1`. Every label mask is `(Time, Height, Width)` of integers. See
  [`autopallios/_typing.py`](autopallios/_typing.py).
- **`debug=True`** writes intermediate masks to disk for inspection in Fiji; **`debug=False`**
  keeps everything in memory for fast HPC runs.

## The 4-week arc (the docs follow this)

| Week | Theme | What we build |
|---|---|---|
| 1 | Foundations & first look | load & visualize a real well; the biology & the assay |
| 2 | Annotate → traditional CV → evaluate | the classic-CV baseline + the metrics you implement (IoU, P/R/F1) |
| 3 | The algorithm → the supercomputer | deep-learning segmentation (Cellpose-SAM) on SDSC Expanse via Slurm |
| 4 | Finish & present | one-command tool, validation study (incl. head-to-head vs Agilent's AI), poster |

## Team roles

Data Lead · Model Lead · HPC/Infra Lead · Viz/Comms Lead, see [`docs/roles/`](docs/).

## License

[MIT](LICENSE).
