# Chapter 0 · Setup

Goal: by the end of this page you have the project running on your machine and a green
check from the test suite — on Day 1, before you understand any internals.

## 1. Get the code

```bash
git clone https://github.com/opals-ucsd/autopallios
cd autopallios
```

## 2. Install the environment with pixi

We use [pixi](https://pixi.sh) so everyone — Mac, Windows, the Linux supercomputer — gets
the *same* environment from one file.

```bash
pixi install
```

This builds the `default` environment: Python 3.13 plus the lightweight scientific stack
(numpy, scipy, scikit-image, tifffile, imageio, pandas, matplotlib). It also installs
`autopallios` itself in *editable* mode, which is what lets you write
`from autopallios.core import segmenter` from any folder.

!!! note "No GPU, no heavy downloads yet"
    Deep-learning models (Cellpose, CellSAM) and trackers (trackpy, btrack) are **optional
    extras**. The default environment deliberately leaves them out so Day 1 is fast.

## 3. Run the demo

```bash
pixi run demo
```

This runs the wound-healing recipe on a **synthetic** cell movie — no real data needed. It
segments, tracks, and prints a Temporal Consistency Score, and writes a folder of mask
TIFFs you can open in Fiji/ImageJ.

## 4. Run the tests

```bash
pixi run test
```

All green? You're set up.

## 5. Make your own workspace

```bash
cp -r recipes/_template recipes/wound_healing/<your_name>
python recipes/wound_healing/<your_name>/run_experiment.py
```

Everything you produce lives under your folder's `output/`, so you and your teammates
never collide. See [CONTRIBUTING](https://github.com/opals-ucsd/autopallios/blob/main/CONTRIBUTING.md)
for the handful of rules that keep the shared repo tidy.

## Useful commands

| Command | What it does |
|---|---|
| `pixi run demo` | wound-healing recipe on synthetic data |
| `pixi run killcurve` | cancer cell-death recipe (supervised metrics) on synthetic data |
| `pixi run test` | the test suite |
| `pixi run lint` | ruff style + docstring checks |
| `pixi run docs` | serve this textbook locally |
