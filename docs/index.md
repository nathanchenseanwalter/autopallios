# autopallios

**Automated cell segmentation, tracking & analysis for microscopy — and a textbook for
learning how it works.**

This site is two things at once:

1. **A launchpad** for the [OPALS @ UC San Diego](https://iem.ucsd.edu/programs/opals)
   internship — a guided, week-by-week path from "images are just numbers" to a real
   deep-learning pipeline running on a supercomputer.
2. **The documentation** for the `autopallios` library the cohort builds together.

## The problem we're solving

Our lab images living cells over time. To get science out — how many cells, how big, how
fast they move and divide — every image must be **segmented** (each cell outlined) and
**tracked** (followed frame to frame). The commercial tool we currently use does this
with classic computer vision that needs constant manual tuning, merges cells with weak
boundaries, mistakes plate scratches for cells, and runs slowly on a single licensed
machine.

`autopallios` is the open replacement: *import everything, press run, get the best
results — no per-image babysitting, no single-seat license.*

## Three-line quickstart

```bash
pixi install      # build the environment (no GPU needed)
pixi run demo     # run the whole pipeline on synthetic data — works on Day 1
```

## How to read this textbook

The chapters follow the four-week program. Each week you build a working piece of the
real tool, and each chapter links to the exact code module it explains.

| Week | You learn | You build | Chapter |
|---|---|---|---|
| 1 | Images as arrays; the biology & assay; visualization | load & visualize a real well | [Week 1](week1_foundations/index.md) |
| 2 | Annotation; classic CV; **you implement** IoU / precision / recall / F1 | the baseline scored by your own metrics | [Week 2](week2_annotate_cv_eval/index.md) |
| 3 | Deep-learning segmentation; HPC & Slurm (tracking optional) | a model that beats the baseline, run on Expanse | [Week 3](week3_algorithm_hpc/index.md) |
| 4 | Honest validation; packaging | the one-command tool + a validation study | [Week 4](week4_finish_present/index.md) |

Each week also ships a set of **notebooks** under `notebooks/<week>/` — that's where you do
the work (the library is the answer key). Open them with `pixi run -e teach lab`.

New here? Start with [Chapter 0 · Setup](chapter0_setup.md).
