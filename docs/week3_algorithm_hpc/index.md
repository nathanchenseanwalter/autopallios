# Week 3 · The algorithm → the supercomputer

> **Goal:** run a deep-learning segmenter that beats your Week-2 baseline on your own
> metric, then scale the same job to SDSC Expanse via Slurm.

!!! note "Do the work, this week's notebooks"
    [`notebooks/week3_algorithm_hpc/`](https://github.com/nathanchenseanwalter/autopallios/tree/main/notebooks/week3_algorithm_hpc):
    `01_run_deep_model` · `02_beat_the_baseline` · `03_to_the_supercomputer` ·
    `04_tracking_advanced` (optional). **Your bar:** the deep model's F1 clears the
    baseline's, and the same job runs on Expanse.

## Part 1, the algorithm

### Supervised learning, told as a story

The baseline needs a knob hand-tuned for every experiment. A modern **generalist** model
learned cell shapes from huge labeled datasets, so it outlines faint, touching cells you
could never threshold, with no tuning. That's the "wow": training data + labels + a loss
function buys you robustness the rule-based pipeline can't have.

### Run the deep model, one line

Our recommended model is **Cellpose-SAM** (`cellpose_sam`, the `cpsam` generalist). Swapping
models is a single line, thanks to the backend registry:

```python
from autopallios.core.segmenter import Segmenter

seg = Segmenter(model="cellpose_sam")        # needs the dl extra: pixi run -e teach ...
labels = seg.segment(movie, channel_idx=0)
```

!!! note "Day-1 stays light on purpose"
    The registry **default is still `mock`** (the classic baseline) so `import autopallios`,
    the tests, and `pixi run demo` need no PyTorch and no downloads. `cellpose_sam` runs on
    CPU on a small crop for a first "wow" without an Expanse allocation; it just downloads
    its weights on first use.

### Beat the baseline, on *your* metric

This is the deliverable: not "it looks nicer," but a higher F1 on the exact
`SupervisedMetrics` you implemented in Week 2. `02_beat_the_baseline` scores both models
against the same ground truth and checks the deep model wins.

### A few good labels: minimal fine-tuning

If the generalist isn't quite right for your cells, fine-tune it on the **5 gold labels**
you made in Week 2, *a few good labels beat thousands of bad ones*. Feed the
`annotations.load_gold_pair(...)` pairs to Cellpose training, then load the result with
`Segmenter(model="cellpose", pretrained_model="<path>")` and re-score. (Confirm the
fine-tuning scope with Hua, see the [Mentor Runbook](../mentor_runbook.md).)

## Part 2, the supercomputer

One well on a laptop is fine; a whole plate × many timepoints with a deep model is not.
HPC turns that from days into minutes because the work is **embarrassingly parallel**, each
well is independent.

- **What a cluster is**, login vs. compute nodes, the shared filesystem, the Slurm
  scheduler, GPUs. The ACCESS / SDSC Expanse picture.
- **Batch at scale**, a Slurm *job array* with one task per well. See the template:
  [`slurm/segment_array.sbatch`](https://github.com/nathanchenseanwalter/autopallios/blob/main/slurm/segment_array.sbatch).
  This is why `debug=False` matters at scale: no per-frame TIFF dumps flooding the parallel
  filesystem.

```bash
pixi install                       # build the env on Expanse
sbatch slurm/segment_array.sbatch  # one array task per well; results land in scratch
squeue -u $USER                    # watch them run in parallel
```

The segmentation call is identical to your laptop's, the cluster just runs many copies of
it at once.

## The code behind this chapter

- [`autopallios.core.segmenter`](../reference/core.md), `Segmenter`, the
  `SegmentationBackend` protocol, and `register_backend`.
- [`autopallios.modules.evaluation`](../reference/modules.md), `SupervisedMetrics`, to prove
  the win.

Cell tracking is the optional sequel: [Tracking (advanced / optional)](tracking_advanced.md).
The validation study and poster come next in [Week 4](../week4_finish_present/index.md).
