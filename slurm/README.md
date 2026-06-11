# Running on SDSC Expanse (HPC)

This folder holds the Slurm batch template for running `autopallios` at scale on the
[SDSC Expanse](https://www.sdsc.edu/systems/expanse/) GPU cluster via NSF ACCESS.

## The mental model

- A **login node** is where you `ssh` in, edit files, and submit jobs. Do **not** run heavy
  work here.
- **Compute nodes** (with GPUs) actually run your job. You request them through Slurm.
- `sbatch` submits a job; `squeue -u $USER` shows its status; output lands in `logs/`.

## One-time setup

```bash
ssh <user>@login.expanse.sdsc.edu
git clone https://github.com/opals-ucsd/autopallios && cd autopallios
# install pixi (https://pixi.sh), then build the GPU environment:
pixi install -e gpu
```

## Submit the array

[`segment_array.sbatch`](segment_array.sbatch) runs **one task per well** — the natural
unit of parallelism, since one well is one time-series. Edit the `--account`, `--array`
range, and the data glob, then:

```bash
sbatch slurm/segment_array.sbatch
squeue -u $USER
```

## Why `--no-debug` on the cluster

In production we pass `--no-debug` so masks stay in memory and only the final metric tables
are written. Per-frame TIFF dumps (`debug=True`) are for *local* inspection in Fiji — on a
shared parallel filesystem they create thousands of tiny files and slow everyone down.

## If under-18 accounts aren't ready

Per the program plan, weeks 1–2 run fine on a laptop or Colab; treat Expanse as the
Week-3 scale-up demo, and fall back to a shared/mentor allocation if individual accounts
slip. The pipeline is identical — only the `--account` and where you launch it change.
