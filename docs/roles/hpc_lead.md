# Role guide · HPC / Infra Lead

> *"How we ran it on a supercomputer."*

**You own:** SDSC accounts, environment setup, Slurm batch jobs, reproducibility, and
speed benchmarks.

**Your code touch-points:**

- `pyproject.toml` (`[tool.pixi.*]`), the `gpu` environment (CUDA PyTorch, Linux-only) and the task shortcuts.
- `slurm/segment_array.sbatch`, the Expanse job array (one task per well). This is where
  `debug=False` matters: no per-frame TIFF dumps flooding the parallel filesystem.
- `autopallios/cli.py`, the one-command entry point your batch script calls.

**First task:** get `pixi install` working on Expanse, run the demo recipe on a login-node
test, then submit `slurm/segment_array.sbatch` as a small array and read the logs.
