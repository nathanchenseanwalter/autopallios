#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# autopallios — SDSC Expanse GPU job array
#
# One array task per well (the natural unit of parallelism: one well = one
# time-series). Runs the kill-curve recipe in PRODUCTION mode (--no-debug), so
# masks stay in memory and only final metrics/tables are written — no per-frame
# TIFF dumps flooding the parallel filesystem.
#
# Submit:   sbatch slurm/segment_array.sbatch
# Watch:    squeue -u $USER
# ---------------------------------------------------------------------------
#SBATCH --job-name=autopallios
#SBATCH --account=REPLACE_WITH_ACCESS_ALLOCATION   # your NSF ACCESS / SDSC project
#SBATCH --partition=gpu-shared                     # Expanse GPU partition
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=92G
#SBATCH --time=02:00:00
#SBATCH --array=0-29                               # one task per brightfield well (edit to match)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs results

module purge
module load gpu slurm                              # site modules

# --- Choose ONE way to get the environment -------------------------------
# Option A (recommended): the pixi 'gpu' environment (CUDA PyTorch, pinned).
eval "$(pixi shell-hook --environment gpu)"
# Option B (if the cluster mandates its own build): module load pytorch
#   and `pip install -e .` once into that module's environment.

# --- Map this array index to one well's data -----------------------------
mapfile -t WELLS < <(ls data/samples/*_W.avi | sort)
INPUT="${WELLS[$SLURM_ARRAY_TASK_ID]}"
echo "Task ${SLURM_ARRAY_TASK_ID}: processing ${INPUT}"

# --- Run the production recipe (debug=False) ------------------------------
python recipes/cancer_cell_death/run_kill_curve.py \
    --input "${INPUT}" \
    --channel 0 \
    --model cellpose \
    --out "results/${SLURM_ARRAY_TASK_ID}" \
    --no-debug
