#!/bin/bash
# Feature benchmark, held-out evaluation and final model, on CPU.
# Submit from the repository root with the environment active:
#     sbatch slurm/workflow_cpu.slurm
# Edit the partition (and add --account if your cluster needs one).
#SBATCH --job-name=prism_workflow
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export JOBLIB_TEMP_FOLDER="${TMPDIR:-/tmp}/joblib_${SLURM_JOB_ID}"; mkdir -p "${JOBLIB_TEMP_FOLDER}"

python -m pytest -q
python scripts/benchmark_features.py
python scripts/holdout.py
python scripts/train_final.py
