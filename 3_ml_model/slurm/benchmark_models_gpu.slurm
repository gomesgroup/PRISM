#!/bin/bash
# Learner-family benchmark, including the graph networks, on one GPU.
# Submit from the repository root with the environment active (torch installed):
#     sbatch slurm/benchmark_models_gpu.slurm
# Edit the partition (and add --account if your cluster needs one).
#SBATCH --job-name=prism_models
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=1
python -c "import torch, sys; print('CUDA:', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"
python scripts/benchmark_models.py
