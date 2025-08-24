#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   finetune_chemeleon.sh DATA_CSV TARGET_COL OUTDIR [EXTRA_CLI_ARGS]
# Example:
#   finetune_chemeleon.sh data/demo/chemeleon_demo.csv y runs/chemeleon_demo_gpu \
#     --epochs 10 --batch-size 64 --accelerator gpu --devices 1

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 DATA_CSV TARGET_COL OUTDIR [EXTRA_CLI_ARGS]" >&2
  exit 1
fi

DATA_CSV="$1"; shift
TARGET="$1"; shift
OUTDIR="$1"; shift

# Activate chemprop env
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  : # already in a conda env
else
  eval "$(conda shell.bash hook)"
  conda activate chemprop-v2
fi

chemprop train \
  -i "$DATA_CSV" \
  -o "$OUTDIR" \
  --target-columns "$TARGET" \
  --from-foundation CheMeleon \
  --warmup-epochs 0 \
  "$@"
