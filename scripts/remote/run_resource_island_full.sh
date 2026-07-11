#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/thesis"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

mkdir -p outputs

.venv/bin/python run_resource_island_smoke.py \
  --steps 40000 \
  --n-seeds 20 \
  --final-window 1000 \
  --save-dir outputs/resource_island_full
