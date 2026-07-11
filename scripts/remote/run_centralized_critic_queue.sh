#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/thesis"

while kill -0 "$(cat outputs/ppo.pid)" 2>/dev/null; do
  sleep 60
done

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

.venv/bin/python run_multiseed.py \
  --mind centralized_critic \
  --steps 40000 \
  --n-seeds 20 \
  --save-dir outputs/centralized_critic_v0_multiseed
