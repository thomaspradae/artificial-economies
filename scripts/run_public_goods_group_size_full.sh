#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/logs

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

.venv/bin/python run_public_goods_group_size_sweep.py \
  --steps 40000 \
  --n-seeds 20 \
  --final-window 1000 \
  --agent-counts 2 4 8 16 \
  --minds q_learning dqn ppo independent_dqn centralized_critic \
  --institutions none contribution_matching public_goods_reputation tax_schedule \
  --save-dir outputs/public_goods_group_size_sweep_full
