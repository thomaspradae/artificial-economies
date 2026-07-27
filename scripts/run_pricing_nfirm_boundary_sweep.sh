#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/pricing_nfirm_boundary outputs/logs

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

for n_firms in 2 3 4 5; do
  for mind in q_learning dqn ppo; do
    save_dir="outputs/pricing_nfirm_boundary/${mind}_n${n_firms}"
    log_path="outputs/logs/pricing_nfirm_boundary_${mind}_n${n_firms}.log"
    echo "=== boundary n_firms=${n_firms} mind=${mind} ===" | tee "${log_path}"
    .venv/bin/python run_multiseed.py \
      --steps 40000 \
      --n-seeds 20 \
      --final-window 1000 \
      --mechanisms none price_cap \
      --mind "${mind}" \
      --n-firms "${n_firms}" \
      --no-plots \
      --save-dir "${save_dir}" 2>&1 | tee -a "${log_path}"
  done
done
