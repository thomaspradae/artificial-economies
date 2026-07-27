#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/pricing_nfirm_boundary_old2_tail outputs/pricing_nfirm_exploitability_old2_tail outputs/logs

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

for mind in q_learning dqn ppo; do
  save_dir="outputs/pricing_nfirm_boundary_old2_tail/${mind}_n5"
  log_path="outputs/logs/pricing_nfirm_boundary_old2_tail_${mind}_n5.log"
  echo "=== old2 boundary n_firms=5 mind=${mind} ===" | tee "${log_path}"
  .venv/bin/python run_multiseed.py \
    --steps 40000 \
    --n-seeds 20 \
    --final-window 1000 \
    --mechanisms none price_cap \
    --mind "${mind}" \
    --n-firms 5 \
    --no-plots \
    --save-dir "${save_dir}" 2>&1 | tee -a "${log_path}"
done

for mind in q_learning dqn ppo; do
  save_dir="outputs/pricing_nfirm_exploitability_old2_tail/${mind}_n5"
  log_path="outputs/logs/pricing_nfirm_exploitability_old2_tail_${mind}_n5.log"
  echo "=== old2 exploitability n_firms=5 mind=${mind} ===" | tee "${log_path}"
  .venv/bin/python run_exploitability.py \
    --incumbent-steps 40000 \
    --adversary-steps 20000 \
    --evaluation-steps 5000 \
    --evaluation-burn-in 500 \
    --adversary-train-final-window 1000 \
    --n-seeds 20 \
    --adversary-restarts 3 \
    --mechanisms none price_cap \
    --incumbent-mind "${mind}" \
    --n-firms 5 \
    --adversary-index -1 \
    --no-plots \
    --save-dir "${save_dir}" 2>&1 | tee -a "${log_path}"
done
