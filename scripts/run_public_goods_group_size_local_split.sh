#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/logs

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

run_chunk() {
  local n_agents="$1"
  local save_dir="outputs/public_goods_group_size_sweep_gashadokuro_n${n_agents}"
  .venv/bin/python run_public_goods_group_size_sweep.py \
    --steps 40000 \
    --n-seeds 20 \
    --final-window 1000 \
    --agent-counts "${n_agents}" \
    --minds q_learning dqn ppo independent_dqn centralized_critic \
    --institutions none contribution_matching public_goods_reputation tax_schedule \
    --save-dir "${save_dir}" \
    --resume
}

declare -a pids=()
read -r -a agent_counts <<< "${PUBLIC_GOODS_AGENT_COUNTS:-2 4 8 16}"
for n_agents in "${agent_counts[@]}"; do
  log_path="outputs/logs/public_goods_group_size_gashadokuro_n${n_agents}.log"
  echo "[launch] n_agents=${n_agents} log=${log_path}"
  run_chunk "${n_agents}" > "${log_path}" 2>&1 &
  pid="$!"
  echo "${pid}" > "outputs/public_goods_group_size_gashadokuro_n${n_agents}.pid"
  pids+=("${pid}")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
