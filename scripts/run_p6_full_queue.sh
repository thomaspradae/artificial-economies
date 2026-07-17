#!/usr/bin/env bash
set -euo pipefail

slot="${1:?usage: scripts/run_p6_full_queue.sh <old1|old2|ofi1>}"
cd "${THESIS_DIR:-$HOME/thesis}"

python_bin=".venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

mkdir -p outputs/p6_full_logs

run_job() {
  local name="$1"
  shift
  local save_dir="$1"
  shift
  local log="outputs/p6_full_logs/${name}.log"

  if [[ -s "${save_dir}/summary_aggregate.csv" && -s "${save_dir}/experiment_manifest.json" ]]; then
    echo "$(date -Is) SKIP ${name}: completed output exists" | tee -a "$log"
    return 0
  fi

  mkdir -p "$save_dir"
  echo "$(date -Is) START ${name}" | tee -a "$log"
  "$python_bin" "$@" --save-dir "$save_dir" 2>&1 | tee -a "$log"
  echo "$(date -Is) DONE ${name}" | tee -a "$log"
}

case "$slot" in
  old1)
    run_job auction_dqn outputs/auction_house_phase3_full/dqn \
      run_auction_house_smoke.py --mind dqn --steps 40000 --n-seeds 20 --final-window 5000 \
      --scenarios second_price first_price second_price_reserve clock second_price_public_signal second_price_noisy_signal
    run_job auction_ppo outputs/auction_house_phase3_full/ppo \
      run_auction_house_smoke.py --mind ppo --steps 40000 --n-seeds 20 --final-window 5000 \
      --scenarios second_price first_price second_price_reserve clock second_price_public_signal second_price_noisy_signal
    run_job labor_dqn outputs/labor_market_phase3_full/dqn \
      run_labor_market_smoke.py --mind dqn --steps 40000 --n-seeds 20 --final-window 5000
    run_job labor_ppo outputs/labor_market_phase3_full/ppo \
      run_labor_market_smoke.py --mind ppo --steps 40000 --n-seeds 20 --final-window 5000
    ;;
  old2)
    run_job auction_independent_dqn outputs/auction_house_phase3_full/independent_dqn \
      run_auction_house_smoke.py --mind independent_dqn --steps 40000 --n-seeds 20 --final-window 5000 \
      --scenarios second_price first_price second_price_reserve clock second_price_public_signal second_price_noisy_signal
    run_job auction_centralized_critic outputs/auction_house_phase3_full/centralized_critic \
      run_auction_house_smoke.py --mind centralized_critic --steps 40000 --n-seeds 20 --final-window 5000 \
      --scenarios second_price first_price second_price_reserve clock second_price_public_signal second_price_noisy_signal
    run_job labor_independent_dqn outputs/labor_market_phase3_full/independent_dqn \
      run_labor_market_smoke.py --mind independent_dqn --steps 40000 --n-seeds 20 --final-window 5000
    run_job labor_centralized_critic outputs/labor_market_phase3_full/centralized_critic \
      run_labor_market_smoke.py --mind centralized_critic --steps 40000 --n-seeds 20 --final-window 5000
    ;;
  ofi1)
    run_job public_goods_dqn outputs/public_goods_phase3_full/dqn \
      run_public_goods_smoke.py --mind dqn --steps 40000 --n-seeds 20 --final-window 5000 \
      --institutions none public_goods_penalty contribution_matching public_goods_reputation information_restriction tax_schedule
    run_job public_goods_ppo outputs/public_goods_phase3_full/ppo \
      run_public_goods_smoke.py --mind ppo --steps 40000 --n-seeds 20 --final-window 5000 \
      --institutions none public_goods_penalty contribution_matching public_goods_reputation information_restriction tax_schedule
    run_job public_goods_independent_dqn outputs/public_goods_phase3_full/independent_dqn \
      run_public_goods_smoke.py --mind independent_dqn --steps 40000 --n-seeds 20 --final-window 5000 \
      --institutions none public_goods_penalty contribution_matching public_goods_reputation information_restriction tax_schedule
    run_job public_goods_centralized_critic outputs/public_goods_phase3_full/centralized_critic \
      run_public_goods_smoke.py --mind centralized_critic --steps 40000 --n-seeds 20 --final-window 5000 \
      --institutions none public_goods_penalty contribution_matching public_goods_reputation information_restriction tax_schedule
    ;;
  *)
    echo "unknown slot: $slot" >&2
    exit 2
    ;;
esac
