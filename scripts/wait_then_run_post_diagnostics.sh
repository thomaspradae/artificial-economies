#!/usr/bin/env bash
set -euo pipefail

cd "${THESIS_DIR:-$HOME/thesis}"
mkdir -p outputs/logs

LOG_PATH="${LOG_PATH:-outputs/logs/post_run_diagnostics_waiter.log}"
OLD1_HOST="${OLD1_HOST:-100.80.3.43}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"

timestamp() {
  date --iso-8601=seconds
}

local_public_goods_busy() {
  pgrep -af 'run_public_goods_group_size_relaunch|run_public_goods_group_size_sweep.py' \
    | grep -v wait_then_run_post_diagnostics \
    | grep -v grep >/dev/null
}

old1_pricing_busy() {
  local output
  if output="$(ssh -F /dev/null -o ConnectTimeout=8 "uace@${OLD1_HOST}" \
    "pgrep -af 'run_pricing_nfirm_n5_relaunch|run_multiseed.py .*--n-firms 5|run_exploitability.py .*--n-firms 5' | grep -v grep" 2>&1)"; then
    [[ -n "$output" ]]
    return
  fi
  # If old1 cannot be reached, treat it as busy rather than starting diagnostics early.
  echo "[$(timestamp)] old1 status check failed; treating as busy: ${output}" >> "$LOG_PATH"
  return 0
}

echo "[$(timestamp)] post-run diagnostics waiter started" >> "$LOG_PATH"
while true; do
  local_busy=0
  old1_busy=0
  if local_public_goods_busy; then
    local_busy=1
  fi
  if old1_pricing_busy; then
    old1_busy=1
  fi
  echo "[$(timestamp)] wait local_public_goods_busy=${local_busy} old1_pricing_busy=${old1_busy}" >> "$LOG_PATH"
  if [[ "$local_busy" -eq 0 && "$old1_busy" -eq 0 ]]; then
    break
  fi
  sleep "$SLEEP_SECONDS"
done

echo "[$(timestamp)] active sweeps clear; starting post-run diagnostics" >> "$LOG_PATH"
exec bash scripts/run_post_run_diagnostics.sh >> "$LOG_PATH" 2>&1
