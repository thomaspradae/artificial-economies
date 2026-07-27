#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p literature/logs

LOCK_PATH="${THEORY_SCOUT_FOUNDATION_LOCK:-literature/foundation_card_fill.lock}"
LOG_PATH="${THEORY_SCOUT_FOUNDATION_LOG:-literature/logs/foundation_card_fill_worker.log}"

export THEORY_SCOUT_DIRECT_OLLAMA_URL="${THEORY_SCOUT_DIRECT_OLLAMA_URL:-http://127.0.0.1:11434}"
export THEORY_SCOUT_LLM_MODEL="${THEORY_SCOUT_LLM_MODEL:-qwen3:8b}"
export THEORY_SCOUT_NUM_THREAD="${THEORY_SCOUT_NUM_THREAD:-8}"
export THEORY_SCOUT_NUM_CTX="${THEORY_SCOUT_NUM_CTX:-8192}"
export THEORY_SCOUT_NUM_PREDICT="${THEORY_SCOUT_NUM_PREDICT:-1400}"
export THEORY_SCOUT_RETRY_SECONDS="${THEORY_SCOUT_RETRY_SECONDS:-300}"

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_PATH"
  if ! flock -n 9; then
    echo "[$(date --iso-8601=seconds)] foundation-card worker already running; exiting" >> "$LOG_PATH"
    exit 0
  fi
fi

{
  echo "[$(date --iso-8601=seconds)] worker start host=$(hostname) model=${THEORY_SCOUT_LLM_MODEL}"
  scripts/run_foundation_cards_resumable_ofi1.sh
  echo "[$(date --iso-8601=seconds)] worker complete"
} >> "$LOG_PATH" 2>&1
