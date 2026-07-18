#!/usr/bin/env bash
set -euo pipefail

PORT="${THEORY_SCOUT_OLLAMA_TUNNEL_PORT:-11435}"
HOST="${THEORY_SCOUT_OLLAMA_HOST:-uace@100.107.98.78}"
MODEL="${THEORY_SCOUT_LLM_MODEL:-llama3.2:3b}"
LIMIT="${THEORY_SCOUT_CARD_LIMIT:-10}"
PYTHON_BIN="${THEORY_SCOUT_PYTHON:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

cleanup() {
  if [[ -n "${TUNNEL_PID:-}" ]]; then
    kill "$TUNNEL_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

ssh -F /dev/null \
  -o BatchMode=yes \
  -o StrictHostKeyChecking=accept-new \
  -o ExitOnForwardFailure=yes \
  -N \
  -L "127.0.0.1:${PORT}:127.0.0.1:11434" \
  "$HOST" &
TUNNEL_PID=$!

sleep 2

"$PYTHON_BIN" -m tools.theory_scout.cli fill-cards \
  --ollama-url "http://127.0.0.1:${PORT}" \
  --model "$MODEL" \
  --limit "$LIMIT" \
  --num-thread 8 \
  "$@"
