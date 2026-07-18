#!/usr/bin/env bash
set -euo pipefail

PORT="${THEORY_SCOUT_OLLAMA_TUNNEL_PORT:-11435}"
HOST="${THEORY_SCOUT_OLLAMA_HOST:-uace@100.107.98.78}"
MODEL="${THEORY_SCOUT_LLM_MODEL:-llama3.2:3b}"
PER_QUERY="${THEORY_SCOUT_PER_QUERY:-8}"
CARD_LIMIT="${THEORY_SCOUT_CARD_LIMIT:-150}"
TEXT_LIMIT="${THEORY_SCOUT_TEXT_LIMIT:-80}"
FILL_LIMIT="${THEORY_SCOUT_FILL_LIMIT:-40}"
OUT_DIR="${THEORY_SCOUT_OUT_DIR:-literature}"
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

"$PYTHON_BIN" -m tools.theory_scout.cli full \
  --env-file "${OUT_DIR}/secrets.env" \
  --per-query "$PER_QUERY" \
  --card-limit "$CARD_LIMIT" \
  --text-limit "$TEXT_LIMIT" \
  --fill-limit "$FILL_LIMIT" \
  --semantic-delay-seconds 1.1 \
  --resolve-pdfs \
  --download \
  --fill-cards \
  --ollama-url "http://127.0.0.1:${PORT}" \
  --model "$MODEL" \
  --num-thread 8 \
  "$@"
