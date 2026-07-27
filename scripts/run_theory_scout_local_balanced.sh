#!/usr/bin/env bash
set -euo pipefail

MODEL="${THEORY_SCOUT_LLM_MODEL:-llama3.2:3b}"
OLLAMA_URL="${THEORY_SCOUT_OLLAMA_URL:-http://127.0.0.1:11434}"
PER_QUERY="${THEORY_SCOUT_PER_QUERY:-10}"
CARD_LIMIT="${THEORY_SCOUT_CARD_LIMIT:-250}"
TEXT_LIMIT="${THEORY_SCOUT_TEXT_LIMIT:-150}"
TEXT_PER_WORLD_LIMIT="${THEORY_SCOUT_TEXT_PER_WORLD_LIMIT:-30}"
FILL_LIMIT="${THEORY_SCOUT_FILL_LIMIT:-75}"
FILL_PER_WORLD_LIMIT="${THEORY_SCOUT_FILL_PER_WORLD_LIMIT:-15}"
OUT_DIR="${THEORY_SCOUT_OUT_DIR:-literature}"
NUM_THREAD="${THEORY_SCOUT_NUM_THREAD:-8}"
PYTHON_BIN="${THEORY_SCOUT_PYTHON:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

PYTHONUNBUFFERED=1 \
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-0}" \
OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}" \
"$PYTHON_BIN" -m tools.theory_scout.cli full \
  --env-file "${OUT_DIR}/secrets.env" \
  --per-query "$PER_QUERY" \
  --card-limit "$CARD_LIMIT" \
  --text-limit "$TEXT_LIMIT" \
  --text-per-world-limit "$TEXT_PER_WORLD_LIMIT" \
  --fill-limit "$FILL_LIMIT" \
  --fill-per-world-limit "$FILL_PER_WORLD_LIMIT" \
  --semantic-delay-seconds 1.1 \
  --resolve-pdfs \
  --download \
  --fill-cards \
  --ollama-url "$OLLAMA_URL" \
  --model "$MODEL" \
  --num-thread "$NUM_THREAD" \
  "$@"
