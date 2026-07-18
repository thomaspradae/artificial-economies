#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

mkdir -p literature

"$PYTHON_BIN" -m tools.theory_scout.cli full \
  --env-file literature/secrets.env \
  --per-query "${THEORY_SCOUT_PER_QUERY:-5}" \
  --semantic-delay-seconds "${THEORY_SCOUT_SEMANTIC_DELAY_SECONDS:-1.1}" \
  --card-limit "${THEORY_SCOUT_CARD_LIMIT:-150}" \
  "$@"
