#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

HOST="${THEORY_SCOUT_OLLAMA_HOST:-uace@100.107.98.78}"
PORT="${THEORY_SCOUT_OLLAMA_TUNNEL_PORT:-11435}"
MODEL="${THEORY_SCOUT_LLM_MODEL:-llama3.2:3b}"
RETRY_SECONDS="${THEORY_SCOUT_RETRY_SECONDS:-300}"
NUM_THREAD="${THEORY_SCOUT_NUM_THREAD:-8}"
NUM_CTX="${THEORY_SCOUT_NUM_CTX:-8192}"
NUM_PREDICT="${THEORY_SCOUT_NUM_PREDICT:-1400}"
DIRECT_OLLAMA_URL="${THEORY_SCOUT_DIRECT_OLLAMA_URL:-}"
PYTHON_BIN="${THEORY_SCOUT_PYTHON:-}"

RECORDS="literature/foundation_papers.csv"
CARDS_DIR="literature/foundation_paper_cards"
TEXT_DIR="literature/text"
MANIFEST="literature/foundation_card_fill_manifest.json"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

mkdir -p "$CARDS_DIR" literature/logs

TUNNEL_PID=""

timestamp() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

cleanup_tunnel() {
  if [[ -n "$DIRECT_OLLAMA_URL" ]]; then
    return
  fi
  if [[ -n "$TUNNEL_PID" ]]; then
    kill "$TUNNEL_PID" 2>/dev/null || true
    wait "$TUNNEL_PID" 2>/dev/null || true
    TUNNEL_PID=""
  fi
}

trap cleanup_tunnel EXIT

card_status() {
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
cards = sorted(Path("literature/foundation_paper_cards").glob("*.md"))
done = 0
for path in cards:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "TODO" not in text:
        done += 1
print(f"{done}/{len(cards)} completed cards")
PY
}

foundation_target_count() {
  "$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path
path = Path("literature/foundation_papers.csv")
if not path.exists():
    print(0)
else:
    with path.open(newline="", encoding="utf-8") as handle:
        print(sum(1 for _ in csv.DictReader(handle)))
PY
}

all_cards_done() {
  "$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path
records = Path("literature/foundation_papers.csv")
cards_dir = Path("literature/foundation_paper_cards")
if not records.exists():
    raise SystemExit(1)
target = sum(1 for _ in csv.DictReader(records.open(newline="", encoding="utf-8")))
cards = sorted(cards_dir.glob("*.md"))
done = sum(1 for path in cards if "TODO" not in path.read_text(encoding="utf-8", errors="replace"))
raise SystemExit(0 if target > 0 and done >= target else 1)
PY
}

ollama_ready() {
  local base_url="${DIRECT_OLLAMA_URL:-http://127.0.0.1:${PORT}}"
  local url="${base_url}/api/tags"
  "$PYTHON_BIN" - "$url" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=8) as response:
        data = json.loads(response.read().decode("utf-8"))
except Exception:
    raise SystemExit(1)
models = data.get("models", [])
print("available_models=" + ",".join(str(model.get("name", "")) for model in models))
raise SystemExit(0)
PY
}

start_tunnel() {
  if [[ -n "$DIRECT_OLLAMA_URL" ]]; then
    echo "[$(timestamp)] using direct Ollama URL: ${DIRECT_OLLAMA_URL}"
    return
  fi
  cleanup_tunnel
  ssh -F /dev/null \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -N \
    -L "127.0.0.1:${PORT}:127.0.0.1:11434" \
    "$HOST" &
  TUNNEL_PID=$!
}

run_fill_once() {
  local base_url="${DIRECT_OLLAMA_URL:-http://127.0.0.1:${PORT}}"
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m tools.theory_scout.cli fill-cards \
    --records "$RECORDS" \
    --cards-dir "$CARDS_DIR" \
    --text-dir "$TEXT_DIR" \
    --ollama-url "$base_url" \
    --model "$MODEL" \
    --limit 100 \
    --num-ctx "$NUM_CTX" \
    --num-predict "$NUM_PREDICT" \
    --num-thread "$NUM_THREAD" \
    --out-manifest "$MANIFEST"
}

postprocess() {
  "$PYTHON_BIN" -m tools.theory_scout.cli foundation-papers
  "$PYTHON_BIN" -m tools.theory_scout.cli obligations
  "$PYTHON_BIN" -m tools.theory_scout.cli audit-obligations \
    --out-csv literature/obligation_audit.csv \
    --out-md literature/obligation_audit.md
}

echo "[$(timestamp)] starting foundation-card resumable fill"
echo "[$(timestamp)] target_records=$(foundation_target_count) cards_dir=$CARDS_DIR model=$MODEL host=$HOST port=$PORT"
echo "[$(timestamp)] $(card_status)"

while true; do
  if all_cards_done; then
    echo "[$(timestamp)] all foundation cards already complete; running postprocess"
    postprocess
    echo "[$(timestamp)] complete"
    exit 0
  fi

  echo "[$(timestamp)] opening/refreshing Ollama tunnel"
  start_tunnel
  sleep 5

  if ollama_ready; then
    echo "[$(timestamp)] Ollama ready; $(card_status)"
    run_fill_once
    status=$?
    echo "[$(timestamp)] fill exit status=$status; $(card_status)"
    cleanup_tunnel
    if [[ "$status" -eq 0 ]]; then
      if all_cards_done; then
        echo "[$(timestamp)] all foundation cards filled; running postprocess"
        postprocess
        echo "[$(timestamp)] complete"
        exit 0
      fi
    fi
  else
    echo "[$(timestamp)] Ollama/SSH tunnel not ready"
    cleanup_tunnel
  fi

  echo "[$(timestamp)] sleeping ${RETRY_SECONDS}s before retry"
  sleep "$RETRY_SECONDS"
done
