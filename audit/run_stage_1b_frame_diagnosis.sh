#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/audit_config.sh"

FRAME_DUMP_DIR="${FRAME_DUMP_DIR:-$OUT/frames}"
FRAME_DUMP_STEPS="${FRAME_DUMP_STEPS:-20}"
FRAME_DIAGNOSIS_OUT="${FRAME_DIAGNOSIS_OUT:-$OUT/frame_diagnosis.txt}"
export OUT FRAME_DUMP_DIR FRAME_DUMP_STEPS FRAME_DIAGNOSIS_OUT

echo "=== compare ==="
if [[ -f "$OUT/env_compare.txt" ]]; then
  cat "$OUT/env_compare.txt"
else
  echo "missing $OUT/env_compare.txt"
fi

echo
echo "=== frame diagnosis ==="
"$PYTHON_BIN" "$AUDIT_DIR/diagnose_frame_mismatch.py" \
  --frames-dir "$FRAME_DUMP_DIR" \
  --max-step "$FRAME_DUMP_STEPS" \
  --out "$FRAME_DIAGNOSIS_OUT"

echo
echo "=== generated frame files ==="
if [[ -d "$FRAME_DUMP_DIR" ]]; then
  find "$FRAME_DUMP_DIR" -maxdepth 1 -type f | sort
else
  echo "missing $FRAME_DUMP_DIR"
fi
