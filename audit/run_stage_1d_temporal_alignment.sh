#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/audit_config.sh"

OUT="${OUT:-$ATARI_DIR/audit_outputs/stage1d_fire}"
TRACE_STEPS="${STAGE1D_TRACE_STEPS:-24}"
FRAME_DUMP_STEPS="${FRAME_DUMP_STEPS:-20}"
ACTION_TAPE_MODE="${ACTION_TAPE_MODE:-index}"
ACTION_TAPE_SEQUENCE="${ACTION_TAPE_SEQUENCE:-1,1,1,1,0x20}"
TEMPORAL_ALIGNMENT_OUT="${TEMPORAL_ALIGNMENT_OUT:-$OUT/temporal_alignment.txt}"

export OUT TRACE_STEPS FRAME_DUMP_STEPS ACTION_TAPE_MODE ACTION_TAPE_SEQUENCE TEMPORAL_ALIGNMENT_OUT

mkdir -p "$OUT"
set +e
bash "$AUDIT_DIR/run_stage_1_env.sh" > "$OUT/stage1d_stage1.log" 2>&1
stage_status=$?
set -e

echo "=== stage 1 log ==="
cat "$OUT/stage1d_stage1.log"

echo
echo "=== env compare ==="
if [[ -f "$OUT/env_compare.txt" ]]; then
  cat "$OUT/env_compare.txt"
else
  echo "missing $OUT/env_compare.txt"
fi

echo
echo "=== temporal alignment ==="
"$PYTHON_BIN" "$AUDIT_DIR/diagnose_temporal_alignment.py" \
  --frames-dir "$OUT/frames" \
  --pytorch-jsonl "$OUT/pytorch_env.jsonl" \
  --deepmind-jsonl "$OUT/deepmind_env.jsonl" \
  --rom "$ROM" \
  --out "$TEMPORAL_ALIGNMENT_OUT"

echo
echo "=== frame files ==="
find "$OUT/frames" -maxdepth 1 -type f | sort | sed -n '1,220p'

exit "$stage_status"
