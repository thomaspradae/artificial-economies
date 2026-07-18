#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/freeze_config.sh"

mkdir -p "$OUT" "$CANONICAL_TAPE_DIR"

DEEPMIND_PREPROCESS_OUT="$OUT/deepmind_preprocess.jsonl"
PYTORCH_PREPROCESS_OUT="$OUT/pytorch_preprocess.jsonl"
SOURCE_REPORT_OUT="$OUT/deepmind_preprocess_source.txt"
INTERMEDIATE_DIR="$OUT/deepmind_preprocess_intermediates"
VARIANT_REPORT_OUT="$OUT/preprocess_variants_ranked.txt"
VARIANT_JSONL_OUT="$OUT/preprocess_variants_ranked.jsonl"
STAGE2B_CONSOLE_OUT="$OUT/stage2b_console.txt"

export AUDIT_DIR ATARI_DIR DM_DIR ROM ENV_ID SEED TRACE_STEPS FRAME_SKIP OUT
export CANONICAL_TAPE_DIR PYTHON_BIN LUAJIT_BIN PYTORCH_RESIZE_INTERPOLATION
export DEEPMIND_PREPROCESS_OUT PYTORCH_PREPROCESS_OUT
export DEEPMIND_PREPROCESS_SOURCE_OUT="$SOURCE_REPORT_OUT"
export DEEPMIND_PREPROCESS_INTERMEDIATE_DIR="$INTERMEDIATE_DIR"
export DEEPMIND_PREPROCESS_DUMP_STEPS="${DEEPMIND_PREPROCESS_DUMP_STEPS:-3}"

echo "=== Stage 2b: rebuild frozen Stage 2 artifacts and DeepMind intermediates ==="
set +e
bash "$AUDIT_DIR/run_stage_2_preprocess.sh"
stage2_status=$?
set -e
echo "Stage 2 comparison status: $stage2_status"
if [[ ! -s "$DEEPMIND_PREPROCESS_OUT" ]]; then
  echo "missing DeepMind preprocess trace: $DEEPMIND_PREPROCESS_OUT" >&2
  exit 1
fi
if [[ ! -s "$CANONICAL_TAPE_DIR/transitions.jsonl" ]]; then
  echo "missing canonical frame tape: $CANONICAL_TAPE_DIR/transitions.jsonl" >&2
  exit 1
fi

echo
echo "=== Stage 2b: rank preprocessing variants ==="
set +e
"$PYTHON_BIN" "$AUDIT_DIR/pytorch/preprocess_variants.py" \
  --tape-dir "$CANONICAL_TAPE_DIR" \
  --deepmind "$DEEPMIND_PREPROCESS_OUT" \
  --intermediate-dir "$INTERMEDIATE_DIR" \
  --out "$VARIANT_REPORT_OUT" \
  --jsonl "$VARIANT_JSONL_OUT" \
  --write-matching-implementation "$AUDIT_DIR/pytorch/deepmind_preprocess.py" | tee "$STAGE2B_CONSOLE_OUT"
variant_status=${PIPESTATUS[0]}
set -e

echo
echo "=== Stage 2b: DeepMind source report ==="
cat "$SOURCE_REPORT_OUT"

echo
echo "=== Stage 2b: generated intermediate files ==="
find "$INTERMEDIATE_DIR" -maxdepth 1 -type f -printf "%f %s bytes\n" | sort

echo
echo "=== Stage 2b: top ranked variants ==="
sed -n '1,40p' "$VARIANT_REPORT_OUT"

echo
echo "=== Stage 2b: outputs ==="
echo "$SOURCE_REPORT_OUT"
echo "$INTERMEDIATE_DIR"
echo "$VARIANT_REPORT_OUT"
echo "$VARIANT_JSONL_OUT"
echo "$STAGE2B_CONSOLE_OUT"

if [[ "$variant_status" -eq 0 ]]; then
  echo
  echo "MATCH"
else
  echo
  echo "NO EXACT MATCH"
fi
