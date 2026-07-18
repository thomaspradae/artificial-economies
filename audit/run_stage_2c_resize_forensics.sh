#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/freeze_config.sh"

STAGE2B_DIR="${STAGE2B_DIR:-$ATARI_DIR/audit_outputs/stage2_preprocess}"
OUT="${OUT:-$ATARI_DIR/audit_outputs/stage2c_resize}"
FIXTURE_DIR="$OUT/fixtures"
DEEPMIND_DIR_OUT="$OUT/deepmind_outputs"
PYTORCH_DIR_OUT="$OUT/pytorch_outputs"
CANDIDATE_DIR="$OUT/candidate_reports"
REPORT_OUT="$OUT/resize_forensics_report.txt"
RANKED_JSONL_OUT="$OUT/resize_forensics_ranked.jsonl"

mkdir -p "$OUT" "$FIXTURE_DIR" "$DEEPMIND_DIR_OUT" "$PYTORCH_DIR_OUT" "$CANDIDATE_DIR"

export AUDIT_DIR ATARI_DIR DM_DIR OUT PYTHON_BIN LUAJIT_BIN

echo "=== Stage 2c: build resize fixtures ==="
"$PYTHON_BIN" "$AUDIT_DIR/build_resize_fixtures.py" \
  --out-dir "$FIXTURE_DIR" \
  --stage2b-dir "$STAGE2B_DIR" \
  --max-atari 64

echo
echo "=== Stage 2c: DeepMind resize oracle run 1 ==="
(
  cd "$DM_DIR/dqn"
  RESIZE_FIXTURE_PATHS="$FIXTURE_DIR/fixture_paths.txt" \
  RESIZE_DEEPMIND_OUT="$DEEPMIND_DIR_OUT/run1.jsonl" \
  RESIZE_DEEPMIND_OUTPUT_DIR="$DEEPMIND_DIR_OUT/run1_arrays" \
  "$LUAJIT_BIN" "$AUDIT_DIR/deepmind/trace_resize_fixtures.lua"
)

echo
echo "=== Stage 2c: DeepMind resize oracle run 2 ==="
(
  cd "$DM_DIR/dqn"
  RESIZE_FIXTURE_PATHS="$FIXTURE_DIR/fixture_paths.txt" \
  RESIZE_DEEPMIND_OUT="$DEEPMIND_DIR_OUT/run2.jsonl" \
  RESIZE_DEEPMIND_OUTPUT_DIR="$DEEPMIND_DIR_OUT/run2_arrays" \
  "$LUAJIT_BIN" "$AUDIT_DIR/deepmind/trace_resize_fixtures.lua"
)

echo
echo "=== Stage 2c: Python resize candidates ==="
"$PYTHON_BIN" "$AUDIT_DIR/pytorch/trace_resize_fixtures.py" \
  --fixtures "$FIXTURE_DIR/fixtures.jsonl" \
  --out "$PYTORCH_DIR_OUT/pytorch_resize.jsonl" \
  --output-dir "$PYTORCH_DIR_OUT/arrays"

echo
echo "=== Stage 2c: diagnose ==="
set +e
"$PYTHON_BIN" "$AUDIT_DIR/diagnose_resize_mismatch.py" \
  --fixtures "$FIXTURE_DIR/fixtures.jsonl" \
  --deepmind "$DEEPMIND_DIR_OUT/run1.jsonl" \
  --deepmind-repeat "$DEEPMIND_DIR_OUT/run2.jsonl" \
  --pytorch "$PYTORCH_DIR_OUT/pytorch_resize.jsonl" \
  --report "$REPORT_OUT" \
  --ranked-jsonl "$RANKED_JSONL_OUT" \
  --write-matching-implementation "$AUDIT_DIR/pytorch/deepmind_preprocess.py" | tee "$CANDIDATE_DIR/diagnose_console.txt"
diagnose_status=${PIPESTATUS[0]}
set -e

if [[ "$diagnose_status" -eq 0 ]]; then
  echo
  echo "=== Stage 2c: exact clone found; rerun Stage 2 with clone ==="
  PYTORCH_PREPROCESS_SOURCE=deepmind_clone bash "$AUDIT_DIR/run_stage_2_preprocess.sh"
fi

echo
echo "=== Stage 2c: outputs ==="
echo "$FIXTURE_DIR"
echo "$DEEPMIND_DIR_OUT/run1.jsonl"
echo "$DEEPMIND_DIR_OUT/run2.jsonl"
echo "$PYTORCH_DIR_OUT/pytorch_resize.jsonl"
echo "$REPORT_OUT"
echo "$RANKED_JSONL_OUT"

echo
echo "=== Stage 2c: report head ==="
sed -n '1,120p' "$REPORT_OUT"

if [[ "$diagnose_status" -eq 0 ]]; then
  echo
  echo "MATCH"
else
  echo
  echo "NO EXACT CLONE"
fi
