#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/freeze_config.sh"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$ATARI_DIR/audit_outputs}"
STAGE3_REPLAY_DIR="${STAGE3_REPLAY_DIR:-$BASE_OUTPUT_DIR/stage3_replay}"
OUT="${OUT:-$BASE_OUTPUT_DIR/stage4_replay_sample}"
REPLAY_DIR="${STAGE4_REPLAY_DIR:-$OUT/canonical_replay}"

REQUESTED_INDICES="$OUT/requested_indices.txt"
VALID_INDICES="$OUT/valid_indices.txt"
REJECTED_INDICES="$OUT/rejected_indices.jsonl"
FIXTURE_VALIDATION_OUT="$OUT/fixture_validation.txt"
PYTORCH_REPLAY_SAMPLE_OUT="$OUT/pytorch_batch.jsonl"
DEEPMIND_REPLAY_SAMPLE_OUT="$OUT/deepmind_batch.jsonl"
BATCH_COMPARE_OUT="$OUT/batch_compare.txt"
SUMMARY_OUT="$OUT/STAGE4_SUMMARY.md"

mkdir -p "$OUT"

export AUDIT_DIR ATARI_DIR DM_DIR OUT PYTHON_BIN LUAJIT_BIN
export STAGE3_REPLAY_DIR STAGE4_REPLAY_DIR="$REPLAY_DIR"
export STAGE4_REQUESTED_INDICES="$REQUESTED_INDICES"
export DEEPMIND_REPLAY_SAMPLE_OUT

write_summary() {
  local status="$1"
  local reason="$2"
  {
    echo "# Stage 4 Replay Sampling Summary"
    echo
    echo "- Status: $status"
    echo "- Reason: $reason"
    echo "- Stage 3 replay source: $STAGE3_REPLAY_DIR"
    echo "- Canonical replay fixture: $REPLAY_DIR"
    echo "- Requested indices: $REQUESTED_INDICES"
    echo "- Valid indices: $VALID_INDICES"
    echo "- Rejected indices: $REJECTED_INDICES"
    echo "- Fixture validation: $FIXTURE_VALIDATION_OUT"
    echo "- PyTorch replay-sample trace: $PYTORCH_REPLAY_SAMPLE_OUT"
    echo "- DeepMind replay-sample trace: $DEEPMIND_REPLAY_SAMPLE_OUT"
    echo "- Batch compare: $BATCH_COMPARE_OUT"
    echo
    if [[ -f "$FIXTURE_VALIDATION_OUT" ]]; then
      echo "## Fixture Validation"
      echo
      echo '```text'
      sed -n '1,120p' "$FIXTURE_VALIDATION_OUT"
      echo '```'
      echo
    fi
    if [[ -f "$BATCH_COMPARE_OUT" ]]; then
      echo "## Batch Compare"
      echo
      echo '```text'
      sed -n '1,120p' "$BATCH_COMPARE_OUT"
      echo '```'
      echo
    fi
  } > "$SUMMARY_OUT"
}

echo "=== Stage 4: build canonical replay fixture ==="
"$PYTHON_BIN" "$AUDIT_DIR/build_stage4_replay_fixture.py" \
  --stage3-dir "$STAGE3_REPLAY_DIR" \
  --out-dir "$REPLAY_DIR"

echo
echo "=== Stage 4: generate explicit replay indices ==="
"$PYTHON_BIN" "$AUDIT_DIR/make_replay_indices.py" \
  --replay-dir "$REPLAY_DIR" \
  --requested-out "$REQUESTED_INDICES" \
  --valid-out "$VALID_INDICES" \
  --rejected-out "$REJECTED_INDICES"

echo
echo "=== Stage 4: validate replay fixture and index decisions ==="
"$PYTHON_BIN" "$AUDIT_DIR/validate_stage4_replay_fixture.py" \
  --replay-dir "$REPLAY_DIR" \
  --requested "$REQUESTED_INDICES" \
  --valid "$VALID_INDICES" \
  --rejected "$REJECTED_INDICES" \
  --report "$FIXTURE_VALIDATION_OUT"

echo
echo "=== Stage 4: PyTorch replay sample trace ==="
"$PYTHON_BIN" "$AUDIT_DIR/pytorch/trace_replay_sample.py" \
  --replay-dir "$REPLAY_DIR" \
  --requested "$REQUESTED_INDICES" \
  --out "$PYTORCH_REPLAY_SAMPLE_OUT"

echo
echo "=== Stage 4: DeepMind replay sample trace ==="
(
  cd "$DM_DIR/dqn"
  "$LUAJIT_BIN" "$AUDIT_DIR/deepmind/trace_replay_sample.lua"
)

echo
echo "=== Stage 4: compare replay sample semantics ==="
set +e
"$PYTHON_BIN" "$AUDIT_DIR/compare_jsonl.py" \
  --left "$PYTORCH_REPLAY_SAMPLE_OUT" \
  --right "$DEEPMIND_REPLAY_SAMPLE_OUT" \
  --left-label pytorch \
  --right-label deepmind \
  --ignore "source" \
  --float-tol "$FLOAT_TOL" | tee "$BATCH_COMPARE_OUT"
compare_status=${PIPESTATUS[0]}
set -e
if [[ "$compare_status" -ne 0 ]]; then
  write_summary "FAIL" "replay sample mismatch"
  sed -n '1,160p' "$SUMMARY_OUT"
  exit "$compare_status"
fi

write_summary "PASS" "replay sampling, validity decisions, batch layout, and terminal handling match"

echo
echo "=== Stage 4 summary ==="
sed -n '1,180p' "$SUMMARY_OUT"
echo
echo "MATCH"
