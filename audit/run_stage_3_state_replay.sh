#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/freeze_config.sh"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$ATARI_DIR/audit_outputs}"
STAGE2_PREPROCESS_DIR="${STAGE2_PREPROCESS_DIR:-$BASE_OUTPUT_DIR/stage2_preprocess}"
OUT="${OUT:-$BASE_OUTPUT_DIR/stage3_replay}"
CANONICAL_PROCESSED_DIR="${CANONICAL_PROCESSED_DIR:-$OUT/canonical}"

PYTORCH_STATE_STACK_OUT="$OUT/pytorch_state_stack.jsonl"
DEEPMIND_STATE_STACK_OUT="$OUT/deepmind_state_stack.jsonl"
PYTORCH_REPLAY_INSERT_OUT="$OUT/pytorch_replay_insert.jsonl"
DEEPMIND_REPLAY_INSERT_OUT="$OUT/deepmind_replay_insert.jsonl"
STATE_STACK_COMPARE_OUT="$OUT/state_stack_compare.txt"
REPLAY_INSERT_COMPARE_OUT="$OUT/replay_insert_compare.txt"
SUMMARY_OUT="$OUT/STAGE3_SUMMARY.md"
VALIDATION_OUT="$OUT/canonical_validation.txt"

mkdir -p "$OUT"

export AUDIT_DIR ATARI_DIR DM_DIR OUT PYTHON_BIN LUAJIT_BIN
export STAGE3_CANONICAL_DIR="$CANONICAL_PROCESSED_DIR"
export DEEPMIND_STATE_STACK_OUT DEEPMIND_REPLAY_INSERT_OUT

write_summary() {
  local status="$1"
  local reason="$2"
  {
    echo "# Stage 3 State Stack / Replay Insert Summary"
    echo
    echo "- Status: $status"
    echo "- Reason: $reason"
    echo "- Canonical processed tape: $CANONICAL_PROCESSED_DIR"
    echo "- Canonical validation: $VALIDATION_OUT"
    echo "- PyTorch state stack trace: $PYTORCH_STATE_STACK_OUT"
    echo "- DeepMind state stack trace: $DEEPMIND_STATE_STACK_OUT"
    echo "- State-stack compare: $STATE_STACK_COMPARE_OUT"
    echo "- PyTorch replay insert trace: $PYTORCH_REPLAY_INSERT_OUT"
    echo "- DeepMind replay insert trace: $DEEPMIND_REPLAY_INSERT_OUT"
    echo "- Replay-insert compare: $REPLAY_INSERT_COMPARE_OUT"
    echo
    if [[ -f "$STATE_STACK_COMPARE_OUT" ]]; then
      echo "## State Stack Compare"
      echo
      echo '```text'
      sed -n '1,80p' "$STATE_STACK_COMPARE_OUT"
      echo '```'
      echo
    fi
    if [[ -f "$REPLAY_INSERT_COMPARE_OUT" ]]; then
      echo "## Replay Insert Compare"
      echo
      echo '```text'
      sed -n '1,80p' "$REPLAY_INSERT_COMPARE_OUT"
      echo '```'
      echo
    fi
  } > "$SUMMARY_OUT"
}

echo "=== Stage 3: build canonical DeepMind processed tape ==="
"$PYTHON_BIN" "$AUDIT_DIR/build_canonical_processed_tape.py" \
  --stage2-dir "$STAGE2_PREPROCESS_DIR" \
  --out-dir "$CANONICAL_PROCESSED_DIR" \
  --hist-len 4

echo
echo "=== Stage 3: validate canonical processed tape ==="
"$PYTHON_BIN" "$AUDIT_DIR/validate_canonical_processed_tape.py" \
  --tape-dir "$CANONICAL_PROCESSED_DIR" \
  --report "$VALIDATION_OUT"

echo
echo "=== Stage 3: PyTorch state stack trace ==="
"$PYTHON_BIN" "$AUDIT_DIR/pytorch/trace_state_stack.py" \
  --tape-dir "$CANONICAL_PROCESSED_DIR" \
  --out "$PYTORCH_STATE_STACK_OUT" \
  --hist-len 4

echo
echo "=== Stage 3: DeepMind state stack trace ==="
(
  cd "$DM_DIR/dqn"
  "$LUAJIT_BIN" "$AUDIT_DIR/deepmind/trace_state_stack.lua"
)

echo
echo "=== Stage 3: compare state stacks ==="
set +e
"$PYTHON_BIN" "$AUDIT_DIR/compare_jsonl.py" \
  --left "$PYTORCH_STATE_STACK_OUT" \
  --right "$DEEPMIND_STATE_STACK_OUT" \
  --left-label pytorch \
  --right-label deepmind \
  --ignore "source" \
  --float-tol "$FLOAT_TOL" | tee "$STATE_STACK_COMPARE_OUT"
state_status=${PIPESTATUS[0]}
set -e
if [[ "$state_status" -ne 0 ]]; then
  write_summary "FAIL" "state stack mismatch"
  sed -n '1,120p' "$SUMMARY_OUT"
  exit "$state_status"
fi

echo
echo "=== Stage 3: PyTorch replay insert trace ==="
"$PYTHON_BIN" "$AUDIT_DIR/pytorch/trace_replay_insert.py" \
  --tape-dir "$CANONICAL_PROCESSED_DIR" \
  --out "$PYTORCH_REPLAY_INSERT_OUT" \
  --hist-len 4

echo
echo "=== Stage 3: DeepMind replay insert trace ==="
(
  cd "$DM_DIR/dqn"
  "$LUAJIT_BIN" "$AUDIT_DIR/deepmind/trace_replay_insert.lua"
)

echo
echo "=== Stage 3: compare replay insert semantics ==="
set +e
"$PYTHON_BIN" "$AUDIT_DIR/compare_jsonl.py" \
  --left "$PYTORCH_REPLAY_INSERT_OUT" \
  --right "$DEEPMIND_REPLAY_INSERT_OUT" \
  --left-label pytorch \
  --right-label deepmind \
  --ignore "source" \
  --float-tol "$FLOAT_TOL" | tee "$REPLAY_INSERT_COMPARE_OUT"
replay_status=${PIPESTATUS[0]}
set -e
if [[ "$replay_status" -ne 0 ]]; then
  write_summary "FAIL" "replay insert mismatch"
  sed -n '1,120p' "$SUMMARY_OUT"
  exit "$replay_status"
fi

write_summary "PASS" "state stacks and semantic replay insertions match"

echo
echo "=== Stage 3 summary ==="
sed -n '1,160p' "$SUMMARY_OUT"
echo
echo "MATCH"
