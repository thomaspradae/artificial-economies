#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/audit_config.sh"

BASE_OUT="${OUT:-$ATARI_DIR/audit_outputs/stage1e}"
TRACE_STEPS="${STAGE1E_TRACE_STEPS:-200}"
FRAME_DUMP_STEPS="${STAGE1E_FRAME_DUMP_STEPS:-199}"
ACTION_TAPE_MODE="${ACTION_TAPE_MODE:-index}"

mkdir -p "$BASE_OUT"

run_tape() {
  local label="$1"
  local sequence="$2"
  local tape_out="$BASE_OUT/$label"
  mkdir -p "$tape_out"

  echo "=== Stage 1e $label ==="
  OUT="$tape_out" \
  TRACE_STEPS="$TRACE_STEPS" \
  FRAME_DUMP_STEPS="$FRAME_DUMP_STEPS" \
  FRAME_DUMP_PNG="${STAGE1E_FRAME_DUMP_PNG:-0}" \
  ACTION_TAPE_MODE="$ACTION_TAPE_MODE" \
  ACTION_TAPE_SEQUENCE="$sequence" \
  bash "$SCRIPT_DIR/run_stage_1_env.sh" > "$tape_out/stage1.log" 2>&1
  local stage_status=$?

  echo "stage1_status: $stage_status" | tee "$tape_out/status.txt"
  echo "=== env_compare $label ==="
  if [[ -f "$tape_out/env_compare.txt" ]]; then
    cat "$tape_out/env_compare.txt"
  else
    echo "missing env_compare.txt"
  fi

  "$PYTHON_BIN" "$SCRIPT_DIR/diagnose_state_vs_screen.py" \
    --out "$tape_out" \
    --pytorch "$tape_out/pytorch_env.jsonl" \
    --deepmind "$tape_out/deepmind_env.jsonl" \
    --frames-dir "$tape_out/frames" \
    --report "$tape_out/report.txt" \
    --label "$label"
  local diag_status=$?

  echo "stage1_status=$stage_status diagnostic_status=$diag_status" >> "$tape_out/status.txt"
}

run_tape noop "0x${TRACE_STEPS}"
run_tape fire "1x${TRACE_STEPS}"
run_tape right "2x${TRACE_STEPS}"
run_tape left "3x${TRACE_STEPS}"

SUMMARY="$BASE_OUT/summary.txt"
{
  echo "Stage 1e summary"
  echo
  for label in noop fire right left; do
    echo "=== $label ==="
    if [[ -f "$BASE_OUT/$label/report.txt" ]]; then
      sed -n '/^Questions$/,/^First events$/p' "$BASE_OUT/$label/report.txt"
      grep -E '^(first_raw_frame_mismatch|first_ram_mismatch|Conclusion|RAM unavailable|RAM differs|RAM/reward|No state)' "$BASE_OUT/$label/report.txt" || true
    else
      echo "missing report"
    fi
    echo
  done
} | tee "$SUMMARY"

echo "wrote $SUMMARY"
