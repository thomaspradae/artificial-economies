#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/audit_config.sh"

BASE_OUT="${OUT:-$ATARI_DIR/audit_outputs/stage1f}"
SEED_MIN="${STAGE1F_SEED_MIN:-0}"
SEED_MAX="${STAGE1F_SEED_MAX:-50}"
DELAYS="${STAGE1F_DELAYS:-0,1,2,3,4,5,10,20,30}"

mkdir -p "$BASE_OUT"

run_trace() {
  local label="$1"
  local seed="$2"
  local steps="$3"
  local frame_skip="$4"
  local dump_steps="$5"
  local sequence="$6"
  local run_out="$BASE_OUT/$label"

  mkdir -p "$run_out"
  echo "=== Stage 1f trace $label seed=$seed steps=$steps frame_skip=$frame_skip sequence=$sequence ==="
  OUT="$run_out" \
  SEED="$seed" \
  TRACE_STEPS="$steps" \
  FRAME_SKIP="$frame_skip" \
  FRAME_DUMP_STEPS="$dump_steps" \
  FRAME_DUMP_PNG=0 \
  ACTION_TAPE_MODE=index \
  ACTION_TAPE_SEQUENCE="$sequence" \
  bash "$SCRIPT_DIR/run_stage_1_env.sh" > "$run_out/stage1.log" 2>&1
  local status=$?
  echo "stage1_status=$status" > "$run_out/status.txt"
  if [[ -f "$run_out/env_compare.txt" ]]; then
    cat "$run_out/env_compare.txt" >> "$run_out/status.txt"
  fi
}

echo "=== Stage 1f seed grid ==="
for seed in $(seq "$SEED_MIN" "$SEED_MAX"); do
  printf -v seed_label "seed_grid/seed_%03d" "$seed"
  run_trace "$seed_label" "$seed" 1 4 0 "1"
done

echo "=== Stage 1f delayed FIRE ==="
IFS=',' read -r -a delay_values <<< "$DELAYS"
for delay in "${delay_values[@]}"; do
  delay="${delay//[[:space:]]/}"
  [[ -z "$delay" ]] && continue
  printf -v delay_label "delayed_fire/delay_%03d" "$delay"
  steps=$((delay + 5))
  dump_steps="$delay"
  if [[ "$delay" -eq 0 ]]; then
    sequence="1,0x4"
  else
    sequence="0x${delay},1,0x4"
  fi
  run_trace "$delay_label" "$SEED" "$steps" 4 "$dump_steps" "$sequence"
done

echo "=== Stage 1f repeat boundary ==="
run_trace "repeat_boundary/repeat_logic" "$SEED" 1 4 0 "1"
run_trace "repeat_boundary/one_step_fire" "$SEED" 1 1 0 "1"
run_trace "repeat_boundary/fire_then_noop" "$SEED" 2 1 1 "1,0"
run_trace "repeat_boundary/fire4" "$SEED" 4 1 3 "1x4"

echo "=== Stage 1f same-side determinism ==="
run_trace "determinism/run_a" "$SEED" 1 4 0 "1"
run_trace "determinism/run_b" "$SEED" 1 4 0 "1"

"$PYTHON_BIN" "$SCRIPT_DIR/diagnose_fire_rng.py" \
  --out "$BASE_OUT" \
  --seed-max "$SEED_MAX" \
  --delays "$DELAYS"

echo "wrote $BASE_OUT/summary.txt"
