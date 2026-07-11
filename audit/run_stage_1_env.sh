#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/audit_config.sh"

mkdir -p "$OUT"

ACTION_TAPE_NPY="$OUT/action_tape_seed${SEED}_${TRACE_STEPS}.npy"
ACTION_TAPE_TXT="$OUT/action_tape_seed${SEED}_${TRACE_STEPS}.txt"
PYTORCH_ENV_OUT="$OUT/pytorch_env.jsonl"
DEEPMIND_ENV_OUT="$OUT/deepmind_env.jsonl"
COMPARE_OUT="$OUT/env_compare.txt"
FRAME_DUMP_DIR="${FRAME_DUMP_DIR:-$OUT/frames}"
FRAME_DUMP_STEPS="${FRAME_DUMP_STEPS:-20}"
FRAME_DUMP_PNG="${FRAME_DUMP_PNG:-1}"

export AUDIT_DIR ATARI_DIR DM_DIR ENV_ID ROM SEED TRACE_STEPS FRAME_SKIP OUT
export ACTION_TAPE_TXT PYTORCH_ENV_OUT DEEPMIND_ENV_OUT
export LUA_ACTION_OFFSET DM_ACTION_MODE ACTION_TAPE_MODE ACTION_TAPE_VALUES ACTION_TAPE_SEQUENCE
export FRAME_DUMP_DIR FRAME_DUMP_STEPS
export FRAME_DUMP_PNG

if [[ -n "${ACTION_TAPE_SEQUENCE:-}" ]]; then
  "$PYTHON_BIN" "$AUDIT_DIR/make_action_tape.py" \
    --seed "$SEED" \
    --steps "$TRACE_STEPS" \
    --action-count "$ACTION_COUNT" \
    --sequence "$ACTION_TAPE_SEQUENCE" \
    --out-dir "$OUT"
elif [[ "$ACTION_TAPE_MODE" == "ale_code" ]]; then
  "$PYTHON_BIN" "$AUDIT_DIR/make_action_tape.py" \
    --seed "$SEED" \
    --steps "$TRACE_STEPS" \
    --action-count "$ACTION_COUNT" \
    --action-values "$ACTION_TAPE_VALUES" \
    --out-dir "$OUT"
else
  "$PYTHON_BIN" "$AUDIT_DIR/make_action_tape.py" \
    --seed "$SEED" \
    --steps "$TRACE_STEPS" \
    --action-count "$ACTION_COUNT" \
    --out-dir "$OUT"
fi

"$PYTHON_BIN" "$AUDIT_DIR/pytorch/trace_env.py" \
  --env-id "$ENV_ID" \
  --seed "$SEED" \
  --steps "$TRACE_STEPS" \
  --frame-skip "$FRAME_SKIP" \
  --action-tape "$ACTION_TAPE_NPY" \
  --out "$PYTORCH_ENV_OUT" \
  --frame-dump-dir "$FRAME_DUMP_DIR" \
  --frame-dump-steps "$FRAME_DUMP_STEPS" \
  --action-tape-mode "$ACTION_TAPE_MODE"

if [[ ! -d "$DM_DIR" ]]; then
  echo "DeepMind directory does not exist: $DM_DIR" >&2
  echo "Set DM_DIR=/path/to/DeepMind-Atari-Deep-Q-Learner and rerun." >&2
  exit 2
fi

(
  cd "$DM_DIR"
  "$LUAJIT_BIN" "$AUDIT_DIR/deepmind/trace_env.lua"
)

set +e
"$PYTHON_BIN" "$AUDIT_DIR/compare_jsonl.py" \
  --left "$PYTORCH_ENV_OUT" \
  --right "$DEEPMIND_ENV_OUT" \
  --left-label pytorch \
  --right-label deepmind \
  --ignore "$COMPARE_IGNORE" \
  --ignore-diagnostic-metadata \
  --float-tol "$FLOAT_TOL" | tee "$COMPARE_OUT"
status=${PIPESTATUS[0]}
set -e

exit "$status"
