#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/freeze_config.sh"

mkdir -p "$OUT" "$CANONICAL_TAPE_DIR"

if [[ -z "${ACTION_TAPE_SEQUENCE:-}" ]]; then
  if [[ "$TRACE_STEPS" -le 1 ]]; then
    ACTION_TAPE_SEQUENCE="1"
  else
    ACTION_TAPE_SEQUENCE="1,0x$((TRACE_STEPS - 1))"
  fi
fi

ACTION_TAPE_NPY="$CANONICAL_TAPE_DIR/action_tape_seed${SEED}_${TRACE_STEPS}.npy"
ACTION_TAPE_TXT="$CANONICAL_TAPE_DIR/action_tape_seed${SEED}_${TRACE_STEPS}.txt"
PYTORCH_PREPROCESS_OUT="$OUT/pytorch_preprocess.jsonl"
DEEPMIND_PREPROCESS_OUT="$OUT/deepmind_preprocess.jsonl"
PREPROCESS_COMPARE_OUT="$OUT/preprocess_compare.txt"
FREEZE_MANIFEST_OUT="$OUT/freeze_manifest.json"

export AUDIT_DIR ATARI_DIR DM_DIR ROM ENV_ID SEED TRACE_STEPS FRAME_SKIP OUT
export ACTION_TAPE_TXT ACTION_TAPE_MODE ACTION_TAPE_VALUES ACTION_TAPE_SEQUENCE ACTION_COUNT
export CANONICAL_TAPE_DIR DM_ACTION_MODE LUA_ACTION_OFFSET PYTORCH_RESIZE_INTERPOLATION PYTORCH_PREPROCESS_SOURCE
export DEEPMIND_PREPROCESS_OUT
export DEEPMIND_PROCESSED_DIR="$OUT/deepmind_processed"

echo "=== Stage 2: action tape ==="
"$PYTHON_BIN" "$AUDIT_DIR/make_action_tape.py" \
  --seed "$SEED" \
  --steps "$TRACE_STEPS" \
  --action-count "$ACTION_COUNT" \
  --sequence "$ACTION_TAPE_SEQUENCE" \
  --out-dir "$CANONICAL_TAPE_DIR"

echo "=== Stage 2: canonical DeepMind env tape ==="
(
  cd "$DM_DIR"
  "$LUAJIT_BIN" "$AUDIT_DIR/deepmind/build_canonical_env_tape.lua"
)

echo "=== Stage 2: validate canonical tape ==="
"$PYTHON_BIN" "$AUDIT_DIR/validate_canonical_env_tape.py" \
  --tape-dir "$CANONICAL_TAPE_DIR" \
  --report "$OUT/canonical_tape_validation.txt"

"$PYTHON_BIN" "$AUDIT_DIR/build_canonical_frame_tape.py" \
  --tape-dir "$CANONICAL_TAPE_DIR" \
  --out "$FREEZE_MANIFEST_OUT" \
  --repo-root "$ATARI_DIR" \
  --dm-dir "$DM_DIR"

echo "=== Stage 2: DeepMind preprocessing ==="
(
  cd "$DM_DIR/dqn"
  "$LUAJIT_BIN" "$AUDIT_DIR/deepmind/trace_preprocess.lua"
)

echo "=== Stage 2: PyTorch preprocessing ==="
"$PYTHON_BIN" "$AUDIT_DIR/pytorch/trace_preprocess.py" \
  --tape-dir "$CANONICAL_TAPE_DIR" \
  --out "$PYTORCH_PREPROCESS_OUT" \
  --processed-dir "$OUT/pytorch_processed" \
  --resize-interpolation "$PYTORCH_RESIZE_INTERPOLATION" \
  --preprocess-source "${PYTORCH_PREPROCESS_SOURCE:-train_nature}"

echo "=== Stage 2: compare ==="
set +e
"$PYTHON_BIN" "$AUDIT_DIR/compare_jsonl.py" \
  --left "$PYTORCH_PREPROCESS_OUT" \
  --right "$DEEPMIND_PREPROCESS_OUT" \
  --left-label pytorch \
  --right-label deepmind \
  --ignore "source,input_path,input_hash,input_t7_path,processed_path,preprocess_source,preprocess_source_mode,resize_interpolation,processed_float,dtype,source_report_path,intermediate_dir" \
  --float-tol "$FLOAT_TOL" | tee "$PREPROCESS_COMPARE_OUT"
status=${PIPESTATUS[0]}
set -e

echo
echo "=== outputs ==="
echo "$CANONICAL_TAPE_DIR/manifest.jsonl"
echo "$PYTORCH_PREPROCESS_OUT"
echo "$DEEPMIND_PREPROCESS_OUT"
echo "$PREPROCESS_COMPARE_OUT"

exit "$status"
