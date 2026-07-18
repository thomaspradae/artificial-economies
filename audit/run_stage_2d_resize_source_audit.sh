#!/usr/bin/env bash
set -euo pipefail

USER_OUT="${OUT:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/freeze_config.sh"

if [[ -n "$USER_OUT" ]]; then
  STAGE2D_OUT="${STAGE2D_OUT:-$USER_OUT}"
else
  STAGE2D_OUT="${STAGE2D_OUT:-$ATARI_DIR/audit_outputs/stage2d_resize_source}"
fi

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$ATARI_DIR/audit_outputs}"
STAGE2C_DIR="${STAGE2C_DIR:-$BASE_OUTPUT_DIR/stage2c_resize}"
STAGE2_PREPROCESS_DIR="${STAGE2_PREPROCESS_DIR:-$BASE_OUTPUT_DIR/stage2_preprocess}"
FIXTURE_PATHS="${STAGE2D_FIXTURE_PATHS:-$STAGE2C_DIR/fixtures/fixture_paths.txt}"
CONTRACT_JSON="$STAGE2D_OUT/lua_contract.json"
ORACLE_DIR="$STAGE2D_OUT/oracle_outputs"
PYTHON_OUTPUT_DIR="$STAGE2D_OUT/python_outputs"
COMPARISON="$STAGE2D_OUT/comparison.txt"
UNRESOLVED="$STAGE2D_OUT/UNRESOLVED.md"
SOURCE_REPORT="$STAGE2D_OUT/source_report.txt"
FREEZE2_RERUN_DIR="$STAGE2D_OUT/freeze2_rerun"

mkdir -p "$STAGE2D_OUT" "$ORACLE_DIR" "$PYTHON_OUTPUT_DIR"

export AUDIT_DIR ATARI_DIR DM_DIR ROM ENV_ID SEED TRACE_STEPS FRAME_SKIP PYTHON_BIN LUAJIT_BIN
export STAGE2D_OUT STAGE2D_FIXTURE_PATHS="$FIXTURE_PATHS"

if [[ ! -f "$FIXTURE_PATHS" ]]; then
  echo "missing Stage 2c fixture paths: $FIXTURE_PATHS" >&2
  echo "Run audit/run_stage_2c_resize_forensics.sh first." >&2
  exit 2
fi

echo "=== Stage 2d: extract installed Torch7 resize source ==="
OUT="$STAGE2D_OUT" STAGE2D_OUT="$STAGE2D_OUT" bash "$AUDIT_DIR/extract_torch7_resize_source.sh"

echo
echo "=== Stage 2d: dump installed Lua resize contract ==="
(
  cd "$DM_DIR/dqn"
  STAGE2D_CONTRACT_OUT="$CONTRACT_JSON" \
  STAGE2D_ORACLE_DIR="$ORACLE_DIR" \
  STAGE2D_FIXTURE_PATHS="$FIXTURE_PATHS" \
  "$LUAJIT_BIN" "$AUDIT_DIR/deepmind/dump_resize_contract.lua"
)

echo
echo "=== Stage 2d: test Python Torch7 resize clone ==="
set +e
"$PYTHON_BIN" "$AUDIT_DIR/tests/test_torch7_resize_clone.py" \
  --contract "$CONTRACT_JSON" \
  --representation float_0_1 \
  --python-output-dir "$PYTHON_OUTPUT_DIR" \
  --comparison "$COMPARISON" \
  --unresolved "$UNRESOLVED" \
  --source-report "$SOURCE_REPORT" \
  --write-preprocess "$AUDIT_DIR/pytorch/deepmind_preprocess.py"
clone_status=$?
set -e

if [[ "$clone_status" -ne 0 ]]; then
  echo
  echo "=== Stage 2d unresolved ==="
  echo "$COMPARISON"
  echo "$UNRESOLVED"
  sed -n '1,160p' "$COMPARISON"
  exit "$clone_status"
fi

echo
echo "=== Stage 2d: exact clone found; rerun normal Freeze 2 ==="
OUT="$FREEZE2_RERUN_DIR" \
CANONICAL_TAPE_DIR="$STAGE2_PREPROCESS_DIR/canonical_frames" \
PYTORCH_PREPROCESS_SOURCE=deepmind_clone \
bash "$AUDIT_DIR/run_stage_2_preprocess.sh"

if ! grep -q '^MATCH$' "$FREEZE2_RERUN_DIR/preprocess_compare.txt"; then
  echo "Stage 2d clone matched fixtures, but normal Freeze 2 did not report MATCH." >&2
  echo "$FREEZE2_RERUN_DIR/preprocess_compare.txt" >&2
  exit 1
fi

echo
echo "=== Stage 2d outputs ==="
echo "$SOURCE_REPORT"
echo "$CONTRACT_JSON"
echo "$ORACLE_DIR"
echo "$PYTHON_OUTPUT_DIR"
echo "$COMPARISON"
echo "$AUDIT_DIR/pytorch/deepmind_preprocess.py"
echo "$FREEZE2_RERUN_DIR/preprocess_compare.txt"
echo
echo "MATCH"
