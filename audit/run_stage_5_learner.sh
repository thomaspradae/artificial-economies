#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/freeze_config.sh"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$ATARI_DIR/audit_outputs}"
STAGE4_REPLAY_SAMPLE_DIR="${STAGE4_REPLAY_SAMPLE_DIR:-$BASE_OUTPUT_DIR/stage4_replay_sample}"
OUT="${OUT:-$BASE_OUTPUT_DIR/stage5_learner}"
FIXTURE_DIR="${STAGE5_FIXTURE_DIR:-$OUT/learner_fixture}"
MODEL_DIR="${STAGE5_MODEL_DIR:-$OUT/model_exchange}"

PYTORCH_LEARNER_OUT="$OUT/pytorch_learner.jsonl"
DEEPMIND_LEARNER_OUT="$OUT/deepmind_learner.jsonl"
LEARNER_COMPARE_OUT="$OUT/learner_compare.txt"
FORWARD_DIAGNOSIS_OUT="$OUT/forward_architecture_diagnosis.txt"
SUMMARY_OUT="$OUT/STAGE5_SUMMARY.md"
IMPORT_REPORT="$MODEL_DIR/import_report.json"
MAPPING_REPORT="$MODEL_DIR/mapping_verification.json"

mkdir -p "$OUT" "$MODEL_DIR"

export AUDIT_DIR ATARI_DIR DM_DIR OUT PYTHON_BIN LUAJIT_BIN
export STAGE4_REPLAY_SAMPLE_DIR STAGE5_FIXTURE_DIR="$FIXTURE_DIR" STAGE5_MODEL_DIR="$MODEL_DIR"
export DEEPMIND_LEARNER_OUT ACTION_COUNT="${ACTION_COUNT:-4}"
export LEARNER_LR="${LEARNER_LR:-0.00025}" GAMMA="${GAMMA:-0.99}" MODEL_SEED="${MODEL_SEED:-1}"

write_summary() {
  local status="$1"
  local reason="$2"
  {
    echo "# Stage 5 Learner Math Summary"
    echo
    echo "- Status: $status"
    echo "- Reason: $reason"
    echo "- Stage 4 source: $STAGE4_REPLAY_SAMPLE_DIR"
    echo "- Learner fixture: $FIXTURE_DIR"
    echo "- Model exchange: $MODEL_DIR"
    echo "- PyTorch learner trace: $PYTORCH_LEARNER_OUT"
    echo "- DeepMind learner trace: $DEEPMIND_LEARNER_OUT"
    echo "- Learner compare: $LEARNER_COMPARE_OUT"
    echo "- Forward architecture diagnosis: $FORWARD_DIAGNOSIS_OUT"
    echo
    if [[ -f "$FIXTURE_DIR/manifest.json" ]]; then
      echo "## Fixture"
      echo
      "$PYTHON_BIN" - "$FIXTURE_DIR/manifest.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
print("```text")
print(f"batch_name: {data['batch_name']}")
print(f"batch_size: {data['batch_size']}")
print(f"content_hash: {data['content_hash']}")
print(f"indices_first_8: {data['indices_zero_based'][:8]}")
print("```")
PY
      echo
    fi
    if [[ -f "$MAPPING_REPORT" ]]; then
      echo "## Model Mapping"
      echo
      echo '```json'
      sed -n '1,120p' "$MAPPING_REPORT"
      echo '```'
      echo
    fi
    if [[ -f "$LEARNER_COMPARE_OUT" ]]; then
      echo "## Learner Compare"
      echo
      echo '```text'
      sed -n '1,120p' "$LEARNER_COMPARE_OUT"
      echo '```'
      echo
    fi
    if [[ -f "$FORWARD_DIAGNOSIS_OUT" ]]; then
      echo "## Forward Architecture Diagnosis"
      echo
      echo '```text'
      sed -n '1,160p' "$FORWARD_DIAGNOSIS_OUT"
      echo '```'
      echo
    fi
  } > "$SUMMARY_OUT"
}

echo "=== Stage 5: build frozen learner minibatch ==="
"$PYTHON_BIN" "$AUDIT_DIR/build_stage5_learner_fixture.py" \
  --stage4-dir "$STAGE4_REPLAY_SAMPLE_DIR" \
  --out-dir "$FIXTURE_DIR" \
  --gamma "$GAMMA"

echo
echo "=== Stage 5: export DeepMind convnet_atari3 weights ==="
(
  cd "$DM_DIR/dqn"
  "$LUAJIT_BIN" "$AUDIT_DIR/model_exchange/export_deepmind_model.lua"
)

echo
echo "=== Stage 5: import weights into PyTorch Atari QNetwork ==="
"$PYTHON_BIN" "$AUDIT_DIR/model_exchange/import_deepmind_model.py" \
  --model-dir "$MODEL_DIR" \
  --atari-dir "$ATARI_DIR" \
  --out "$MODEL_DIR/pytorch_model.pt" \
  --report "$IMPORT_REPORT"

echo
echo "=== Stage 5: verify parameter mapping ==="
"$PYTHON_BIN" "$AUDIT_DIR/model_exchange/verify_model_mapping.py" \
  --model-dir "$MODEL_DIR" \
  --pytorch-model "$MODEL_DIR/pytorch_model.pt" \
  --report "$MAPPING_REPORT"

echo
echo "=== Stage 5: PyTorch learner trace ==="
"$PYTHON_BIN" "$AUDIT_DIR/pytorch/trace_learner.py" \
  --fixture-dir "$FIXTURE_DIR" \
  --model "$MODEL_DIR/pytorch_model.pt" \
  --atari-dir "$ATARI_DIR" \
  --out "$PYTORCH_LEARNER_OUT" \
  --lr "$LEARNER_LR" \
  --gamma "$GAMMA"

echo
echo "=== Stage 5: DeepMind learner trace ==="
(
  cd "$DM_DIR/dqn"
  "$LUAJIT_BIN" "$AUDIT_DIR/deepmind/trace_learner.lua"
)

echo
echo "=== Stage 5: compare learner math ==="
set +e
"$PYTHON_BIN" "$AUDIT_DIR/compare_jsonl.py" \
  --left "$PYTORCH_LEARNER_OUT" \
  --right "$DEEPMIND_LEARNER_OUT" \
  --left-label pytorch \
  --right-label deepmind \
  --ignore "timestamp,runtime,path,source_path,source,env_id,rom,reset_info,info,dm_action_mode,lua_action_offset,lua_action,ale_action,action_meanings,ale_frame_number,dtype,hash,qnetwork_source_kind,loss_source_kind,optimizer_source_kind" \
  --float-tol "${STAGE5_FLOAT_TOL:-1e-4}" | tee "$LEARNER_COMPARE_OUT"
compare_status=${PIPESTATUS[0]}
set -e

echo
echo "=== Stage 5: diagnose forward architecture mismatch ==="
"$PYTHON_BIN" "$AUDIT_DIR/pytorch/diagnose_forward_padding.py" \
  --fixture-dir "$FIXTURE_DIR" \
  --model-dir "$MODEL_DIR" \
  --deepmind-trace "$DEEPMIND_LEARNER_OUT" \
  --atari-dir "$ATARI_DIR" \
  --report "$FORWARD_DIAGNOSIS_OUT"

if [[ "$compare_status" -eq 0 ]]; then
  write_summary "PASS" "forward, target, loss-gradient, and one optimizer update match"
  sed -n '1,180p' "$SUMMARY_OUT"
  echo
  echo "MATCH"
else
  write_summary "MISMATCH" "learner comparison found a first mismatch; inspect learner_compare.txt"
  sed -n '1,220p' "$SUMMARY_OUT"
  echo
  echo "STAGE5_MISMATCH"
fi
