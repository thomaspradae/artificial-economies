#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/freeze_config.sh"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$ATARI_DIR/audit_outputs}"
STAGE4_REPLAY_SAMPLE_DIR="${STAGE4_REPLAY_SAMPLE_DIR:-$BASE_OUTPUT_DIR/stage4_replay_sample}"
OUT="${OUT:-$BASE_OUTPUT_DIR/stage5_learner}"
FIXTURE_DIR="${STAGE5_FIXTURE_DIR:-$OUT/learner_fixture}"
MODEL_DIR="${STAGE5_MODEL_DIR:-$OUT/model_exchange}"

PYTORCH_FORWARD_OUT="$OUT/pytorch_forward_layers.jsonl"
DEEPMIND_FORWARD_OUT="$OUT/deepmind_forward_layers.jsonl"
PYTORCH_TENSOR_DIR="$OUT/forward_layers/pytorch"
DEEPMIND_TENSOR_DIR="$OUT/forward_layers/deepmind"
ARCHITECTURE_MANIFEST="$OUT/architecture_manifest.json"
INTERMEDIATE_COMPARE_OUT="$OUT/intermediate_forward_compare.txt"
INTERMEDIATE_COMPARE_JSONL="$OUT/intermediate_forward_compare.jsonl"
LEARNER_COMPARE_OUT="$OUT/learner_compare.txt"
MODEL_MAPPING_TXT="$OUT/model_mapping_report.txt"
MAPPING_REPORT="$MODEL_DIR/mapping_verification_faithful.json"
SUMMARY_OUT="$OUT/STAGE5A_SUMMARY.md"

mkdir -p "$OUT" "$MODEL_DIR"

export AUDIT_DIR ATARI_DIR DM_DIR OUT PYTHON_BIN LUAJIT_BIN
export STAGE4_REPLAY_SAMPLE_DIR STAGE5_FIXTURE_DIR="$FIXTURE_DIR" STAGE5_MODEL_DIR="$MODEL_DIR"
export DEEPMIND_FORWARD_LAYERS_OUT="$DEEPMIND_FORWARD_OUT"
export DEEPMIND_FORWARD_TENSOR_DIR="$DEEPMIND_TENSOR_DIR"
export ACTION_COUNT="${ACTION_COUNT:-4}" MODEL_SEED="${MODEL_SEED:-1}"

write_mapping_text() {
  "$PYTHON_BIN" - "$MAPPING_REPORT" "$MODEL_MAPPING_TXT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
lines = [
    "Stage 5A model mapping report",
    f"status: {report['status']}",
    f"network_kind: {report.get('network_kind')}",
    "",
    "Layers:",
]
for layer in report["layers"]:
    lines.append(
        f"{layer['name']}: weight_match={layer['weight_match']} "
        f"bias_match={layer['bias_match']} weight_key={layer['weight_key']} bias_key={layer['bias_key']}"
    )
open(sys.argv[2], "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY
}

write_learner_compare() {
  local status="$1"
  {
    echo "$status"
    echo
    sed -n '1,120p' "$INTERMEDIATE_COMPARE_OUT"
  } > "$LEARNER_COMPARE_OUT"
}

write_summary() {
  local status="$1"
  local reason="$2"
  {
    echo "# Stage 5A Forward-Pass Summary"
    echo
    echo "- Status: $status"
    echo "- Reason: $reason"
    echo "- Stage 4 source: $STAGE4_REPLAY_SAMPLE_DIR"
    echo "- Learner fixture: $FIXTURE_DIR"
    echo "- Model exchange: $MODEL_DIR"
    echo "- Model mapping report: $MODEL_MAPPING_TXT"
    echo "- Architecture manifest: $ARCHITECTURE_MANIFEST"
    echo "- PyTorch forward trace: $PYTORCH_FORWARD_OUT"
    echo "- DeepMind forward trace: $DEEPMIND_FORWARD_OUT"
    echo "- Intermediate forward compare: $INTERMEDIATE_COMPARE_OUT"
    echo "- Learner compare: $LEARNER_COMPARE_OUT"
    echo
    if [[ -f "$MODEL_MAPPING_TXT" ]]; then
      echo "## Weight Exchange"
      echo
      echo '```text'
      sed -n '1,120p' "$MODEL_MAPPING_TXT"
      echo '```'
      echo
    fi
    if [[ -f "$ARCHITECTURE_MANIFEST" ]]; then
      echo "## Architecture Manifest"
      echo
      echo '```json'
      sed -n '1,160p' "$ARCHITECTURE_MANIFEST"
      echo '```'
      echo
    fi
    if [[ -f "$INTERMEDIATE_COMPARE_OUT" ]]; then
      echo "## Forward Compare"
      echo
      echo '```text'
      sed -n '1,180p' "$INTERMEDIATE_COMPARE_OUT"
      echo '```'
      echo
    fi
  } > "$SUMMARY_OUT"
}

echo "=== Stage 5A: build frozen learner minibatch ==="
"$PYTHON_BIN" "$AUDIT_DIR/build_stage5_learner_fixture.py" \
  --stage4-dir "$STAGE4_REPLAY_SAMPLE_DIR" \
  --out-dir "$FIXTURE_DIR"

echo
echo "=== Stage 5A: export DeepMind convnet_atari3 weights ==="
(
  cd "$DM_DIR/dqn"
  "$LUAJIT_BIN" "$AUDIT_DIR/model_exchange/export_deepmind_model.lua"
)

echo
echo "=== Stage 5A: import weights into faithful PyTorch network ==="
"$PYTHON_BIN" "$AUDIT_DIR/model_exchange/import_deepmind_model.py" \
  --model-dir "$MODEL_DIR" \
  --atari-dir "$ATARI_DIR" \
  --network-kind faithful \
  --out "$MODEL_DIR/pytorch_faithful_model.pt" \
  --report "$MODEL_DIR/import_report_faithful.json"

echo
echo "=== Stage 5A: verify faithful parameter mapping ==="
"$PYTHON_BIN" "$AUDIT_DIR/model_exchange/verify_model_mapping.py" \
  --model-dir "$MODEL_DIR" \
  --pytorch-model "$MODEL_DIR/pytorch_faithful_model.pt" \
  --network-kind faithful \
  --report "$MAPPING_REPORT"
write_mapping_text

echo
echo "=== Stage 5A: PyTorch faithful layer trace ==="
"$PYTHON_BIN" "$AUDIT_DIR/pytorch/trace_forward_layers.py" \
  --fixture-dir "$FIXTURE_DIR" \
  --model-dir "$MODEL_DIR" \
  --atari-dir "$ATARI_DIR" \
  --out "$PYTORCH_FORWARD_OUT" \
  --tensor-dir "$PYTORCH_TENSOR_DIR" \
  --architecture-manifest "$ARCHITECTURE_MANIFEST"

echo
echo "=== Stage 5A: DeepMind layer trace ==="
(
  cd "$DM_DIR/dqn"
  "$LUAJIT_BIN" "$AUDIT_DIR/deepmind/trace_forward_layers.lua"
)

echo
echo "=== Stage 5A: compare every forward layer ==="
set +e
"$PYTHON_BIN" "$AUDIT_DIR/compare_forward_layers.py" \
  --pytorch "$PYTORCH_FORWARD_OUT" \
  --deepmind "$DEEPMIND_FORWARD_OUT" \
  --pytorch-tensor-dir "$PYTORCH_TENSOR_DIR" \
  --deepmind-tensor-dir "$DEEPMIND_TENSOR_DIR" \
  --report "$INTERMEDIATE_COMPARE_OUT" \
  --jsonl "$INTERMEDIATE_COMPARE_JSONL"
compare_status=$?
set -e

if [[ "$compare_status" -eq 0 ]]; then
  write_learner_compare "MATCH"
  write_summary "PASS" "weight exchange, architecture, intermediate activations, final Q-values, and argmax actions match within tolerance"
  sed -n '1,220p' "$SUMMARY_OUT"
  echo
  echo "MATCH"
else
  write_learner_compare "MISMATCH"
  write_summary "MISMATCH" "intermediate forward comparison failed"
  sed -n '1,220p' "$SUMMARY_OUT"
  echo
  echo "STAGE5A_MISMATCH"
  exit "$compare_status"
fi
