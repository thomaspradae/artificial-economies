#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/freeze_config.sh"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$ATARI_DIR/audit_outputs}"
STAGE4_REPLAY_SAMPLE_DIR="${STAGE4_REPLAY_SAMPLE_DIR:-$BASE_OUTPUT_DIR/stage4_replay_sample}"
STAGE5_LEARNER_DIR="${STAGE5_LEARNER_DIR:-$BASE_OUTPUT_DIR/stage5_learner}"
OUT="${OUT:-$BASE_OUTPUT_DIR/stage5_bellman}"
MODEL_DIR="${STAGE5_MODEL_DIR:-$STAGE5_LEARNER_DIR/model_exchange}"

PYTORCH_OUT="$OUT/pytorch_bellman.jsonl"
DEEPMIND_OUT="$OUT/deepmind_bellman.jsonl"
SPEC_OUT="$OUT/bellman_batches.tsv"
COMPARE_OUT="$OUT/bellman_compare.txt"
SUMMARY_OUT="$OUT/STAGE5B_SUMMARY.md"
FLOAT_TOL="${STAGE5B_FLOAT_TOL:-1e-7}"

mkdir -p "$OUT"

export AUDIT_DIR ATARI_DIR DM_DIR OUT PYTHON_BIN LUAJIT_BIN
export BASE_OUTPUT_DIR STAGE4_REPLAY_SAMPLE_DIR STAGE5_MODEL_DIR="$MODEL_DIR"
export STAGE5B_BATCH_SPEC="$SPEC_OUT"
export DEEPMIND_BELLMAN_OUT="$DEEPMIND_OUT"
export ACTION_COUNT="${ACTION_COUNT:-4}" GAMMA="${GAMMA:-0.99}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing $label: $path" >&2
    exit 2
  fi
}

write_summary() {
  local status="$1"
  local reason="$2"
  {
    echo "# Stage 5B Bellman Target Summary"
    echo
    echo "- Status: $status"
    echo "- Reason: $reason"
    echo "- Stage 4 source: $STAGE4_REPLAY_SAMPLE_DIR"
    echo "- Stage 5A source: $STAGE5_LEARNER_DIR"
    echo "- Model exchange: $MODEL_DIR"
    echo "- PyTorch Bellman trace: $PYTORCH_OUT"
    echo "- DeepMind Bellman trace: $DEEPMIND_OUT"
    echo "- Batch spec: $SPEC_OUT"
    echo "- Compare report: $COMPARE_OUT"
    echo "- Float tolerance: $FLOAT_TOL"
    echo
    if [[ -f "$COMPARE_OUT" ]]; then
      echo "## Compare"
      echo
      echo '```text'
      sed -n '1,160p' "$COMPARE_OUT"
      echo '```'
      echo
    fi
    if [[ -f "$PYTORCH_OUT" ]]; then
      echo "## Coverage"
      echo
      "$PYTHON_BIN" - "$PYTORCH_OUT" <<'PY'
import json
import sys
from collections import Counter

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
samples = [row for row in rows if row.get("phase") == "bellman_sample"]
batches = [row for row in rows if row.get("phase") == "bellman_batch"]
names = Counter(row["batch_name"] for row in samples)
print(f"- Sample rows: {len(samples)}")
print(f"- Batch rows: {len(batches)}")
print(f"- Batch names: {', '.join(sorted(names))}")
print(f"- True terminal samples: {sum(1 for row in samples if row.get('true_terminal'))}")
print(f"- Life-loss terminal samples: {sum(1 for row in samples if row.get('life_loss_terminal'))}")
print(f"- Zero-reward samples: {sum(1 for row in samples if row.get('reward') == 0)}")
print(f"- Positive-reward samples: {sum(1 for row in samples if row.get('reward', 0) > 0)}")
print(f"- Terminal-target checks true: {sum(1 for row in samples if row.get('terminal_target_equals_reward'))}")
print(f"- Nonterminal gamma checks true: {sum(1 for row in samples if row.get('nonterminal_gamma_applied'))}")
PY
      echo
    fi
  } > "$SUMMARY_OUT"
}

require_file "$STAGE4_REPLAY_SAMPLE_DIR/canonical_replay/records.jsonl" "Stage 4 canonical replay"
require_file "$STAGE4_REPLAY_SAMPLE_DIR/requested_indices.txt" "Stage 4 requested indices"
require_file "$STAGE5_LEARNER_DIR/STAGE5A_SUMMARY.md" "Stage 5A summary"
require_file "$MODEL_DIR/deepmind_model_manifest.json" "Stage 5A DeepMind model manifest"

if ! grep -q "Status: PASS" "$STAGE5_LEARNER_DIR/STAGE5A_SUMMARY.md"; then
  echo "Stage 5A summary does not report PASS: $STAGE5_LEARNER_DIR/STAGE5A_SUMMARY.md" >&2
  exit 2
fi

echo "=== Stage 5B: PyTorch Bellman trace and batch materialization ==="
"$PYTHON_BIN" "$AUDIT_DIR/pytorch/trace_bellman.py" \
  --stage4-dir "$STAGE4_REPLAY_SAMPLE_DIR" \
  --model-dir "$MODEL_DIR" \
  --atari-dir "$ATARI_DIR" \
  --out-dir "$OUT" \
  --out "$PYTORCH_OUT" \
  --spec-out "$SPEC_OUT" \
  --gamma "$GAMMA"

echo
echo "=== Stage 5B: DeepMind Bellman trace through NeuralQLearner:getQUpdate ==="
(
  cd "$DM_DIR/dqn"
  "$LUAJIT_BIN" "$AUDIT_DIR/deepmind/trace_bellman.lua"
)

echo
echo "=== Stage 5B: compare Bellman target traces ==="
set +e
"$PYTHON_BIN" "$AUDIT_DIR/compare_jsonl.py" \
  --left "$PYTORCH_OUT" \
  --right "$DEEPMIND_OUT" \
  --left-label pytorch \
  --right-label deepmind \
  --float-tol "$FLOAT_TOL" \
  > "$COMPARE_OUT"
compare_status=$?
set -e

if [[ "$compare_status" -eq 0 ]]; then
  write_summary "PASS" "Bellman targets, masks, maximizing actions, selected Q-values, and target-network next-Q values match within tolerance"
  sed -n '1,220p' "$SUMMARY_OUT"
  echo
  echo "MATCH"
else
  write_summary "MISMATCH" "Bellman comparison found a mismatch"
  sed -n '1,220p' "$SUMMARY_OUT"
  echo
  echo "STAGE5B_MISMATCH"
  exit "$compare_status"
fi
