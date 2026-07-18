#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/freeze_config.sh"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$ATARI_DIR/audit_outputs}"
STAGE5_BELLMAN_DIR="${STAGE5_BELLMAN_DIR:-$BASE_OUTPUT_DIR/stage5_bellman}"
OUT="${OUT:-$BASE_OUTPUT_DIR/stage5_loss}"
FIXTURE_JSON="$OUT/loss_fixture.json"
FIXTURE_TSV="$OUT/loss_fixture.tsv"
THRESHOLD_CONTROLS="$OUT/loss_threshold_controls.tsv"
PYTORCH_OUT="$OUT/pytorch_loss.jsonl"
DEEPMIND_OUT="$OUT/deepmind_loss.jsonl"
COMPARE_OUT="$OUT/loss_compare.txt"
SOURCE_REPORT="$OUT/deepmind_loss_source.txt"
SUMMARY_OUT="$OUT/STAGE5C_SUMMARY.md"
TEST_OUT="$OUT/loss_test.txt"
FLOAT_TOL="${STAGE5C_FLOAT_TOL:-1e-7}"

mkdir -p "$OUT"

export AUDIT_DIR ATARI_DIR DM_DIR OUT PYTHON_BIN LUAJIT_BIN
export BASE_OUTPUT_DIR STAGE5_BELLMAN_DIR
export STAGE5C_FIXTURE_TSV="$FIXTURE_TSV"
export DEEPMIND_LOSS_OUT="$DEEPMIND_OUT"
export ACTION_COUNT="${ACTION_COUNT:-4}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing $label: $path" >&2
    exit 2
  fi
}

write_source_report() {
  {
    echo "# DeepMind Stage 5C Loss Source Report"
    echo
    echo "## Source Search"
    echo
    echo '```text'
    grep -RIn "clip_delta\|delta\|td_err\|criterion\|backward" "$DM_DIR" | head -160 || true
    echo '```'
    echo
    echo "## NeuralQLearner:getQUpdate"
    echo
    echo '```lua'
    sed -n '180,235p' "$DM_DIR/dqn/NeuralQLearner.lua"
    echo '```'
    echo
    echo "## NeuralQLearner:qLearnMinibatch"
    echo
    echo '```lua'
    sed -n '247,315p' "$DM_DIR/dqn/NeuralQLearner.lua"
    echo '```'
    echo
    echo "## Findings"
    echo
    printf '%s\n' '- Raw TD error convention: `target - Q(s,a)`.'
    printf '%s\n' '- `target` is `r + gamma * (1 - terminal) * max_a Q_target(s2,a)` before clipping.'
    printf '%s\n' '- `clip_delta=1` clamps the TD error in place to `[-1, 1]`.'
    printf '%s\n' '- No explicit Huber scalar criterion is computed in the released learner path.'
    printf '%s\n' '- The clamped TD error is written into a sparse `targets` tensor with nonzero value only at the selected action.'
    printf '%s\n' '- `network:backward(s, targets)` receives that sparse clipped-delta tensor directly.'
    printf '%s\n' '- There is no division by minibatch size in `getQUpdate` or before `network:backward`.'
    printf '%s\n' '- The optimizer update is outside Stage 5C and is not executed by this runner.'
  } > "$SOURCE_REPORT"
}

write_summary() {
  local status="$1"
  local reason="$2"
  {
    echo "# Stage 5C TD Error Clipping / Loss Semantics Summary"
    echo
    echo "- Status: $status"
    echo "- Reason: $reason"
    echo "- Stage 5B source: $STAGE5_BELLMAN_DIR"
    echo "- Fixture: $FIXTURE_JSON"
    echo "- Threshold controls: $THRESHOLD_CONTROLS"
    echo "- DeepMind source report: $SOURCE_REPORT"
    echo "- PyTorch loss trace: $PYTORCH_OUT"
    echo "- DeepMind loss trace: $DEEPMIND_OUT"
    echo "- Compare report: $COMPARE_OUT"
    echo "- Regression test report: $TEST_OUT"
    echo "- Float tolerance: $FLOAT_TOL"
    echo
    if [[ -f "$SOURCE_REPORT" ]]; then
      echo "## Source Findings"
      echo
      sed -n '/## Findings/,$p' "$SOURCE_REPORT"
      echo
    fi
    if [[ -f "$COMPARE_OUT" ]]; then
      echo "## Compare"
      echo
      echo '```text'
      sed -n '1,160p' "$COMPARE_OUT"
      echo '```'
      echo
    fi
    if [[ -f "$TEST_OUT" ]]; then
      echo "## Regression Test"
      echo
      echo '```text'
      sed -n '1,160p' "$TEST_OUT"
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
samples = [row for row in rows if row.get("phase") == "loss_sample"]
batches = [row for row in rows if row.get("phase") == "loss_batch"]
batch_names = Counter(row["batch_name"] for row in samples)
threshold = [row for row in samples if row.get("sample_kind", "").startswith("synthetic_threshold")]
print(f"- Sample rows: {len(samples)}")
print(f"- Batch rows: {len(batches)}")
print(f"- Batch names: {', '.join(sorted(batch_names))}")
print(f"- Threshold-control samples: {len(threshold)}")
print(f"- Clipped samples: {sum(1 for row in samples if abs(float(row['raw_td_error'])) > 1.0)}")
print(f"- At-threshold samples: {sum(1 for row in samples if row.get('at_clip_threshold'))}")
print(f"- Unselected-action-zero checks true: {sum(1 for row in samples if row.get('unselected_actions_zero'))}")
PY
      echo
    fi
  } > "$SUMMARY_OUT"
}

require_file "$STAGE5_BELLMAN_DIR/STAGE5B_SUMMARY.md" "Stage 5B summary"
require_file "$STAGE5_BELLMAN_DIR/pytorch_bellman.jsonl" "Stage 5B PyTorch Bellman trace"
if ! grep -q "Status: PASS" "$STAGE5_BELLMAN_DIR/STAGE5B_SUMMARY.md"; then
  echo "Stage 5B summary does not report PASS: $STAGE5_BELLMAN_DIR/STAGE5B_SUMMARY.md" >&2
  exit 2
fi

echo "=== Stage 5C: inspect DeepMind loss/clipping source ==="
write_source_report
sed -n '/## Findings/,$p' "$SOURCE_REPORT"

echo
echo "=== Stage 5C: build real and synthetic loss fixtures ==="
"$PYTHON_BIN" "$AUDIT_DIR/build_stage5c_loss_fixture.py" \
  --stage5b-dir "$STAGE5_BELLMAN_DIR" \
  --out-dir "$OUT"

echo
echo "=== Stage 5C: PyTorch loss trace ==="
"$PYTHON_BIN" "$AUDIT_DIR/pytorch/trace_loss.py" \
  --fixture-tsv "$FIXTURE_TSV" \
  --out "$PYTORCH_OUT" \
  --action-count "$ACTION_COUNT"

echo
echo "=== Stage 5C: DeepMind loss trace through NeuralQLearner:getQUpdate ==="
(
  cd "$DM_DIR/dqn"
  "$LUAJIT_BIN" "$AUDIT_DIR/deepmind/trace_loss.lua"
)

echo
echo "=== Stage 5C: compare clipped TD-error and output-gradient contracts ==="
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

echo
echo "=== Stage 5C: regression tests ==="
set +e
AUDIT_STAGE5C_DIR="$OUT" "$PYTHON_BIN" -m unittest audit.tests.test_loss_match -v > "$TEST_OUT" 2>&1
test_status=$?
set -e

if [[ "$compare_status" -eq 0 && "$test_status" -eq 0 ]]; then
  write_summary "PASS" "raw TD errors, clipping, sparse selected-action output gradients, and threshold controls match DeepMind"
  sed -n '1,240p' "$SUMMARY_OUT"
  echo
  echo "MATCH"
else
  write_summary "MISMATCH" "loss/clipped-delta comparison or regression test failed"
  sed -n '1,240p' "$SUMMARY_OUT"
  echo
  echo "STAGE5C_MISMATCH"
  if [[ "$compare_status" -ne 0 ]]; then exit "$compare_status"; fi
  exit "$test_status"
fi
