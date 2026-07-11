#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/audit_config.sh"

ACTION_REPEAT_DIAGNOSIS_OUT="${ACTION_REPEAT_DIAGNOSIS_OUT:-$OUT/action_repeat_diagnosis.txt}"
export OUT ACTION_REPEAT_DIAGNOSIS_OUT

echo "=== compare ==="
if [[ -f "$OUT/env_compare.txt" ]]; then
  cat "$OUT/env_compare.txt"
else
  echo "missing $OUT/env_compare.txt"
fi

echo
echo "=== action/repeat diagnosis ==="
"$PYTHON_BIN" "$AUDIT_DIR/diagnose_action_repeat.py" \
  --pytorch "$OUT/pytorch_env.jsonl" \
  --deepmind "$OUT/deepmind_env.jsonl" \
  --step 0 \
  --out "$ACTION_REPEAT_DIAGNOSIS_OUT"
