#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p outputs

rsync -avz \
  -e "ssh -F /dev/null -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8" \
  uace@100.80.3.43:~/thesis/outputs/ outputs/

ssh -F /dev/null \
  -o BatchMode=yes \
  -o StrictHostKeyChecking=accept-new \
  -o ConnectTimeout=8 \
  uace@100.107.98.78 \
  "ssh -F /dev/null -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 uace@100.118.75.20 'cd ~/thesis && tar -czf - outputs'" \
  | tar -xzf -

echo "Pulled at $(date -Is)" >> outputs/pull_log.txt
