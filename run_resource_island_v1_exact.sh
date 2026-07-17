#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "usage: $0 MIND STEPS N_SEEDS FINAL_WINDOW SAVE_DIR" >&2
  exit 2
fi

mind="$1"
steps="$2"
n_seeds="$3"
final_window="$4"
save_dir="$5"

case "$mind" in
  dqn|ppo|independent_dqn|centralized_critic) ;;
  *) echo "unsupported mind: $mind" >&2; exit 2 ;;
esac

source .venv/bin/activate
python -u run_resource_island_smoke.py \
  --mind "$mind" \
  --steps "$steps" \
  --n-seeds "$n_seeds" \
  --final-window "$final_window" \
  --resource-layout contested \
  --activation-preset pressure \
  --specialization-preset complementary \
  --trade-food-units 2 \
  --trade-wood-units 1 \
  --trade-acquisition-reward 0.2 \
  --trade-radius 8 \
  --institutions none property_rights trade_price_controls reputation_system \
  --save-dir "$save_dir"
