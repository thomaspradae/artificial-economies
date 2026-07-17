#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/thesis"
batch_dir="outputs/resource_island_v1_batch"
mkdir -p "$batch_dir"
log="$batch_dir/finalizer.log"
exec >>"$log" 2>&1

echo "$(date -Is) finalizer started"

remote_complete() {
  local host="$1"
  local directory="$2"
  ssh -F /dev/null "uace@$host" "test -f ~/thesis/$directory/summary_by_seed.csv -a -f ~/thesis/$directory/summary_aggregate.csv -a -f ~/thesis/$directory/experiment_manifest.json"
}

remote_running() {
  local host="$1"
  local pattern="$2"
  ssh -F /dev/null "uace@$host" "pgrep -f '$pattern' >/dev/null"
}

fail_batch() {
  echo "$(date -Is) FAIL: $1" | tee "$batch_dir/FAILED"
  exit 1
}

while true; do
  ppo=0; dqn=0; independent=0; critic=0
  test -f outputs/resource_island_v1_ppo_full/summary_aggregate.csv && ppo=1
  remote_complete 100.80.3.43 outputs/resource_island_v1_dqn_full && dqn=1 || true
  remote_complete 100.80.3.43 outputs/resource_island_v1_independent_dqn_full && independent=1 || true
  remote_complete 100.118.75.20 outputs/resource_island_v1_centralized_critic_full && critic=1 || true
  echo "$(date -Is) complete ppo=$ppo dqn=$dqn independent_dqn=$independent centralized_critic=$critic"
  if [ "$ppo$dqn$independent$critic" = "1111" ]; then
    break
  fi
  if [ "$ppo" -eq 0 ] && ! pgrep -f '[r]un_resource_island_smoke.py --mind ppo --steps 40000' >/dev/null; then
    fail_batch "PPO stopped before producing complete artifacts"
  fi
  if [ "$dqn" -eq 0 ] && ! remote_running 100.80.3.43 '[r]un_resource_island_smoke.py --mind dqn --steps 40000'; then
    fail_batch "DQN stopped before producing complete artifacts"
  fi
  if [ "$dqn" -eq 1 ] && [ "$independent" -eq 0 ] && ! remote_running 100.80.3.43 '[r]un_resource_island_smoke.py --mind independent_dqn --steps 40000'; then
    fail_batch "independent-DQN did not start or stopped before producing complete artifacts"
  fi
  if [ "$critic" -eq 0 ] && ! remote_running 100.118.75.20 '[r]un_resource_island_smoke.py --mind centralized_critic --steps 40000'; then
    fail_batch "centralized critic stopped before producing complete artifacts"
  fi
  sleep 60
done

rsync -a uace@100.80.3.43:~/thesis/outputs/resource_island_v1_dqn_full/ outputs/resource_island_v1_dqn_full/
rsync -a uace@100.80.3.43:~/thesis/outputs/resource_island_v1_independent_dqn_full/ outputs/resource_island_v1_independent_dqn_full/
rsync -a uace@100.118.75.20:~/thesis/outputs/resource_island_v1_centralized_critic_full/ outputs/resource_island_v1_centralized_critic_full/

source .venv/bin/activate
python validate_resource_island_v1_gate.py \
  --steps 40000 --n-seeds 20 \
  --result dqn:outputs/resource_island_v1_dqn_full \
  --result ppo:outputs/resource_island_v1_ppo_full \
  --result independent_dqn:outputs/resource_island_v1_independent_dqn_full \
  --result centralized_critic:outputs/resource_island_v1_centralized_critic_full \
  --output "$batch_dir/full_validation.json"
python build_world_mind_comparison.py --world resource_island
python validate_full_ladders.py
python build_cross_world_synthesis.py --save-dir outputs/cross_world_synthesis
python build_publication_inference.py
echo "$(date -Is) DONE" | tee "$batch_dir/DONE"
