#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/thesis"
batch_dir="outputs/resource_island_v1_batch"
mkdir -p "$batch_dir"

# The original launcher used nohup through SSH.  The remote process is safe,
# but the SSH wrapper waits for its child shell.  Remove only those wrappers;
# the already-running PPO and old1 DQN processes are intentionally preserved.
pkill -f '^ssh -F /dev/null uace@100\.80\.3\.43 .*resource_island_v1_dqn_full' || true
pkill -f '^bash start_resource_island_v1_batch\.sh$' || true
sleep 2

if ! ssh -F /dev/null uace@100.80.3.43 pgrep -f resource_island_v1_dqn_full >/dev/null; then
  echo "old1 DQN did not survive detachment" >&2
  exit 1
fi
if ! pgrep -f resource_island_v1_ppo_full >/dev/null; then
  echo "ofi1 PPO is not running" >&2
  exit 1
fi

if ! ssh -F /dev/null uace@100.118.75.20 pgrep -f '[r]un_resource_island_smoke.py --mind centralized_critic --steps 40000' >/dev/null; then
  ssh -F /dev/null uace@100.118.75.20 '
    cd ~/thesis
    mkdir -p outputs/resource_island_v1_batch
    setsid -f bash -lc "cd ~/thesis; exec bash run_resource_island_v1_exact.sh centralized_critic 40000 20 1000 outputs/resource_island_v1_centralized_critic_full > outputs/resource_island_v1_batch/full_centralized_critic.log 2>&1"
  '
fi

if ! pgrep -f finalize_resource_island_v1_batch.sh >/dev/null; then
  setsid -f bash -lc 'cd ~/thesis; exec bash finalize_resource_island_v1_batch.sh'
fi

echo "$(date -Is) full batch and finalizer launched after explicit detach" >>"$batch_dir/status.log"
echo "launch recovery complete"
