#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/thesis"
batch_dir="outputs/resource_island_v1_batch"
mkdir -p "$batch_dir"

echo "$(date -Is) smoke gates starting" | tee "$batch_dir/status.log"

bash run_resource_island_v1_exact.sh dqn 2000 3 500 outputs/resource_island_v1_dqn_smoke_gate >"$batch_dir/smoke_dqn.log" 2>&1 &
p1=$!
bash run_resource_island_v1_exact.sh ppo 2000 3 500 outputs/resource_island_v1_ppo_smoke_gate >"$batch_dir/smoke_ppo.log" 2>&1 &
p2=$!
ssh -F /dev/null uace@100.80.3.43 'cd ~/thesis && bash run_resource_island_v1_exact.sh independent_dqn 2000 3 500 outputs/resource_island_v1_independent_dqn_smoke_gate' >"$batch_dir/smoke_independent_dqn.log" 2>&1 &
p3=$!
ssh -F /dev/null uace@100.118.75.20 'cd ~/thesis && bash run_resource_island_v1_exact.sh centralized_critic 2000 3 500 outputs/resource_island_v1_centralized_critic_smoke_gate' >"$batch_dir/smoke_centralized_critic.log" 2>&1 &
p4=$!

failed=0
for pid in "$p1" "$p2" "$p3" "$p4"; do
  wait "$pid" || failed=1
done
if [ "$failed" -ne 0 ]; then
  echo "$(date -Is) smoke execution failed; full batch not launched" | tee -a "$batch_dir/status.log"
  exit 1
fi

rsync -a uace@100.80.3.43:~/thesis/outputs/resource_island_v1_independent_dqn_smoke_gate/ outputs/resource_island_v1_independent_dqn_smoke_gate/
rsync -a uace@100.118.75.20:~/thesis/outputs/resource_island_v1_centralized_critic_smoke_gate/ outputs/resource_island_v1_centralized_critic_smoke_gate/

source .venv/bin/activate
python validate_resource_island_v1_gate.py \
  --steps 2000 --n-seeds 3 \
  --result dqn:outputs/resource_island_v1_dqn_smoke_gate \
  --result ppo:outputs/resource_island_v1_ppo_smoke_gate \
  --result independent_dqn:outputs/resource_island_v1_independent_dqn_smoke_gate \
  --result centralized_critic:outputs/resource_island_v1_centralized_critic_smoke_gate \
  --output "$batch_dir/smoke_validation.json" | tee "$batch_dir/smoke_validation.log"

echo "$(date -Is) smoke gates passed; launching full runs" | tee -a "$batch_dir/status.log"

nohup bash -lc 'cd ~/thesis && bash run_resource_island_v1_exact.sh ppo 40000 20 1000 outputs/resource_island_v1_ppo_full' >"$batch_dir/full_ppo.log" 2>&1 < /dev/null &
echo $! >"$batch_dir/full_ppo.pid"

ssh -F /dev/null uace@100.80.3.43 'cd ~/thesis && mkdir -p outputs/resource_island_v1_batch && setsid -f bash -lc "cd ~/thesis; exec > outputs/resource_island_v1_batch/full_old1_queue.log 2>&1; bash run_resource_island_v1_exact.sh dqn 40000 20 1000 outputs/resource_island_v1_dqn_full && exec bash run_resource_island_v1_exact.sh independent_dqn 40000 20 1000 outputs/resource_island_v1_independent_dqn_full"'

ssh -F /dev/null uace@100.118.75.20 'cd ~/thesis && mkdir -p outputs/resource_island_v1_batch && setsid -f bash -lc "cd ~/thesis; exec bash run_resource_island_v1_exact.sh centralized_critic 40000 20 1000 outputs/resource_island_v1_centralized_critic_full > outputs/resource_island_v1_batch/full_centralized_critic.log 2>&1"'

setsid -f bash -lc 'cd ~/thesis; exec bash finalize_resource_island_v1_batch.sh'
echo "$(date -Is) full batch and finalizer launched" | tee -a "$batch_dir/status.log"
