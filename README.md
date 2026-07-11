That mismatch is the real finding — `ps aux` shows **nothing** for those PIDs, but `status.json` still claims all 4 are `running`. That means the training processes are already dead (very likely OOM-killed during the swap-thrashing period), and the orchestrator either also died or hasn't polled recently enough to notice. You have a zombie status file, not zombie jobs. Let's confirm and clean this up properly, then move to old1/old2.

## Roadmap

### Step 1 — Confirm what's actually alive on ofi1

```bash
# is the orchestrator itself (parent PID 379605) still alive?
ps -p 379605 -f

# any python at all still running?
ps aux | grep python | grep -v grep

# when did status.json last update?
stat outputs/phase3_full/status.json
```

If `ps -p 379605` returns nothing, the orchestrator is dead too — `status.json` is a frozen snapshot from whenever it last wrote, not live state. Either way, the 4 training processes are gone.

### Step 2 — Check how far each dead job actually got, before discarding anything

```bash
for m in dqn ppo independent_dqn centralized_critic; do
  echo "=== $m ==="
  tail -20 outputs/phase3_full/logs/${m}_multiseed.log
  ls -la outputs/${m}_v0_multiseed/ 2>&1
done
```

This tells you if any seeds actually completed and got written to disk (partial CSV rows), or if 12 hours produced nothing durable. Don't assume zero — check.

### Step 3 — Clean up ofi1 fully

```bash
# kill orchestrator if somehow still alive
kill 379605 2>/dev/null

# check resource island too
ps aux | grep resource_island | grep -v grep
```

Confirm `ps aux | grep python` is clean before moving to old1/old2 — you don't want a half-dead process holding memory while you're trying to diagnose whether ofi1 is usable again.

### Step 4 — Reconcile status.json instead of trusting it

Don't just relaunch blind. Decide per-job: **complete / partial-restart / full-restart**, based on Step 2's findings. If zero seeds landed for any job, that's a full restart. If partial output exists but your runner doesn't support resume-from-seed, that's still effectively a full restart — check whether `run_multiseed.py` has a `--resume` or `--skip-existing-seeds` flag before assuming you have to redo everything.

### Step 5 — Deploy code to old1 and old2

```bash
rsync -avz --exclude 'outputs' --exclude '.venv' --exclude '__pycache__' \
  /home/t/Downloads/fogo/thesis/ old1:~/thesis/
rsync -avz --exclude 'outputs' --exclude '.venv' --exclude '__pycache__' \
  /home/t/Downloads/fogo/thesis/ old2:~/thesis/

ssh old1 "cd ~/thesis && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
ssh old2 "cd ~/thesis && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
```

**Important constraint you didn't have on ofi1**: old1/old2 are i5-2400s with only **3.8GB RAM each** (from the neofetch output — "Memory: 526MiB / 3873MiB" and "268MiB / 3803MiB", both nearly empty right now, which is good, but the ceiling is much lower than ofi1's 7.7GB). Run **one job per machine**, not two — you just learned what happens when you overcommit memory on this exact codebase.

### Step 6 — Launch in a controlled, verifiable way (one job per machine, nohup + explicit log)

```bash
ssh old1 "cd ~/thesis && source .venv/bin/activate && \
  nohup python run_multiseed.py --mind dqn --steps 40000 --n-seeds 20 \
  --save-dir outputs/dqn_v0_multiseed > outputs/dqn_multiseed.log 2>&1 & \
  echo \$! > outputs/dqn.pid"

ssh old2 "cd ~/thesis && source .venv/bin/activate && \
  nohup python run_multiseed.py --mind ppo --steps 40000 --n-seeds 20 \
  --save-dir outputs/ppo_v0_multiseed > outputs/ppo_multiseed.log 2>&1 & \
  echo \$! > outputs/ppo.pid"
```

Wait and confirm each actually started and is climbing in CPU-time (not thrashing) before queuing the next job behind it:

```bash
ssh old1 "cat ~/thesis/outputs/dqn.pid; ps -p \$(cat ~/thesis/outputs/dqn.pid) -o pid,%cpu,%mem,etime,cmd"
```

Once `dqn` on old1 is confirmed healthy, queue `independent_dqn` behind it on the same machine (sequential, not parallel) — same for `centralized_critic` behind `ppo` on old2:

```bash
ssh old1 "cd ~/thesis && source .venv/bin/activate && \
  nohup bash -c 'wait \$(cat outputs/dqn.pid); python run_multiseed.py --mind independent_dqn --steps 40000 --n-seeds 20 --save-dir outputs/independent_dqn_v0_multiseed' > outputs/independent_dqn_multiseed.log 2>&1 &"
```

(If `wait` on an arbitrary PID doesn't work in your shell since it's not a child process, simplest fix: just launch it manually once you've confirmed the first job on that box finished, rather than chaining automatically.)

### Step 7 — Verify health repeatedly, not just once

```bash
watch -n30 'ssh old1 "cat ~/thesis/outputs/dqn.pid | xargs -I{} ps -p {} -o pid,%cpu,%mem,etime,cmd"; ssh old2 "cat ~/thesis/outputs/ppo.pid | xargs -I{} ps -p {} -o pid,%cpu,%mem,etime,cmd"'
```

Or simpler, since you already have `poormans` — just watch it and confirm old1/old2's RAM stays well under 100% (not pinned like ofi1 was) and their `TIME+`/CPU%-over-elapsed-time ratio stays close to 1, not the ~10% you saw on ofi1.

### Step 8 — Where exploitability jobs and Resource Island go

Once the 4 multiseed jobs are confirmed running healthily (2 on old1, 2 on old2, sequential per box), the 4 exploitability jobs queue up behind them the same way. Resource Island (cheap, tabular, low memory) can go back onto ofi1 alone now that it's not fighting 4 torch processes for RAM — that machine's fine for one lightweight job.

### Step 9 — When everything's done, pull results back

```bash
rsync -avz old1:~/thesis/outputs/ ~/thesis/outputs/
rsync -avz old2:~/thesis/outputs/ ~/thesis/outputs/
```

Then `build_combined_table.py` runs locally on ofi1 against the merged `outputs/` as before.

Start with Step 1-2 — reconcile what's real before launching anything new, since if any seeds did complete on ofi1 you don't want to burn old1/old2 cycles redoing that work.