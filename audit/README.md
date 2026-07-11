# Atari DQN Divergence Audit

This package compares boundary contracts between a PyTorch DQN implementation and
the original DeepMind Torch7/Lua Atari DQN code. It is intentionally staged:
environment first, then preprocessing, replay, learner math, and schedule.

Do not interpret later-stage mismatches until earlier stages are understood.

## Stage 1: Environment Trace

Stage 1 answers one question:

```text
Given the same action tape, do both systems see the same game?
```

It writes:

- `audit_outputs/action_tape_seed${SEED}_${TRACE_STEPS}.npy`
- `audit_outputs/action_tape_seed${SEED}_${TRACE_STEPS}.txt`
- `audit_outputs/pytorch_env.jsonl`
- `audit_outputs/deepmind_env.jsonl`
- `audit_outputs/env_compare.txt`

Run it from the Atari repo root:

```bash
bash audit/run_stage_1_env.sh
```

Override paths if needed:

```bash
ATARI_DIR=/home/uace/dqn/atari \
DM_DIR=/home/uace/dqn/reference/DeepMind-Atari-Deep-Q-Learner \
ROM=/home/uace/dqn/reference/DeepMind-Atari-Deep-Q-Learner/roms/breakout.bin \
bash audit/run_stage_1_env.sh
```

The comparator prints either `MATCH` or a `FIRST MISMATCH` block with the event
index, dotted field path, and left/right values.

Common Stage 1 failures:

- wrong ROM
- different action set or action indexing
- random no-op start enabled on one side
- sticky actions enabled on one side
- frame skip or max-pooling mismatch
- ALE/Gymnasium version mismatch

## Files

- `audit_config.sh`: shared paths and constants.
- `common.py`: Python JSONL, hashing, and array summary helpers.
- `make_action_tape.py`: deterministic action tape generator.
- `compare_jsonl.py`: first-difference comparator.
- `pytorch/trace_env.py`: Gymnasium/ALE environment tracer.
- `deepmind/trace_env.lua`: Torch7/Lua `alewrap` environment tracer.
- `run_stage_1_env.sh`: runs Stage 1 end to end.

## Lua Action Mapping

The Lua tracer defaults to passing tape actions as 1-indexed action indices:

```bash
LUA_ACTION_OFFSET=1 DM_ACTION_MODE=index bash audit/run_stage_1_env.sh
```

If the DeepMind wrapper expects ALE action constants instead, use:

```bash
DM_ACTION_MODE=ale bash audit/run_stage_1_env.sh
```

Stage 1 is expected to expose this kind of mismatch early.
