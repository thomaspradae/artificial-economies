#!/usr/bin/env bash

# Shared configuration for the Atari DQN divergence audit.
#
# All values can be overridden by exporting the variable before invoking a
# stage runner.

AUDIT_DIR="${AUDIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ATARI_DIR="${ATARI_DIR:-$(cd "$AUDIT_DIR/.." && pwd)}"

DM_DIR="${DM_DIR:-/home/uace/dqn/reference/DeepMind-Atari-Deep-Q-Learner}"
ENV_ID="${ENV_ID:-ALE/Breakout-v5}"
ROM="${ROM:-$DM_DIR/roms/breakout.bin}"

SEED="${SEED:-1}"
TRACE_STEPS="${TRACE_STEPS:-200}"
FRAME_SKIP="${FRAME_SKIP:-4}"
ACTION_COUNT="${ACTION_COUNT:-4}"
ACTION_TAPE_MODE="${ACTION_TAPE_MODE:-index}"
ACTION_TAPE_VALUES="${ACTION_TAPE_VALUES:-0,1,3,4}"
ACTION_TAPE_SEQUENCE="${ACTION_TAPE_SEQUENCE:-}"
FRAME_DUMP_STEPS="${FRAME_DUMP_STEPS:-20}"

OUT="${OUT:-$ATARI_DIR/audit_outputs}"

PYTHON_BIN="${PYTHON_BIN:-python}"
LUAJIT_BIN="${LUAJIT_BIN:-luajit}"

# DeepMind Lua action mapping. GameEnvironment expects ALE constants from
# env:getActions(); use DM_ACTION_MODE=index only for wrappers that expect
# compact action indices.
DM_ACTION_MODE="${DM_ACTION_MODE:-ale}"
LUA_ACTION_OFFSET="${LUA_ACTION_OFFSET:-1}"

FLOAT_TOL="${FLOAT_TOL:-1e-6}"
COMPARE_IGNORE="${COMPARE_IGNORE:-timestamp,runtime,path,source_path,source,env_id,rom,reset_info,info,dm_action_mode,lua_action_offset,lua_action,ale_action,action_meanings,ale_frame_number,dtype}"
