#!/usr/bin/env bash

# Shared configuration for frozen-input DQN audits.

AUDIT_DIR="${AUDIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ATARI_DIR="${ATARI_DIR:-$(cd "$AUDIT_DIR/.." && pwd)}"

DM_DIR="${DM_DIR:-/home/uace/dqn/reference/DeepMind-Atari-Deep-Q-Learner}"
ROM="${ROM:-$DM_DIR/roms/breakout.bin}"
ENV_ID="${ENV_ID:-ALE/Breakout-v5}"

SEED="${SEED:-1}"
FRAME_SKIP="${FRAME_SKIP:-4}"
TRACE_STEPS="${TRACE_STEPS:-64}"
ACTION_COUNT="${ACTION_COUNT:-4}"
ACTION_TAPE_MODE="${ACTION_TAPE_MODE:-index}"
ACTION_TAPE_SEQUENCE="${ACTION_TAPE_SEQUENCE:-}"
ACTION_TAPE_VALUES="${ACTION_TAPE_VALUES:-0,1,3,4}"

OUT="${OUT:-$ATARI_DIR/audit_outputs}"
CANONICAL_TAPE_DIR="${CANONICAL_TAPE_DIR:-$OUT/canonical_frames}"

PYTHON_BIN="${PYTHON_BIN:-python}"
LUAJIT_BIN="${LUAJIT_BIN:-luajit}"
PYTORCH_RESIZE_INTERPOLATION="${PYTORCH_RESIZE_INTERPOLATION:-area}"

DM_ACTION_MODE="${DM_ACTION_MODE:-ale}"
LUA_ACTION_OFFSET="${LUA_ACTION_OFFSET:-1}"

FLOAT_TOL="${FLOAT_TOL:-1e-6}"
