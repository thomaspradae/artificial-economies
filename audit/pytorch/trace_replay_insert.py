#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_state_stack import build_state_rows, load_transitions  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs/stage3_replay")
    parser = argparse.ArgumentParser(description="Trace PyTorch-side semantic replay insertion from canonical processed frames.")
    parser.add_argument("--tape-dir", default=os.path.join(out, "canonical"))
    parser.add_argument("--out", default=os.path.join(out, "pytorch_replay_insert.jsonl"))
    parser.add_argument("--hist-len", type=int, default=4)
    return parser.parse_args()


def sampleability(prev_state: dict[str, Any], next_state: dict[str, Any], terminal_mask: bool) -> tuple[bool, str]:
    if prev_state["reset_zero_padding"]:
        return False, "state_has_zero_padding"
    if next_state["reset_zero_padding"]:
        return False, "next_state_has_zero_padding"
    if terminal_mask:
        return False, "stored_transition_terminal"
    return True, "sampleable"


def build_replay_rows(transitions: list[dict[str, Any]], source: str = "pytorch", hist_len: int = 4) -> list[dict[str, Any]]:
    state_rows = build_state_rows(transitions, source=source, hist_len=hist_len)
    by_index = {int(row["transition_index"]): row for row in state_rows}
    rows: list[dict[str, Any]] = []

    for current_index in range(1, len(transitions)):
        previous_index = current_index - 1
        previous = transitions[previous_index]
        current = transitions[current_index]
        prev_state = by_index[previous_index]
        next_state = by_index[current_index]
        terminal_mask_stored = bool(previous["terminal_mask"])
        considered_sampleable, reason = sampleability(prev_state, next_state, terminal_mask_stored)
        rows.append(
            {
                "phase": "replay_insert",
                "source": source,
                "step": previous_index,
                "transition_index": previous_index,
                "perceive_index": current_index,
                "replay_insertion_index": previous_index,
                "frame_index_stored": int(previous["frame_index"]),
                "action_stored": int(previous["action_index"]),
                "ale_action_code_stored": int(previous["ale_action_code"]),
                "reward_source_transition_index": current_index,
                "raw_reward_stored": float(current["raw_reward"]),
                "reward_stored": int(current["clipped_reward"]),
                "terminal_stored": bool(previous["true_terminal"]),
                "life_loss_terminal_stored": bool(previous["life_loss"]),
                "terminal_mask_stored": terminal_mask_stored,
                "state_hash": prev_state["state_hash"],
                "state_component_hashes": prev_state["component_frame_hashes"],
                "state_component_frame_indices": prev_state["component_frame_indices"],
                "state_shape": prev_state["state_shape"],
                "state_dtype": prev_state["state_dtype"],
                "next_state_hash": next_state["state_hash"],
                "next_state_component_hashes": next_state["component_frame_hashes"],
                "next_state_component_frame_indices": next_state["component_frame_indices"],
                "next_state_shape": next_state["state_shape"],
                "next_state_dtype": next_state["state_dtype"],
                "replay_size_after_insertion": current_index,
                "considered_sampleable": considered_sampleable,
                "sampleability_reason": reason,
                "episode_boundary": bool(prev_state["episode_boundary"] or next_state["episode_boundary"]),
                "life_loss_boundary": bool(prev_state["life_loss_boundary"] or next_state["life_loss_boundary"]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = build_replay_rows(load_transitions(args.tape_dir), source="pytorch", hist_len=args.hist_len)
    write_jsonl(args.out, rows)
    print(f"wrote {args.out}")
    print(f"replay_insert_rows: {len(rows)}")


if __name__ == "__main__":
    main()
