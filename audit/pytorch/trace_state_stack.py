#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import arr_hash, arr_stats, read_jsonl, write_jsonl  # noqa: E402


HIST_LEN = 4
FRAME_SHAPE = (84, 84)


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs/stage3_replay")
    parser = argparse.ArgumentParser(description="Trace PyTorch-side state stack construction from canonical processed frames.")
    parser.add_argument("--tape-dir", default=os.path.join(out, "canonical"))
    parser.add_argument("--out", default=os.path.join(out, "pytorch_state_stack.jsonl"))
    parser.add_argument("--hist-len", type=int, default=HIST_LEN)
    return parser.parse_args()


def load_transitions(tape_dir: str | Path) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_jsonl(Path(tape_dir) / "transitions.jsonl")
        if row.get("phase") == "canonical_processed_transition"
    ]
    rows.sort(key=lambda row: int(row["transition_index"]))
    return rows


def load_frame(path: str | Path) -> np.ndarray:
    frame = np.asarray(np.load(path))
    if frame.shape != FRAME_SHAPE:
        raise ValueError(f"expected {FRAME_SHAPE}, got {frame.shape}: {path}")
    return np.ascontiguousarray(frame.astype(np.uint8, copy=False))


def zero_record(zero_frame: np.ndarray) -> dict[str, Any]:
    return {
        "frame_index": None,
        "frame": zero_frame,
        "hash": arr_hash(zero_frame),
        "true_terminal": True,
        "life_loss": False,
        "terminal_mask": True,
        "padding": True,
    }


def stack_from_recent(recent: list[dict[str, Any]], zero_frame: np.ndarray) -> dict[str, Any]:
    zero_out = False
    zero_flags = [False] * len(recent)
    for index in range(len(recent) - 2, -1, -1):
        if not zero_out and bool(recent[index]["terminal_mask"]):
            zero_out = True
        if zero_out:
            zero_flags[index] = True

    component_frames: list[np.ndarray] = []
    component_indices: list[int | None] = []
    component_source_indices: list[int | None] = []
    component_hashes: list[str] = []
    component_source_hashes: list[str] = []
    terminal_boundaries: list[bool] = []
    life_loss_boundaries: list[bool] = []
    true_terminal_boundaries: list[bool] = []

    for index, item in enumerate(recent):
        source_index = item["frame_index"]
        component_source_indices.append(source_index)
        component_source_hashes.append(item["hash"])
        terminal_boundaries.append(bool(item["terminal_mask"]))
        life_loss_boundaries.append(bool(item["life_loss"]))
        true_terminal_boundaries.append(bool(item["true_terminal"]))
        if zero_flags[index] or source_index is None:
            frame = zero_frame
            component_indices.append(None)
        else:
            frame = item["frame"]
            component_indices.append(int(source_index))
        component_frames.append(frame)
        component_hashes.append(arr_hash(frame))

    stack = np.ascontiguousarray(np.stack(component_frames, axis=0))
    return {
        "stack": stack,
        "component_frame_indices": component_indices,
        "component_source_frame_indices": component_source_indices,
        "component_frame_hashes": component_hashes,
        "component_source_frame_hashes": component_source_hashes,
        "reset_zero_padding": any(index is None for index in component_indices),
        "zeroed_component_positions": [idx for idx, value in enumerate(component_indices) if value is None],
        "episode_boundary": any(terminal_boundaries[:-1]),
        "life_loss_boundary": any(life_loss_boundaries[:-1]),
        "true_terminal_boundary": any(true_terminal_boundaries[:-1]),
    }


def build_state_rows(transitions: list[dict[str, Any]], source: str = "pytorch", hist_len: int = HIST_LEN) -> list[dict[str, Any]]:
    if hist_len != HIST_LEN:
        raise ValueError("Stage 3 currently implements DeepMind hist_len=4 only")
    zero_frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    recent = [zero_record(zero_frame) for _ in range(hist_len)]
    rows: list[dict[str, Any]] = []

    for transition in transitions:
        frame = load_frame(transition["frame_path"])
        recent.append(
            {
                "frame_index": int(transition["frame_index"]),
                "frame": frame,
                "hash": arr_hash(frame),
                "true_terminal": bool(transition["true_terminal"]),
                "life_loss": bool(transition["life_loss"]),
                "terminal_mask": bool(transition["terminal_mask"]),
                "padding": False,
            }
        )
        recent = recent[-hist_len:]
        stack_info = stack_from_recent(recent, zero_frame)
        stack = stack_info.pop("stack")
        rows.append(
            {
                "phase": "state_stack",
                "source": source,
                "step": int(transition["transition_index"]),
                "transition_index": int(transition["transition_index"]),
                "frame_index": int(transition["frame_index"]),
                "current_frame_hash": transition["processed_frame_hash"],
                "stack_order": "oldest_to_newest",
                "hist_len": hist_len,
                "state_shape": [int(dim) for dim in stack.shape],
                "state_dtype": str(stack.dtype),
                "state_hash": arr_hash(stack),
                "state_frame": arr_stats(stack),
                **stack_info,
            }
        )

    for index, row in enumerate(rows):
        if index + 1 < len(rows):
            row["next_state_component_hashes"] = rows[index + 1]["component_frame_hashes"]
            row["next_state_hash"] = rows[index + 1]["state_hash"]
            row["next_state_shape"] = rows[index + 1]["state_shape"]
            row["next_state_dtype"] = rows[index + 1]["state_dtype"]
        else:
            row["next_state_component_hashes"] = None
            row["next_state_hash"] = None
            row["next_state_shape"] = None
            row["next_state_dtype"] = None
    return rows


def main() -> None:
    args = parse_args()
    rows = build_state_rows(load_transitions(args.tape_dir), source="pytorch", hist_len=args.hist_len)
    write_jsonl(args.out, rows)
    print(f"wrote {args.out}")
    print(f"state_stack_rows: {len(rows)}")


if __name__ == "__main__":
    main()
