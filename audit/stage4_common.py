from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from common import arr_hash, read_jsonl


HIST_LEN = 4
FRAME_SHAPE = (84, 84)
CAPACITY_DEFAULT = 1024
TOTAL_INSERTIONS_DEFAULT = 1600
ACTION_TO_ALE = [0, 1, 3, 4]


def load_records(replay_dir: str | Path) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_jsonl(Path(replay_dir) / "records.jsonl")
        if row.get("phase") == "stage4_replay_record"
    ]
    rows.sort(key=lambda row: int(row["replay_index"]))
    return rows


def load_requested_indices(path: str | Path) -> list[int]:
    indices: list[int] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indices.append(int(stripped.split()[0]))
    return indices


def load_frame(path: str | Path) -> np.ndarray:
    frame = np.asarray(np.load(path))
    if frame.shape != FRAME_SHAPE:
        raise ValueError(f"expected {FRAME_SHAPE}, got {frame.shape}: {path}")
    return np.ascontiguousarray(frame.astype(np.uint8, copy=False))


def index_validity(records: list[dict[str, Any]], requested_index: int, hist_len: int = HIST_LEN) -> tuple[bool, str]:
    max_start = len(records) - hist_len - 1
    if requested_index < 1:
        return False, "before_earliest_deepmind_sample_index"
    if requested_index > max_start:
        return False, "after_latest_deepmind_sample_index"
    action_record = records[requested_index + hist_len - 1]
    if bool(action_record["terminal_mask"]):
        return False, "terminal_at_action_frame"
    return True, "accepted"


def _zero_frame() -> np.ndarray:
    return np.zeros(FRAME_SHAPE, dtype=np.uint8)


def _stack_components(records: list[dict[str, Any]], start: int, hist_len: int = HIST_LEN) -> dict[str, Any]:
    zero = _zero_frame()
    raw_indices = [start + offset for offset in range(hist_len)]
    zero_flags = [False] * hist_len
    zero_out = False
    for pos in range(hist_len - 2, -1, -1):
        if not zero_out and bool(records[raw_indices[pos]]["terminal_mask"]):
            zero_out = True
        if zero_out:
            zero_flags[pos] = True

    frames: list[np.ndarray] = []
    source_indices: list[int | None] = []
    source_hashes: list[str] = []
    raw_hashes: list[str] = []
    for pos, replay_index in enumerate(raw_indices):
        record = records[replay_index]
        raw_hashes.append(str(record["processed_frame_hash"]))
        if zero_flags[pos]:
            frame = zero
            source_indices.append(None)
            source_hashes.append(arr_hash(zero))
        else:
            frame = load_frame(record["frame_path"])
            source_indices.append(replay_index)
            source_hashes.append(arr_hash(frame))
        frames.append(frame)
    stack = np.ascontiguousarray(np.stack(frames, axis=0))
    return {
        "stack": stack,
        "raw_frame_indices": raw_indices,
        "source_frame_indices": source_indices,
        "source_frame_hashes": source_hashes,
        "raw_frame_hashes": raw_hashes,
        "zeroed_positions": [idx for idx, value in enumerate(source_indices) if value is None],
    }


def sample_at(records: list[dict[str, Any]], requested_index: int, hist_len: int = HIST_LEN) -> dict[str, Any]:
    accepted, reason = index_validity(records, requested_index, hist_len=hist_len)
    base: dict[str, Any] = {
        "requested_replay_index": requested_index,
        "accepted": accepted,
        "rejection_reason": "none" if accepted else reason,
    }
    if not accepted:
        return base

    state = _stack_components(records, requested_index, hist_len=hist_len)
    next_state = _stack_components(records, requested_index + 1, hist_len=hist_len)
    action_record = records[requested_index + hist_len - 1]
    terminal_record = records[requested_index + hist_len]
    state_arr = state["stack"]
    next_state_arr = next_state["stack"]
    normalized_state = np.ascontiguousarray(state_arr.astype(np.float32) / np.float32(255.0))
    normalized_next_state = np.ascontiguousarray(next_state_arr.astype(np.float32) / np.float32(255.0))

    base.update(
        {
            "actual_replay_index_used": requested_index,
            "state_raw_frame_indices": state["raw_frame_indices"],
            "state_source_frame_indices": state["source_frame_indices"],
            "next_state_raw_frame_indices": next_state["raw_frame_indices"],
            "next_state_source_frame_indices": next_state["source_frame_indices"],
            "state_component_frame_hashes": state["source_frame_hashes"],
            "next_state_component_frame_hashes": next_state["source_frame_hashes"],
            "state_raw_frame_hashes": state["raw_frame_hashes"],
            "next_state_raw_frame_hashes": next_state["raw_frame_hashes"],
            "state_zeroed_positions": state["zeroed_positions"],
            "next_state_zeroed_positions": next_state["zeroed_positions"],
            "state_hash": arr_hash(state_arr),
            "next_state_hash": arr_hash(next_state_arr),
            "state_shape": [int(dim) for dim in state_arr.shape],
            "next_state_shape": [int(dim) for dim in next_state_arr.shape],
            "state_dtype": str(state_arr.dtype),
            "next_state_dtype": str(next_state_arr.dtype),
            "layout": "CHW",
            "normalized_state_hash": arr_hash(normalized_state),
            "normalized_next_state_hash": arr_hash(normalized_next_state),
            "normalized_dtype": str(normalized_state.dtype),
            "normalized_range": [0.0, 1.0],
            "action": int(action_record["action"]),
            "action_one_based": int(action_record["action"]) + 1,
            "ale_action_code": int(action_record["ale_action_code"]),
            "reward": int(action_record["clipped_reward"]),
            "raw_reward": float(action_record["raw_reward"]),
            "terminal_mask": bool(terminal_record["terminal_mask"]),
            "life_loss_terminal": bool(terminal_record["life_loss_terminal"]),
            "true_terminal": bool(terminal_record["true_terminal"]),
            "next_state_zeroed_or_masked": bool(next_state["zeroed_positions"]),
            "action_record_replay_index": int(action_record["replay_index"]),
            "terminal_record_replay_index": int(terminal_record["replay_index"]),
            "episode_context": {
                "state_episode_ids": [records[idx]["episode_id"] for idx in state["raw_frame_indices"]],
                "next_state_episode_ids": [records[idx]["episode_id"] for idx in next_state["raw_frame_indices"]],
                "action_episode_id": action_record["episode_id"],
                "terminal_episode_id": terminal_record["episode_id"],
            },
        }
    )
    return base


def build_batch(records: list[dict[str, Any]], indices: list[int], batch_name: str) -> dict[str, Any]:
    samples = [sample_at(records, index) for index in indices]
    accepted_samples = [sample for sample in samples if sample["accepted"]]
    if len(accepted_samples) != len(indices):
        return {
            "phase": "replay_batch",
            "batch_name": batch_name,
            "accepted": False,
            "requested_indices": indices,
            "rejection_reason": "batch_contains_rejected_index",
        }

    states = []
    next_states = []
    for sample in accepted_samples:
        states.append(_stack_components(records, int(sample["requested_replay_index"]))["stack"])
        next_states.append(_stack_components(records, int(sample["requested_replay_index"]) + 1)["stack"])
    state_batch = np.ascontiguousarray(np.stack(states, axis=0))
    next_state_batch = np.ascontiguousarray(np.stack(next_states, axis=0))
    norm_state_batch = np.ascontiguousarray(state_batch.astype(np.float32) / np.float32(255.0))
    norm_next_state_batch = np.ascontiguousarray(next_state_batch.astype(np.float32) / np.float32(255.0))
    return {
        "phase": "replay_batch",
        "batch_name": batch_name,
        "accepted": True,
        "requested_indices": indices,
        "batch_size": len(indices),
        "layout": "NCHW",
        "state_batch_shape": [int(dim) for dim in state_batch.shape],
        "next_state_batch_shape": [int(dim) for dim in next_state_batch.shape],
        "state_batch_dtype": str(state_batch.dtype),
        "next_state_batch_dtype": str(next_state_batch.dtype),
        "state_batch_hash": arr_hash(state_batch),
        "next_state_batch_hash": arr_hash(next_state_batch),
        "normalized_state_batch_hash": arr_hash(norm_state_batch),
        "normalized_next_state_batch_hash": arr_hash(norm_next_state_batch),
        "normalized_dtype": str(norm_state_batch.dtype),
        "actions": [int(sample["action"]) for sample in accepted_samples],
        "actions_one_based": [int(sample["action_one_based"]) for sample in accepted_samples],
        "rewards": [int(sample["reward"]) for sample in accepted_samples],
        "raw_rewards": [float(sample["raw_reward"]) for sample in accepted_samples],
        "terminal_masks": [bool(sample["terminal_mask"]) for sample in accepted_samples],
        "life_loss_terminals": [bool(sample["life_loss_terminal"]) for sample in accepted_samples],
        "true_terminals": [bool(sample["true_terminal"]) for sample in accepted_samples],
        "state_hashes": [sample["state_hash"] for sample in accepted_samples],
        "next_state_hashes": [sample["next_state_hash"] for sample in accepted_samples],
    }


def batch_plan(records: list[dict[str, Any]], requested_indices: list[int]) -> list[tuple[str, list[int]]]:
    valid = [index for index in requested_indices if index_validity(records, index)[0]]
    terminal_valid = [index for index in valid if sample_at(records, index)["terminal_mask"]]
    life_valid = [index for index in valid if sample_at(records, index)["life_loss_terminal"]]
    wrap_insert = int(records[0].get("wrap_insert_index", len(records) // 2))
    wrap_valid = [index for index in valid if abs(index - wrap_insert) <= 8]

    batches: list[tuple[str, list[int]]] = []
    if valid:
        batches.append(("batch_1", valid[:1]))
    if len(valid) >= 4:
        batches.append(("batch_4", valid[:4]))
    if len(valid) >= 32:
        batches.append(("batch_32", valid[:32]))
    if terminal_valid:
        batches.append(("batch_terminal", terminal_valid[: min(4, len(terminal_valid))]))
    if life_valid:
        batches.append(("batch_life_loss", life_valid[: min(4, len(life_valid))]))
    if wrap_valid:
        batches.append(("batch_near_wrap", wrap_valid[: min(8, len(wrap_valid))]))
    return batches
