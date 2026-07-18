from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from common import arr_hash
from stage4_common import batch_plan, build_batch, load_records, load_requested_indices, _stack_components


FRAME_SHAPE = (84, 84)
HIST_LEN = 4
DEFAULT_BATCH_NAME = "batch_32"


def load_stage5_manifest(fixture_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(fixture_dir) / "manifest.json").read_text(encoding="utf-8"))


def write_raw_array(path: str | Path, array: np.ndarray) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(array).tofile(out_path)


def read_raw_array(path: str | Path, shape: list[int] | tuple[int, ...], dtype: str | np.dtype) -> np.ndarray:
    array = np.fromfile(path, dtype=np.dtype(dtype))
    expected = int(np.prod(shape))
    if array.size != expected:
        raise ValueError(f"{path}: expected {expected} values, got {array.size}")
    return np.ascontiguousarray(array.reshape(tuple(shape)))


def select_batch_indices(records: list[dict[str, Any]], requested_indices: list[int], batch_name: str) -> list[int]:
    for candidate_name, indices in batch_plan(records, requested_indices):
        if candidate_name == batch_name:
            return [int(index) for index in indices]
    raise RuntimeError(f"batch plan did not contain {batch_name}")


def build_learner_arrays(records: list[dict[str, Any]], indices: list[int]) -> dict[str, np.ndarray]:
    states = []
    next_states = []
    actions = []
    rewards = []
    terminals = []
    for index in indices:
        sample = build_batch(records, [index], f"single_{index}")
        if not sample.get("accepted"):
            raise RuntimeError(f"index {index} is not accepted")

        state = _stack_components(records, index)["stack"]
        next_state = _stack_components(records, index + 1)["stack"]
        action_record = records[index + HIST_LEN - 1]
        terminal_record = records[index + HIST_LEN]

        states.append(state)
        next_states.append(next_state)
        actions.append(int(action_record["action"]))
        rewards.append(float(action_record["clipped_reward"]))
        terminals.append(1 if bool(terminal_record["terminal_mask"]) else 0)

    return {
        "states_uint8": np.ascontiguousarray(np.stack(states, axis=0).astype(np.uint8, copy=False)),
        "next_states_uint8": np.ascontiguousarray(np.stack(next_states, axis=0).astype(np.uint8, copy=False)),
        "actions_zero_based": np.asarray(actions, dtype=np.int64),
        "actions_one_based": np.asarray([action + 1 for action in actions], dtype=np.int64),
        "rewards_float32": np.asarray(rewards, dtype=np.float32),
        "terminals_uint8": np.asarray(terminals, dtype=np.uint8),
    }


def tensor_manifest(name: str, array: np.ndarray, rel_path: str) -> dict[str, Any]:
    return {
        "name": name,
        "path": rel_path,
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "hash": arr_hash(array),
    }
