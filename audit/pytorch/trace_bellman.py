#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import arr_hash, canonical_float, write_jsonl  # noqa: E402
from model_exchange.import_deepmind_model import parameter_mapping, read_raw_tensor  # noqa: E402
from stage4_common import HIST_LEN, _stack_components, index_validity, load_records, load_requested_indices, sample_at  # noqa: E402
from stage5_common import DEFAULT_BATCH_NAME, select_batch_indices, write_raw_array  # noqa: E402


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs/stage5_bellman")
    base = os.getenv("BASE_OUTPUT_DIR", "audit_outputs")
    atari_dir = os.getenv("ATARI_DIR", str(Path.cwd()))
    parser = argparse.ArgumentParser(description="Trace PyTorch Bellman-target computation on frozen replay batches.")
    parser.add_argument("--stage4-dir", default=os.getenv("STAGE4_REPLAY_SAMPLE_DIR", os.path.join(base, "stage4_replay_sample")))
    parser.add_argument("--model-dir", default=os.getenv("STAGE5_MODEL_DIR", os.path.join(base, "stage5_learner", "model_exchange")))
    parser.add_argument("--atari-dir", default=atari_dir)
    parser.add_argument("--out-dir", default=out)
    parser.add_argument("--out", default=os.path.join(out, "pytorch_bellman.jsonl"))
    parser.add_argument("--spec-out", default=os.path.join(out, "bellman_batches.tsv"))
    parser.add_argument("--gamma", type=float, default=float(os.getenv("GAMMA", "0.99")))
    return parser.parse_args()


def import_faithful_network(atari_dir: str | Path) -> type[nn.Module]:
    path = Path(atari_dir) / "atari" / "deepmind_faithful" / "network.py"
    spec = importlib.util.spec_from_file_location("stage5b_faithful_network", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DeepMindAtariQNetwork


def load_raw_model(model: nn.Module, model_dir: Path, manifest: dict[str, Any]) -> None:
    mapping = parameter_mapping("faithful")
    state = model.state_dict()
    for layer in manifest["layers"]:
        weight_key, bias_key = mapping[layer["name"]]
        state[weight_key] = torch.from_numpy(read_raw_tensor(model_dir, layer, "weight").copy())
        state[bias_key] = torch.from_numpy(read_raw_tensor(model_dir, layer, "bias").copy())
    model.load_state_dict(state)


def all_valid_indices(records: list[dict[str, Any]]) -> list[int]:
    max_start = len(records) - HIST_LEN - 1
    return [index for index in range(1, max_start + 1) if index_validity(records, index)[0]]


def first_n(values: list[int], n: int, label: str) -> list[int]:
    if len(values) < n:
        raise RuntimeError(f"not enough indices for {label}: need {n}, got {len(values)}")
    return values[:n]


def first_available(values: list[int], label: str, cap: int = 8) -> list[int]:
    if not values:
        raise RuntimeError(f"no accepted Stage 4 replay index found for required control: {label}")
    return values[: min(cap, len(values))]


def make_batch_plan(records: list[dict[str, Any]], requested: list[int]) -> list[tuple[str, list[int]]]:
    valid = all_valid_indices(records)
    batch32 = select_batch_indices(records, requested, DEFAULT_BATCH_NAME)
    terminal_true = [index for index in valid if sample_at(records, index)["true_terminal"]]
    terminal_life = [index for index in valid if sample_at(records, index)["life_loss_terminal"]]
    nonterminal = [index for index in valid if not sample_at(records, index)["terminal_mask"]]
    zero_reward = [
        index
        for index in valid
        if float(records[index + HIST_LEN - 1]["clipped_reward"]) == 0.0
    ]
    positive_reward = [
        index
        for index in valid
        if float(records[index + HIST_LEN - 1]["clipped_reward"]) > 0.0
    ]
    return [
        ("batch_1", first_n(valid, 1, "batch_1")),
        ("batch_4", first_n(valid, 4, "batch_4")),
        ("batch_32", batch32),
        ("ordinary_nonterminal", first_available(nonterminal, "ordinary_nonterminal")),
        ("true_terminal", first_available(terminal_true, "true_terminal", cap=4)),
        ("life_loss_terminal", first_available(terminal_life, "life_loss_terminal", cap=4)),
        ("zero_reward", first_available(zero_reward, "zero_reward")),
        ("positive_reward", first_available(positive_reward, "positive_reward")),
    ]


def write_vector(path: Path, values: list[int] | list[float], fmt: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(fmt.format(value) for value in values) + "\n", encoding="utf-8")


def materialize_batch(records: list[dict[str, Any]], batch_name: str, indices: list[int], batch_dir: Path) -> dict[str, Any]:
    states = []
    next_states = []
    actions_zero = []
    actions_one = []
    rewards = []
    terminals = []
    true_terminals = []
    life_terminals = []
    for index in indices:
        sample = sample_at(records, index)
        states.append(_stack_components(records, index)["stack"])
        next_states.append(_stack_components(records, index + 1)["stack"])
        action_record = records[index + HIST_LEN - 1]
        actions_zero.append(int(action_record["action"]))
        actions_one.append(int(action_record["action"]) + 1)
        rewards.append(float(action_record["clipped_reward"]))
        terminals.append(1 if sample["terminal_mask"] else 0)
        true_terminals.append(1 if sample["true_terminal"] else 0)
        life_terminals.append(1 if sample["life_loss_terminal"] else 0)

    batch_dir.mkdir(parents=True, exist_ok=True)
    state_arr = np.ascontiguousarray(np.stack(states, axis=0).astype(np.uint8, copy=False))
    next_arr = np.ascontiguousarray(np.stack(next_states, axis=0).astype(np.uint8, copy=False))
    write_raw_array(batch_dir / "states_uint8.bin", state_arr)
    write_raw_array(batch_dir / "next_states_uint8.bin", next_arr)
    write_vector(batch_dir / "replay_indices.txt", indices)
    write_vector(batch_dir / "actions_zero_based.txt", actions_zero)
    write_vector(batch_dir / "actions_one_based.txt", actions_one)
    write_vector(batch_dir / "rewards.txt", rewards, "{:.17g}")
    write_vector(batch_dir / "terminals.txt", terminals)
    write_vector(batch_dir / "true_terminals.txt", true_terminals)
    write_vector(batch_dir / "life_loss_terminals.txt", life_terminals)
    return {
        "batch_name": batch_name,
        "batch_size": len(indices),
        "replay_indices": ",".join(str(index) for index in indices),
        "states_file": str(batch_dir / "states_uint8.bin"),
        "next_states_file": str(batch_dir / "next_states_uint8.bin"),
        "replay_indices_file": str(batch_dir / "replay_indices.txt"),
        "actions_zero_file": str(batch_dir / "actions_zero_based.txt"),
        "actions_one_file": str(batch_dir / "actions_one_based.txt"),
        "rewards_file": str(batch_dir / "rewards.txt"),
        "terminals_file": str(batch_dir / "terminals.txt"),
        "true_terminals_file": str(batch_dir / "true_terminals.txt"),
        "life_loss_terminals_file": str(batch_dir / "life_loss_terminals.txt"),
        "states_hash": arr_hash(state_arr),
        "next_states_hash": arr_hash(next_arr),
    }


def write_spec(spec_path: Path, entries: list[dict[str, Any]]) -> None:
    header = [
        "batch_name",
        "batch_size",
        "replay_indices",
        "states_file",
        "next_states_file",
        "replay_indices_file",
        "actions_zero_file",
        "actions_one_file",
        "rewards_file",
        "terminals_file",
        "true_terminals_file",
        "life_loss_terminals_file",
        "states_hash",
        "next_states_hash",
    ]
    lines = ["\t".join(header)]
    for entry in entries:
        lines.append("\t".join(str(entry[key]) for key in header))
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_ints(path: str | Path) -> list[int]:
    return [int(line.strip()) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_floats(path: str | Path) -> list[float]:
    return [float(line.strip()) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def tensor_rows(values: torch.Tensor) -> list[list[Any]]:
    array = values.detach().cpu().numpy()
    return [[canonical_float(float(value), digits=10) for value in row] for row in array.tolist()]


def float_list(values: torch.Tensor | np.ndarray | list[float]) -> list[Any]:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy().reshape(-1)
    else:
        array = np.asarray(values).reshape(-1)
    return [canonical_float(float(value), digits=10) for value in array.tolist()]


def compute_rows_for_batch(
    model: nn.Module,
    target_model: nn.Module,
    entry: dict[str, Any],
    gamma: float,
    start_step: int,
) -> list[dict[str, Any]]:
    batch_size = int(entry["batch_size"])
    states = np.fromfile(entry["states_file"], dtype=np.uint8).reshape(batch_size, 4, 84, 84)
    next_states = np.fromfile(entry["next_states_file"], dtype=np.uint8).reshape(batch_size, 4, 84, 84)
    replay_indices = load_ints(entry["replay_indices_file"])
    actions_zero = load_ints(entry["actions_zero_file"])
    actions_one = load_ints(entry["actions_one_file"])
    rewards_np = np.asarray(load_floats(entry["rewards_file"]), dtype=np.float32)
    terminals_np = np.asarray(load_ints(entry["terminals_file"]), dtype=np.uint8)
    true_terminals = load_ints(entry["true_terminals_file"])
    life_terminals = load_ints(entry["life_loss_terminals_file"])

    with torch.no_grad():
        state_t = torch.as_tensor(states, dtype=torch.float32)
        next_t = torch.as_tensor(next_states, dtype=torch.float32)
        q_values = model(state_t)
        next_q_values = target_model(next_t)
        max_next_q, next_argmax = next_q_values.max(dim=1)
        actions_t = torch.as_tensor(actions_zero, dtype=torch.long)
        selected_q = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32)
        terminals = torch.as_tensor(terminals_np.astype(np.float32), dtype=torch.float32)
        continuation_mask = 1.0 - terminals
        discounted = continuation_mask * float(gamma) * max_next_q
        targets = rewards + discounted
        td_errors = targets - selected_q

    rows: list[dict[str, Any]] = []
    for i in range(batch_size):
        terminal_flag = bool(terminals_np[i])
        target_equals_reward = abs(float(targets[i]) - float(rewards_np[i])) <= 1e-7
        rows.append(
            {
                "phase": "bellman_sample",
                "source": "pytorch",
                "step": start_step + i,
                "batch_name": entry["batch_name"],
                "batch_position": i,
                "replay_index": int(replay_indices[i]),
                "action": int(actions_zero[i]),
                "action_one_based": int(actions_one[i]),
                "reward": canonical_float(float(rewards_np[i]), digits=10),
                "terminal_flag": terminal_flag,
                "true_terminal": bool(true_terminals[i]),
                "life_loss_terminal": bool(life_terminals[i]),
                "continuation_mask": canonical_float(float(continuation_mask[i]), digits=10),
                "gamma": canonical_float(float(gamma), digits=10),
                "online_q_values": float_list(q_values[i]),
                "selected_q": canonical_float(float(selected_q[i]), digits=10),
                "target_next_q_values": float_list(next_q_values[i]),
                "maximizing_next_action": int(next_argmax[i]),
                "max_next_q": canonical_float(float(max_next_q[i]), digits=10),
                "discounted_continuation": canonical_float(float(discounted[i]), digits=10),
                "bellman_target": canonical_float(float(targets[i]), digits=10),
                "td_error": canonical_float(float(td_errors[i]), digits=10),
                "target_network_used": True,
                "max_dimension": "action",
                "terminal_target_equals_reward": bool(terminal_flag and target_equals_reward),
                "nonterminal_gamma_applied": (not terminal_flag) and abs(float(discounted[i]) - float(gamma) * float(max_next_q[i])) <= 1e-7,
            }
        )
    rows.append(
        {
            "phase": "bellman_batch",
            "source": "pytorch",
            "step": f"batch:{entry['batch_name']}",
            "batch_name": entry["batch_name"],
            "batch_size": batch_size,
            "replay_indices": replay_indices,
            "actions": actions_zero,
            "rewards": float_list(rewards_np),
            "terminal_flags": [bool(value) for value in terminals_np.tolist()],
            "maximizing_next_actions": [int(value) for value in next_argmax.detach().cpu().numpy().tolist()],
            "targets": float_list(targets),
            "td_errors": float_list(td_errors),
            "target_shape": [int(dim) for dim in targets.shape],
        }
    )
    return rows


def main() -> None:
    args = parse_args()
    stage4_dir = Path(args.stage4_dir)
    out_dir = Path(args.out_dir)
    batch_root = out_dir / "batches"
    model_dir = Path(args.model_dir)
    records = load_records(stage4_dir / "canonical_replay")
    requested = load_requested_indices(stage4_dir / "requested_indices.txt")
    plan = make_batch_plan(records, requested)
    spec_entries = [
        materialize_batch(records, batch_name, indices, batch_root / batch_name)
        for batch_name, indices in plan
    ]
    write_spec(Path(args.spec_out), spec_entries)

    manifest = json.loads((model_dir / "deepmind_model_manifest.json").read_text(encoding="utf-8"))
    network_cls = import_faithful_network(args.atari_dir)
    model = network_cls(int(manifest["action_count"]))
    target_model = network_cls(int(manifest["action_count"]))
    load_raw_model(model, model_dir, manifest)
    load_raw_model(target_model, model_dir, manifest)
    model.eval()
    target_model.eval()

    rows: list[dict[str, Any]] = []
    step = 0
    for entry in spec_entries:
        batch_rows = compute_rows_for_batch(model, target_model, entry, args.gamma, step)
        rows.extend(batch_rows)
        step += len(batch_rows)

    write_jsonl(args.out, rows)
    print(f"wrote {args.out}")
    print(f"wrote {args.spec_out}")
    print(f"batches: {len(spec_entries)}")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
