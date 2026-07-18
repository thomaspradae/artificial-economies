#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import arr_hash, canonical_float, write_jsonl  # noqa: E402
from stage5_common import load_stage5_manifest, read_raw_array  # noqa: E402


class FallbackQNetwork(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, num_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs/stage5_learner")
    atari_dir = os.getenv("ATARI_DIR", str(Path.cwd()))
    parser = argparse.ArgumentParser(description="Trace PyTorch Atari learner math on the frozen Stage 5 minibatch.")
    parser.add_argument("--fixture-dir", default=os.path.join(out, "learner_fixture"))
    parser.add_argument("--model", default=os.path.join(out, "model_exchange", "pytorch_model.pt"))
    parser.add_argument("--atari-dir", default=atari_dir)
    parser.add_argument("--out", default=os.path.join(out, "pytorch_learner.jsonl"))
    parser.add_argument("--lr", type=float, default=float(os.getenv("LEARNER_LR", "0.00025")))
    parser.add_argument("--gamma", type=float, default=float(os.getenv("GAMMA", "0.99")))
    return parser.parse_args()


def import_from_path(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_qnetwork_class(atari_dir: str | Path) -> tuple[type[nn.Module], str]:
    network_path = Path(atari_dir) / "network.py"
    if not network_path.exists():
        return FallbackQNetwork, "fallback_audit_qnetwork"
    module = import_from_path("stage5_atari_network_trace", network_path)
    return module.QNetwork, str(network_path)


def load_loss(atari_dir: str | Path) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], str]:
    train_path = Path(atari_dir) / "train_nature.py"
    if train_path.exists():
        try:
            module = import_from_path("stage5_train_nature", train_path)
            return module.clipped_td_error_loss, str(train_path)
        except Exception:
            pass

    def fallback_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        td_errors = predictions - targets
        abs_errors = td_errors.abs()
        quadratic = torch.minimum(abs_errors, torch.ones_like(abs_errors))
        linear = abs_errors - quadratic
        return (0.5 * quadratic.pow(2) + linear).mean()

    return fallback_loss, "fallback_mean_huber"


def load_optimizer_class(atari_dir: str | Path) -> tuple[type[torch.optim.Optimizer], str]:
    opt_path = Path(atari_dir) / "deepmind_rmsprop.py"
    if not opt_path.exists():
        raise RuntimeError(f"missing DeepMindRMSprop implementation: {opt_path}")
    module = import_from_path("stage5_deepmind_rmsprop", opt_path)
    return module.DeepMindRMSprop, str(opt_path)


def to_float_list(array: np.ndarray | torch.Tensor, limit: int | None = None) -> list[Any]:
    if isinstance(array, torch.Tensor):
        data = array.detach().cpu().numpy()
    else:
        data = np.asarray(array)
    flat = data.reshape(-1)
    if limit is not None:
        flat = flat[:limit]
    return [canonical_float(float(value), digits=10) for value in flat.tolist()]


def matrix_rows(tensor: torch.Tensor, rows: int = 8) -> list[list[Any]]:
    data = tensor.detach().cpu().numpy()
    return [[canonical_float(float(value), digits=10) for value in row] for row in data[:rows].tolist()]


def summarize_array(array: np.ndarray | torch.Tensor) -> dict[str, Any]:
    if isinstance(array, torch.Tensor):
        data = array.detach().cpu().numpy()
    else:
        data = np.asarray(array)
    data = np.ascontiguousarray(data.astype(np.float32, copy=False))
    return {
        "shape": [int(dim) for dim in data.shape],
        "dtype": str(data.dtype),
        "hash": arr_hash(data),
        "mean": canonical_float(float(data.mean()), digits=10),
        "max_abs": canonical_float(float(np.abs(data).max()), digits=10),
        "l2": canonical_float(float(np.sqrt(np.square(data.astype(np.float64)).sum())), digits=10),
        "first_values": to_float_list(data, limit=8),
    }


def layer_summaries(model: nn.Module, include_grads: bool = False) -> list[dict[str, Any]]:
    rows = []
    for name, parameter in model.named_parameters():
        item = {"name": name, "param": summarize_array(parameter)}
        if include_grads:
            item["grad"] = None if parameter.grad is None else summarize_array(parameter.grad)
        rows.append(item)
    return rows


def load_fixture(fixture_dir: Path) -> dict[str, np.ndarray]:
    manifest = load_stage5_manifest(fixture_dir)
    tensors = {entry["name"]: entry for entry in manifest["tensors"]}
    return {
        name: read_raw_array(fixture_dir / entry["path"], entry["shape"], entry["dtype"])
        for name, entry in tensors.items()
    }


def exchange_layer_contract(checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for layer in checkpoint["deepmind_manifest"]["layers"]:
        rows.append(
            {
                "name": layer["name"],
                "weight_shape": [int(dim) for dim in layer["weight_shape"]],
                "bias_shape": [int(dim) for dim in layer["bias_shape"]],
                "weight_sha256": layer["weight_sha256"],
                "bias_sha256": layer["bias_sha256"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    fixture_dir = Path(args.fixture_dir)
    manifest = load_stage5_manifest(fixture_dir)
    arrays = load_fixture(fixture_dir)
    checkpoint = torch.load(args.model, map_location="cpu")

    qnetwork_cls, qnetwork_source = load_qnetwork_class(args.atari_dir)
    loss_fn, loss_source = load_loss(args.atari_dir)
    optimizer_cls, optimizer_source = load_optimizer_class(args.atari_dir)

    action_count = int(checkpoint["deepmind_manifest"]["action_count"])
    q_net = qnetwork_cls(action_count)
    target_net = qnetwork_cls(action_count)
    q_net.load_state_dict(checkpoint["state_dict"])
    target_net.load_state_dict(checkpoint["state_dict"])
    q_net.train()
    target_net.eval()

    states = torch.as_tensor(arrays["states_uint8"], dtype=torch.float32)
    next_states = torch.as_tensor(arrays["next_states_uint8"], dtype=torch.float32)
    actions = torch.as_tensor(arrays["actions_zero_based"], dtype=torch.long)
    rewards = torch.as_tensor(arrays["rewards_float32"], dtype=torch.float32)
    terminals = torch.as_tensor(arrays["terminals_uint8"], dtype=torch.float32)

    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "phase": "model_contract",
            "source": "pytorch",
            "step": 0,
            "architecture": "convnet_atari3_compatible",
            "qnetwork_source_kind": "real_atari_qnetwork" if "network.py" in qnetwork_source else "fallback",
            "loss_source_kind": "real_train_nature_loss" if "train_nature.py" in loss_source else "fallback",
            "optimizer_source_kind": "real_atari_deepmind_rmsprop" if "deepmind_rmsprop.py" in optimizer_source else "fallback",
            "action_count": action_count,
            "input_shape": [4, 84, 84],
            "parameter_layers": exchange_layer_contract(checkpoint),
        }
    )

    q_values = q_net(states)
    with torch.no_grad():
        next_q_values = target_net(next_states)
        max_next_q = next_q_values.max(dim=1).values
        target_values = rewards + (1.0 - terminals) * float(args.gamma) * max_next_q
    q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    delta_unclipped = target_values - q_selected.detach()
    clipped_delta = torch.clamp(delta_unclipped, -1.0, 1.0)

    rows.append(
        {
            "phase": "forward",
            "source": "pytorch",
            "step": 1,
            "input_contract": "effective_float32_0_1_network_input",
            "batch_size": int(states.shape[0]),
            "q_values_first_rows": matrix_rows(q_values),
            "next_q_values_first_rows": matrix_rows(next_q_values),
            "q_values_summary": summarize_array(q_values),
            "next_q_values_summary": summarize_array(next_q_values),
        }
    )
    rows.append(
        {
            "phase": "bellman_target",
            "source": "pytorch",
            "step": 2,
            "gamma": canonical_float(float(args.gamma), digits=10),
            "actions_zero_based": [int(x) for x in arrays["actions_zero_based"].tolist()],
            "actions_one_based": [int(x) for x in arrays["actions_one_based"].tolist()],
            "rewards": to_float_list(rewards),
            "terminals": [int(x) for x in arrays["terminals_uint8"].tolist()],
            "q_selected": to_float_list(q_selected),
            "max_next_q": to_float_list(max_next_q),
            "target_values": to_float_list(target_values),
            "delta_unclipped": to_float_list(delta_unclipped),
            "clipped_delta": to_float_list(clipped_delta),
        }
    )

    predictions = q_selected
    loss = loss_fn(predictions, target_values.detach())
    optimizer = optimizer_cls(q_net.parameters(), lr=float(args.lr), alpha=0.95, eps=0.01)
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.sqrt(
        sum(
            parameter.grad.detach().pow(2).sum()
            for parameter in q_net.parameters()
            if parameter.grad is not None
        )
    )

    rows.append(
        {
            "phase": "loss_gradient_contract",
            "source": "pytorch",
            "step": 3,
            "loss_mode": "pytorch_current_mean_huber_scalar_loss",
            "scalar_loss": canonical_float(float(loss.detach().item()), digits=10),
            "output_gradient_scale": "d_loss_d_q_selected_is_clamp(q-target,-1,1)/batch_size",
            "grad_norm": canonical_float(float(grad_norm.item()), digits=10),
            "pred_minus_target_first_values": to_float_list((predictions.detach() - target_values.detach()), limit=8),
        }
    )
    rows.append(
        {
            "phase": "gradient_summary",
            "source": "pytorch",
            "step": 4,
            "parameter_layers": layer_summaries(q_net, include_grads=True),
        }
    )

    before = {name: parameter.detach().clone() for name, parameter in q_net.named_parameters()}
    optimizer.step()
    deltas = []
    for name, parameter in q_net.named_parameters():
        delta = parameter.detach() - before[name]
        deltas.append({"name": name, "delta": summarize_array(delta)})

    rows.append(
        {
            "phase": "optimizer_update",
            "source": "pytorch",
            "step": 5,
            "optimizer_mode": "real_atari_deepmind_rmsprop_on_current_huber_gradients",
            "lr": canonical_float(float(args.lr), digits=10),
            "alpha": 0.95,
            "epsilon": 0.01,
            "epsilon_placement": "inside_sqrt",
            "parameter_deltas": deltas,
            "post_update_layers": layer_summaries(q_net, include_grads=False),
        }
    )

    write_jsonl(args.out, rows)
    print(f"wrote {args.out}")
    print(f"rows: {len(rows)}")
    print(f"fixture_content_hash: {manifest['content_hash']}")


if __name__ == "__main__":
    main()
