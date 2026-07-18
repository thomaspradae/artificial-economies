#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class DeepMindPaddingQNetwork(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs/stage5_learner")
    atari_dir = os.getenv("ATARI_DIR", str(Path.cwd()))
    parser = argparse.ArgumentParser(description="Diagnose Stage 5 forward mismatch against DeepMind conv1 padding.")
    parser.add_argument("--fixture-dir", default=os.path.join(out, "learner_fixture"))
    parser.add_argument("--model-dir", default=os.path.join(out, "model_exchange"))
    parser.add_argument("--deepmind-trace", default=os.path.join(out, "deepmind_learner.jsonl"))
    parser.add_argument("--atari-dir", default=atari_dir)
    parser.add_argument("--report", default=os.path.join(out, "forward_architecture_diagnosis.txt"))
    return parser.parse_args()


def import_qnetwork(atari_dir: str | Path) -> type[nn.Module]:
    path = Path(atari_dir) / "network.py"
    spec = importlib.util.spec_from_file_location("stage5_current_network_diag", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.QNetwork


def load_raw_model(model: nn.Module, model_dir: Path, manifest: dict[str, Any]) -> None:
    mapping = {
        "conv1": ("conv.0.weight", "conv.0.bias"),
        "conv2": ("conv.2.weight", "conv.2.bias"),
        "conv3": ("conv.4.weight", "conv.4.bias"),
        "fc1": ("fc.0.weight", "fc.0.bias"),
        "fc2": ("fc.2.weight", "fc.2.bias"),
    }
    state = model.state_dict()
    for layer in manifest["layers"]:
        weight_key, bias_key = mapping[layer["name"]]
        weight = np.fromfile(model_dir / layer["weight_file"], dtype=np.float32).reshape(layer["weight_shape"])
        bias = np.fromfile(model_dir / layer["bias_file"], dtype=np.float32).reshape(layer["bias_shape"])
        state[weight_key] = torch.from_numpy(weight.copy())
        state[bias_key] = torch.from_numpy(bias.copy())
    model.load_state_dict(state)


def load_fixture_states(fixture_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))
    tensors = {entry["name"]: entry for entry in manifest["tensors"]}
    states_entry = tensors["states_uint8"]
    next_entry = tensors["next_states_uint8"]
    states = np.fromfile(fixture_dir / states_entry["path"], dtype=np.uint8).reshape(states_entry["shape"])
    next_states = np.fromfile(fixture_dir / next_entry["path"], dtype=np.uint8).reshape(next_entry["shape"])
    return states, next_states


def conv_summary(model: nn.Module) -> list[str]:
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            rows.append(
                f"{name}: in={module.in_channels} out={module.out_channels} "
                f"kernel={tuple(module.kernel_size)} stride={tuple(module.stride)} padding={tuple(module.padding)}"
            )
    return rows


def deepmind_forward_rows(trace_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    forward = next(row for row in rows if row.get("phase") == "forward")
    return (
        np.asarray(forward["q_values_first_rows"], dtype=np.float32),
        np.asarray(forward["next_q_values_first_rows"], dtype=np.float32),
    )


def max_abs_first8(model: nn.Module, states: np.ndarray, next_states: np.ndarray, dm_q: np.ndarray, dm_next_q: np.ndarray) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        q = model(torch.as_tensor(states, dtype=torch.float32)).detach().cpu().numpy()
        next_q = model(torch.as_tensor(next_states, dtype=torch.float32)).detach().cpu().numpy()
    return (
        float(np.max(np.abs(q[: dm_q.shape[0]] - dm_q))),
        float(np.max(np.abs(next_q[: dm_next_q.shape[0]] - dm_next_q))),
    )


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    manifest = json.loads((model_dir / "deepmind_model_manifest.json").read_text(encoding="utf-8"))
    current_cls = import_qnetwork(args.atari_dir)
    current = current_cls(int(manifest["action_count"]))
    control = DeepMindPaddingQNetwork(int(manifest["action_count"]))
    load_raw_model(current, model_dir, manifest)
    load_raw_model(control, model_dir, manifest)
    states, next_states = load_fixture_states(Path(args.fixture_dir))
    dm_q, dm_next_q = deepmind_forward_rows(Path(args.deepmind_trace))
    current_q_diff, current_next_diff = max_abs_first8(current, states, next_states, dm_q, dm_next_q)
    control_q_diff, control_next_diff = max_abs_first8(control, states, next_states, dm_q, dm_next_q)

    lines = [
        "Stage 5 forward architecture diagnosis",
        "",
        "DeepMind convnet_atari3 convolution contract:",
        "conv1: kernel=(8,8) stride=(4,4) padding=(1,1)",
        "conv2: kernel=(4,4) stride=(2,2) padding=(0,0)",
        "conv3: kernel=(3,3) stride=(1,1) padding=(0,0)",
        "",
        "Current PyTorch QNetwork:",
        *conv_summary(current),
        "",
        "PyTorch DeepMind-padding control:",
        *conv_summary(control),
        "",
        f"current_q_first8_max_abs_diff: {current_q_diff:.12g}",
        f"current_next_q_first8_max_abs_diff: {current_next_diff:.12g}",
        f"padding_control_q_first8_max_abs_diff: {control_q_diff:.12g}",
        f"padding_control_next_q_first8_max_abs_diff: {control_next_diff:.12g}",
        "",
        "Conclusion:",
    ]
    if control_q_diff < 1e-6 and control_next_diff < 1e-6 and current_next_diff > 1e-5:
        lines.append("forward mismatch is explained by missing PyTorch conv1 padding=1")
    else:
        lines.append("forward mismatch is not fully explained by conv1 padding")
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
