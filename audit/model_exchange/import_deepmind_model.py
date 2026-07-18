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
from common import arr_hash  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Import DeepMind raw model tensors into the PyTorch Atari QNetwork.")
    parser.add_argument("--model-dir", default=os.path.join(out, "model_exchange"))
    parser.add_argument("--atari-dir", default=atari_dir)
    parser.add_argument("--out", default=os.path.join(out, "model_exchange", "pytorch_model.pt"))
    parser.add_argument("--report", default=os.path.join(out, "model_exchange", "import_report.json"))
    parser.add_argument("--network-kind", choices=("current", "faithful"), default="current")
    return parser.parse_args()


def load_qnetwork_class(atari_dir: str | Path, network_kind: str) -> tuple[type[nn.Module], str]:
    if network_kind == "faithful":
        network_path = Path(atari_dir) / "atari" / "deepmind_faithful" / "network.py"
        if not network_path.exists():
            raise FileNotFoundError(f"missing faithful network: {network_path}")
        spec = importlib.util.spec_from_file_location("stage5_faithful_network", network_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not import {network_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.DeepMindAtariQNetwork, str(network_path)

    network_path = Path(atari_dir) / "network.py"
    if not network_path.exists():
        return FallbackQNetwork, "fallback_audit_qnetwork"
    spec = importlib.util.spec_from_file_location("stage5_atari_network", network_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {network_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.QNetwork, str(network_path)


def parameter_mapping(network_kind: str) -> dict[str, tuple[str, str]]:
    if network_kind == "faithful":
        return {
            "conv1": ("conv1.weight", "conv1.bias"),
            "conv2": ("conv2.weight", "conv2.bias"),
            "conv3": ("conv3.weight", "conv3.bias"),
            "fc1": ("fc1.weight", "fc1.bias"),
            "fc2": ("fc2.weight", "fc2.bias"),
        }
    return {
        "conv1": ("conv.0.weight", "conv.0.bias"),
        "conv2": ("conv.2.weight", "conv.2.bias"),
        "conv3": ("conv.4.weight", "conv.4.bias"),
        "fc1": ("fc.0.weight", "fc.0.bias"),
        "fc2": ("fc.2.weight", "fc.2.bias"),
    }


def read_raw_tensor(model_dir: Path, layer: dict[str, Any], field: str) -> np.ndarray:
    shape = [int(dim) for dim in layer[f"{field}_shape"]]
    path = model_dir / layer[f"{field}_file"]
    array = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise ValueError(f"{path}: expected {expected} floats, got {array.size}")
    return np.ascontiguousarray(array.reshape(shape))


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    manifest = json.loads((model_dir / "deepmind_model_manifest.json").read_text(encoding="utf-8"))
    qnetwork_cls, source = load_qnetwork_class(args.atari_dir, args.network_kind)
    model = qnetwork_cls(int(manifest["action_count"]))

    mapping = parameter_mapping(args.network_kind)
    state = model.state_dict()
    report_layers = []
    for layer in manifest["layers"]:
        name = layer["name"]
        if name not in mapping:
            raise KeyError(f"no PyTorch mapping for DeepMind layer {name}")
        weight_key, bias_key = mapping[name]
        weight = read_raw_tensor(model_dir, layer, "weight")
        bias = read_raw_tensor(model_dir, layer, "bias")
        if list(state[weight_key].shape) != list(weight.shape):
            raise ValueError(f"{name} weight shape mismatch: {state[weight_key].shape} vs {weight.shape}")
        if list(state[bias_key].shape) != list(bias.shape):
            raise ValueError(f"{name} bias shape mismatch: {state[bias_key].shape} vs {bias.shape}")
        state[weight_key] = torch.from_numpy(weight.copy())
        state[bias_key] = torch.from_numpy(bias.copy())
        report_layers.append(
            {
                "name": name,
                "pytorch_weight_key": weight_key,
                "pytorch_bias_key": bias_key,
                "weight_shape": list(weight.shape),
                "bias_shape": list(bias.shape),
                "weight_hash": arr_hash(weight),
                "bias_hash": arr_hash(bias),
            }
        )

    model.load_state_dict(state)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "deepmind_manifest": manifest,
            "qnetwork_source": source,
            "network_kind": args.network_kind,
        },
        out_path,
    )
    report = {
        "phase": "stage5_pytorch_model_import",
        "status": "pass",
        "network_kind": args.network_kind,
        "qnetwork_source": source,
        "model_path": str(out_path),
        "layers": report_layers,
    }
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"wrote {args.report}")
    print(f"qnetwork_source: {source}")


if __name__ == "__main__":
    main()
