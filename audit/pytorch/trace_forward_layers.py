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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import arr_hash, canonical_float, write_jsonl  # noqa: E402
from model_exchange.import_deepmind_model import parameter_mapping, read_raw_tensor  # noqa: E402
from stage5_common import load_stage5_manifest, read_raw_array, write_raw_array  # noqa: E402


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs/stage5_learner")
    atari_dir = os.getenv("ATARI_DIR", str(Path.cwd()))
    parser = argparse.ArgumentParser(description="Trace layer-by-layer PyTorch faithful Atari forward pass.")
    parser.add_argument("--fixture-dir", default=os.path.join(out, "learner_fixture"))
    parser.add_argument("--model-dir", default=os.path.join(out, "model_exchange"))
    parser.add_argument("--atari-dir", default=atari_dir)
    parser.add_argument("--out", default=os.path.join(out, "pytorch_forward_layers.jsonl"))
    parser.add_argument("--tensor-dir", default=os.path.join(out, "forward_layers", "pytorch"))
    parser.add_argument("--architecture-manifest", default=os.path.join(out, "architecture_manifest.json"))
    return parser.parse_args()


def load_faithful_class(atari_dir: str | Path) -> type[nn.Module]:
    path = Path(atari_dir) / "atari" / "deepmind_faithful" / "network.py"
    spec = importlib.util.spec_from_file_location("stage5_faithful_network_trace", path)
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


def tensor_stats(array: np.ndarray, tensor_path: str) -> dict[str, Any]:
    data64 = array.astype(np.float64, copy=False)
    flat = array.reshape(-1)
    return {
        "tensor_file": tensor_path,
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "hash": arr_hash(array),
        "min": canonical_float(float(data64.min()), digits=10),
        "max": canonical_float(float(data64.max()), digits=10),
        "mean": canonical_float(float(data64.mean()), digits=10),
        "first_values": [canonical_float(float(value), digits=10) for value in flat[:16].tolist()],
    }


def load_fixture_arrays(fixture_dir: Path) -> dict[str, np.ndarray]:
    manifest = load_stage5_manifest(fixture_dir)
    entries = {entry["name"]: entry for entry in manifest["tensors"]}
    return {
        name: read_raw_array(fixture_dir / entry["path"], entry["shape"], entry["dtype"])
        for name, entry in entries.items()
    }


def trace_prefix(
    rows: list[dict[str, Any]],
    source_name: str,
    layers: dict[str, torch.Tensor],
    tensor_dir: Path,
    step_start: int,
) -> None:
    for offset, (layer_name, tensor) in enumerate(layers.items()):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy().astype(np.float32, copy=False))
        rel_path = f"{source_name}_{layer_name}.float32"
        write_raw_array(tensor_dir / rel_path, array)
        rows.append(
            {
                "phase": "forward_layer",
                "source": "pytorch",
                "step": step_start + offset,
                "batch_source": source_name,
                "layer": layer_name,
                **tensor_stats(array, rel_path),
            }
        )


def main() -> None:
    args = parse_args()
    fixture_dir = Path(args.fixture_dir)
    model_dir = Path(args.model_dir)
    tensor_dir = Path(args.tensor_dir)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    dm_manifest = json.loads((model_dir / "deepmind_model_manifest.json").read_text(encoding="utf-8"))
    network_cls = load_faithful_class(args.atari_dir)
    model = network_cls(int(dm_manifest["action_count"]))
    load_raw_model(model, model_dir, dm_manifest)
    model.eval()
    Path(args.architecture_manifest).write_text(
        json.dumps(model.architecture_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    arrays = load_fixture_arrays(fixture_dir)
    states = torch.as_tensor(arrays["states_uint8"], dtype=torch.float32)
    next_states = torch.as_tensor(arrays["next_states_uint8"], dtype=torch.float32)

    rows: list[dict[str, Any]] = [
        {
            "phase": "forward_architecture",
            "source": "pytorch",
            "step": 0,
            "network_class": "DeepMindAtariQNetwork",
            "architecture_manifest": str(Path(args.architecture_manifest)),
            "conv1_padding": [int(v) for v in model.conv1.padding],
            "weight_mapping": "deepmind_raw_to_faithful_keys",
        }
    ]
    with torch.no_grad():
        trace_prefix(rows, "state", model.forward_layers(states), tensor_dir, 1)
        trace_prefix(rows, "next_state", model.forward_layers(next_states), tensor_dir, 100)
    write_jsonl(args.out, rows)
    print(f"wrote {args.out}")
    print(f"wrote {args.architecture_manifest}")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
