#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from import_deepmind_model import parameter_mapping, read_raw_tensor

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import arr_hash  # noqa: E402


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs/stage5_learner")
    parser = argparse.ArgumentParser(description="Verify DeepMind-to-PyTorch parameter mapping.")
    parser.add_argument("--model-dir", default=os.path.join(out, "model_exchange"))
    parser.add_argument("--pytorch-model", default=os.path.join(out, "model_exchange", "pytorch_model.pt"))
    parser.add_argument("--report", default=os.path.join(out, "model_exchange", "mapping_verification.json"))
    parser.add_argument("--network-kind", choices=("current", "faithful"), default="current")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    manifest = json.loads((model_dir / "deepmind_model_manifest.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(args.pytorch_model, map_location="cpu")
    state = checkpoint["state_dict"]
    mapping = parameter_mapping(args.network_kind)
    errors: list[str] = []
    rows = []
    for layer in manifest["layers"]:
        name = layer["name"]
        weight_key, bias_key = mapping[name]
        dm_weight = read_raw_tensor(model_dir, layer, "weight")
        dm_bias = read_raw_tensor(model_dir, layer, "bias")
        pt_weight = state[weight_key].detach().cpu().numpy().astype(np.float32, copy=False)
        pt_bias = state[bias_key].detach().cpu().numpy().astype(np.float32, copy=False)
        weight_match = arr_hash(dm_weight) == arr_hash(pt_weight)
        bias_match = arr_hash(dm_bias) == arr_hash(pt_bias)
        if not weight_match:
            errors.append(f"{name} weight hash mismatch")
        if not bias_match:
            errors.append(f"{name} bias hash mismatch")
        rows.append(
            {
                "name": name,
                "weight_key": weight_key,
                "bias_key": bias_key,
                "weight_match": weight_match,
                "bias_match": bias_match,
                "weight_hash": arr_hash(pt_weight),
                "bias_hash": arr_hash(pt_bias),
            }
        )

    report = {
        "phase": "stage5_model_mapping_verification",
        "status": "pass" if not errors else "fail",
        "network_kind": args.network_kind,
        "errors": errors,
        "layers": rows,
    }
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
