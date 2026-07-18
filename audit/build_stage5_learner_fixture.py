#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from common import arr_hash
from stage4_common import load_records, load_requested_indices
from stage5_common import (
    DEFAULT_BATCH_NAME,
    build_learner_arrays,
    select_batch_indices,
    tensor_manifest,
    write_raw_array,
)


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs/stage5_learner")
    stage4 = os.getenv("STAGE4_REPLAY_SAMPLE_DIR", "audit_outputs/stage4_replay_sample")
    parser = argparse.ArgumentParser(description="Build the frozen Stage 5 learner minibatch from Stage 4 replay outputs.")
    parser.add_argument("--stage4-dir", default=stage4)
    parser.add_argument("--out-dir", default=os.path.join(out, "learner_fixture"))
    parser.add_argument("--batch-name", default=DEFAULT_BATCH_NAME)
    parser.add_argument("--gamma", type=float, default=float(os.getenv("GAMMA", "0.99")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage4_dir = Path(args.stage4_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    replay_dir = stage4_dir / "canonical_replay"
    records = load_records(replay_dir)
    requested = load_requested_indices(stage4_dir / "requested_indices.txt")
    indices = select_batch_indices(records, requested, args.batch_name)
    arrays = build_learner_arrays(records, indices)

    tensor_entries = []
    for name, array in arrays.items():
        rel = f"{name}.bin"
        write_raw_array(out_dir / rel, array)
        np.save(out_dir / f"{name}.npy", array)
        tensor_entries.append(tensor_manifest(name, array, rel))

    (out_dir / "actions_zero_based.txt").write_text(
        "\n".join(str(int(value)) for value in arrays["actions_zero_based"].tolist()) + "\n",
        encoding="utf-8",
    )
    (out_dir / "actions_one_based.txt").write_text(
        "\n".join(str(int(value)) for value in arrays["actions_one_based"].tolist()) + "\n",
        encoding="utf-8",
    )
    (out_dir / "rewards.txt").write_text(
        "\n".join(f"{float(value):.17g}" for value in arrays["rewards_float32"].tolist()) + "\n",
        encoding="utf-8",
    )
    (out_dir / "terminals.txt").write_text(
        "\n".join(str(int(value)) for value in arrays["terminals_uint8"].tolist()) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "phase": "stage5_learner_fixture",
        "source": "stage4_canonical_replay_sample",
        "stage4_dir": str(stage4_dir),
        "replay_dir": str(replay_dir),
        "out_dir": str(out_dir),
        "batch_name": args.batch_name,
        "batch_size": int(arrays["states_uint8"].shape[0]),
        "hist_len": int(arrays["states_uint8"].shape[1]),
        "frame_shape": [int(dim) for dim in arrays["states_uint8"].shape[2:]],
        "gamma": float(args.gamma),
        "indices_zero_based": indices,
        "action_convention": {
            "pytorch": "zero_based_action_indices",
            "deepmind": "one_based_action_indices",
        },
        "state_input_contract": {
            "pytorch_training": "float32 tensor with pixel values 0..255; QNetwork.forward divides by 255",
            "deepmind_transition_table": "float32 tensor with pixel values 0..1 from TransitionTable.sample",
        },
        "tensors": tensor_entries,
        "content_hash": arr_hash("".join(entry["hash"] for entry in tensor_entries).encode("utf-8")),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "indices.txt").write_text("\n".join(str(index) for index in indices) + "\n", encoding="utf-8")

    print(f"wrote {out_dir / 'manifest.json'}")
    print(f"batch_name: {args.batch_name}")
    print(f"batch_size: {manifest['batch_size']}")
    print(f"indices: {indices[:8]}{'...' if len(indices) > 8 else ''}")


if __name__ == "__main__":
    main()
