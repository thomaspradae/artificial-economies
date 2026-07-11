#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic action tape shared by both tracers."
    )
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "1")))
    parser.add_argument(
        "--steps", type=int, default=int(os.getenv("TRACE_STEPS", "200"))
    )
    parser.add_argument(
        "--action-count", type=int, default=int(os.getenv("ACTION_COUNT", "4"))
    )
    parser.add_argument(
        "--action-values",
        default=None,
        help="Comma-separated action values to sample from instead of 0..action-count-1.",
    )
    parser.add_argument(
        "--sequence",
        default=os.getenv("ACTION_TAPE_SEQUENCE") or None,
        help="Explicit comma sequence. Tokens may be VALUE or VALUExCOUNT, e.g. 1,1,1,1,0x20.",
    )
    parser.add_argument("--out-dir", default=os.getenv("OUT", "audit_outputs"))
    parser.add_argument("--prefix", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.action_count <= 0:
        raise SystemExit("--action-count must be positive")

    rng = np.random.default_rng(args.seed)
    if args.sequence:
        expanded: list[int] = []
        for token in args.sequence.split(","):
            token = token.strip()
            if not token:
                continue
            if "x" in token:
                value_text, count_text = token.split("x", 1)
                expanded.extend([int(value_text)] * int(count_text))
            else:
                expanded.append(int(token))
        if not expanded:
            raise SystemExit("--sequence did not contain any actions")
        actions = np.asarray(expanded[: args.steps], dtype=np.int64)
        if actions.size < args.steps:
            raise SystemExit("--sequence shorter than --steps")
    elif args.action_values:
        values = np.asarray([int(item.strip()) for item in args.action_values.split(",") if item.strip()], dtype=np.int64)
        if values.size == 0:
            raise SystemExit("--action-values did not contain any integers")
        actions = rng.choice(values, size=args.steps, replace=True).astype(np.int64)
    else:
        actions = rng.integers(0, args.action_count, size=args.steps, dtype=np.int64)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"action_tape_seed{args.seed}_{args.steps}"
    npy_path = out_dir / f"{prefix}.npy"
    txt_path = out_dir / f"{prefix}.txt"

    np.save(npy_path, actions)
    np.savetxt(txt_path, actions, fmt="%d")

    print(f"wrote {npy_path}")
    print(f"wrote {txt_path}")


if __name__ == "__main__":
    main()
