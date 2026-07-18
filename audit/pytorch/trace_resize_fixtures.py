#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import arr_stats, read_jsonl, write_jsonl  # noqa: E402
from pytorch.torch7_resize_clone import candidate_registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs/stage2c_resize")
    parser = argparse.ArgumentParser(description="Trace Python resize candidates on grayscale fixtures.")
    parser.add_argument("--fixtures", default=os.path.join(out, "fixtures", "fixtures.jsonl"))
    parser.add_argument("--out", default=os.path.join(out, "pytorch_outputs", "pytorch_resize.jsonl"))
    parser.add_argument("--output-dir", default=os.path.join(out, "pytorch_outputs", "arrays"))
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    fixtures = [row for row in read_jsonl(args.fixtures) if row.get("phase") == "resize_fixture"]

    for candidate in candidate_registry():
        candidate_dir = output_dir / safe_name(candidate.name)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        for fixture in fixtures:
            fixture_name = str(fixture["name"])
            input_frame = np.ascontiguousarray(np.load(fixture["path"]).astype(np.uint8))
            output_path = candidate_dir / f"{safe_name(fixture_name)}.npy"
            error = None
            try:
                output = np.ascontiguousarray(candidate.fn(input_frame).astype(np.uint8))
                np.save(output_path, output)
                output_stats = arr_stats(output)
            except Exception as exc:
                output = None
                output_stats = None
                error = f"{type(exc).__name__}: {exc}"

            rows.append(
                {
                    "phase": "resize_fixture",
                    "source": "pytorch",
                    "candidate": candidate.name,
                    "family": candidate.family,
                    "coordinate": candidate.coordinate,
                    "range_mode": candidate.range_mode,
                    "cast_rule": candidate.cast_rule,
                    "border_rule": candidate.border_rule,
                    "notes": candidate.notes,
                    "fixture_name": fixture_name,
                    "fixture_group": fixture["group"],
                    "input_path": fixture["path"],
                    "input_hash": fixture["hash"],
                    "output_path": str(output_path) if error is None else None,
                    "output_frame": output_stats,
                    "error": error,
                }
            )

    write_jsonl(args.out, rows)
    print(f"wrote {args.out}")
    print(f"candidates: {len(candidate_registry())}")
    print(f"fixtures: {len(fixtures)}")


if __name__ == "__main__":
    main()
