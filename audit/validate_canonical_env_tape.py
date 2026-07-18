#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import arr_hash, frame_to_uint8_hwc, read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs")
    parser = argparse.ArgumentParser(description="Validate a frozen canonical DeepMind env tape.")
    parser.add_argument("--tape-dir", default=os.path.join(out, "canonical_frames"))
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def load_frame(path: str | Path) -> np.ndarray:
    arr = np.load(path)
    return frame_to_uint8_hwc(arr)


def check_frame(path: str | Path, expected: dict[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    frame_path = Path(path)
    if not frame_path.exists():
        return [f"missing {label}: {frame_path}"]
    arr = load_frame(frame_path)
    expected_hash = expected.get("hash")
    actual_hash = arr_hash(arr)
    if expected_hash != actual_hash:
        errors.append(f"{label} hash mismatch: expected={expected_hash} actual={actual_hash} path={frame_path}")
    expected_shape = expected.get("shape")
    if expected_shape is not None and [int(dim) for dim in arr.shape] != list(expected_shape):
        errors.append(f"{label} shape mismatch: expected={expected_shape} actual={list(arr.shape)} path={frame_path}")
    return errors


def main() -> int:
    args = parse_args()
    tape_dir = Path(args.tape_dir)
    transitions_path = tape_dir / "transitions.jsonl"
    manifest_path = tape_dir / "manifest.json"
    manifest_jsonl_path = tape_dir / "manifest.jsonl"
    pooled_paths_path = tape_dir / "pooled_paths.txt"
    pooled_t7_paths_path = tape_dir / "pooled_t7_paths.txt"

    errors: list[str] = []
    for path in (transitions_path, manifest_path, manifest_jsonl_path, pooled_paths_path, pooled_t7_paths_path):
        if not path.exists():
            errors.append(f"missing required file: {path}")
    if errors:
        print("\n".join(errors))
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [row for row in read_jsonl(transitions_path) if row.get("phase") == "transition"]
    for index, row in enumerate(rows):
        errors.extend(check_frame(row["pre_frame_path"], row["pre_frame"], f"transition[{index}].pre_frame"))
        errors.extend(check_frame(row["pooled_path"], row["pooled_frame"], f"transition[{index}].pooled_frame"))
        if not Path(row["pooled_t7_path"]).exists():
            errors.append(f"missing transition[{index}].pooled_t7_path: {row['pooled_t7_path']}")
        for repeat in row.get("repeats", []):
            errors.extend(
                check_frame(
                    repeat["frame_path"],
                    repeat["raw_frame"],
                    f"transition[{index}].repeat[{repeat.get('repeat_i')}].raw_frame",
                )
            )
            if not Path(repeat["frame_t7_path"]).exists():
                errors.append(f"missing repeat t7 path: {repeat['frame_t7_path']}")

    pooled_lines = [line for line in pooled_paths_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    t7_lines = [line for line in pooled_t7_paths_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(pooled_lines) != len(rows):
        errors.append(f"pooled_paths count mismatch: {len(pooled_lines)} vs transitions {len(rows)}")
    if len(t7_lines) != len(rows):
        errors.append(f"pooled_t7_paths count mismatch: {len(t7_lines)} vs transitions {len(rows)}")

    lines = [
        "Canonical env tape validation",
        f"tape_dir: {tape_dir}",
        f"canonical_source: {manifest.get('canonical_source')}",
        f"seed: {manifest.get('seed')}",
        f"frame_skip: {manifest.get('frame_skip')}",
        f"transitions: {len(rows)}",
        f"status: {'pass' if not errors else 'fail'}",
    ]
    if errors:
        lines.append("errors:")
        lines.extend(errors[:100])
    report = "\n".join(lines) + "\n"
    print(report, end="")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report, encoding="utf-8")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
