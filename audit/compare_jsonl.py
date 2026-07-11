#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_jsonl  # noqa: E402


class Difference(NamedTuple):
    path: str
    left: Any
    right: Any
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two JSONL traces.")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument(
        "--ignore",
        default=(
            "timestamp,runtime,path,source_path,source,env_id,rom,reset_info,"
            "info,dm_action_mode,lua_action_offset,lua_action,ale_action,"
            "action_meanings,ale_frame_number,dtype"
        ),
        help="Comma-separated leaf names or dotted paths to ignore.",
    )
    parser.add_argument("--float-tol", type=float, default=1e-6)
    parser.add_argument(
        "--ignore-diagnostic-metadata",
        action="store_true",
        help="Ignore version/provenance fields that explain the run but are not trace semantics.",
    )
    return parser.parse_args()


def ignored(path: str, ignores: set[str]) -> bool:
    leaf = path.rsplit(".", 1)[-1]
    return path in ignores or leaf in ignores


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def numbers_equal(left: Any, right: Any, tol: float) -> bool:
    left_float = float(left)
    right_float = float(right)
    if math.isnan(left_float) or math.isnan(right_float):
        return math.isnan(left_float) and math.isnan(right_float)
    return abs(left_float - right_float) <= tol


def compare_values(
    left: Any, right: Any, path: str, ignores: set[str], float_tol: float
) -> Difference | None:
    if ignored(path, ignores):
        return None

    if is_number(left) and is_number(right):
        if numbers_equal(left, right, float_tol):
            return None
        return Difference(path, left, right, "numeric values differ")

    if isinstance(left, dict) and isinstance(right, dict):
        left_keys = {key for key in left if not ignored(join(path, key), ignores)}
        right_keys = {key for key in right if not ignored(join(path, key), ignores)}
        if left_keys != right_keys:
            return Difference(
                path,
                sorted(left_keys - right_keys),
                sorted(right_keys - left_keys),
                "object keys differ",
            )
        for key in sorted(left_keys):
            diff = compare_values(
                left[key], right[key], join(path, key), ignores, float_tol
            )
            if diff is not None:
                return diff
        return None

    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return Difference(path, len(left), len(right), "list lengths differ")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            diff = compare_values(
                left_item,
                right_item,
                f"{path}[{index}]",
                ignores,
                float_tol,
            )
            if diff is not None:
                return diff
        return None

    if left != right:
        return Difference(path, left, right, "values differ")
    return None


def join(prefix: str, key: Any) -> str:
    key_text = str(key)
    return f"{prefix}.{key_text}" if prefix else key_text


def event_prefix(row: dict[str, Any], index: int) -> str:
    phase = row.get("phase")
    step = row.get("step")
    if isinstance(phase, str) and step is not None:
        return f"{phase}[{step}]"
    if isinstance(phase, str):
        return phase
    return f"event[{index}]"


def main() -> int:
    args = parse_args()
    ignores = {item.strip() for item in args.ignore.split(",") if item.strip()}
    if args.ignore_diagnostic_metadata:
        ignores.update(
            {
                "gymnasium_version",
                "ale_py_version",
                "alewrap_version",
                "ale_version",
                "frame_number",
                "score",
                "ram",
            }
        )

    left_rows = read_jsonl(args.left)
    right_rows = read_jsonl(args.right)

    for index, (left_row, right_row) in enumerate(zip(left_rows, right_rows)):
        diff = compare_values(
            left_row,
            right_row,
            event_prefix(left_row, index),
            ignores,
            args.float_tol,
        )
        if diff is not None:
            print("FIRST MISMATCH")
            print(f"event_index: {index}")
            print(f"path: {diff.path}")
            print(f"reason: {diff.reason}")
            print(f"{args.left_label}: {diff.left!r}")
            print(f"{args.right_label}: {diff.right!r}")
            return 1

    if len(left_rows) != len(right_rows):
        print("FIRST MISMATCH")
        print(f"event_index: {min(len(left_rows), len(right_rows))}")
        print("path: trace.length")
        print("reason: trace lengths differ")
        print(f"{args.left_label}: {len(left_rows)}")
        print(f"{args.right_label}: {len(right_rows)}")
        return 1

    print("MATCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
