#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from common import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs/stage5_learner")
    parser = argparse.ArgumentParser(description="Compare PyTorch and DeepMind Stage 5A forward layer tensors.")
    parser.add_argument("--pytorch", default=os.path.join(out, "pytorch_forward_layers.jsonl"))
    parser.add_argument("--deepmind", default=os.path.join(out, "deepmind_forward_layers.jsonl"))
    parser.add_argument("--pytorch-tensor-dir", default=os.path.join(out, "forward_layers", "pytorch"))
    parser.add_argument("--deepmind-tensor-dir", default=os.path.join(out, "forward_layers", "deepmind"))
    parser.add_argument("--report", default=os.path.join(out, "intermediate_forward_compare.txt"))
    parser.add_argument("--jsonl", default=os.path.join(out, "intermediate_forward_compare.jsonl"))
    parser.add_argument("--layer-tol", type=float, default=float(os.getenv("STAGE5A_LAYER_TOL", "1e-6")))
    parser.add_argument("--q-tol", type=float, default=float(os.getenv("STAGE5A_Q_TOL", "1e-7")))
    return parser.parse_args()


def rows_by_key(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = {}
    for row in read_jsonl(path):
        if row.get("phase") != "forward_layer":
            continue
        rows[(str(row["batch_source"]), str(row["layer"]))] = row
    return rows


def load_tensor(row: dict[str, Any], root: str | Path) -> np.ndarray:
    shape = [int(dim) for dim in row["shape"]]
    path = Path(root) / row["tensor_file"]
    array = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise ValueError(f"{path}: expected {expected} float32 values, got {array.size}")
    return np.ascontiguousarray(array.reshape(shape))


def argmax_equal(left: np.ndarray, right: np.ndarray) -> tuple[int, int]:
    left_argmax = left.argmax(axis=1)
    right_argmax = right.argmax(axis=1)
    return int((left_argmax == right_argmax).sum()), int(left_argmax.size)


def main() -> int:
    args = parse_args()
    pytorch_rows = rows_by_key(args.pytorch)
    deepmind_rows = rows_by_key(args.deepmind)
    keys = sorted(set(pytorch_rows) | set(deepmind_rows))
    result_rows = []
    errors: list[str] = []

    for key in keys:
        if key not in pytorch_rows:
            errors.append(f"missing PyTorch row: {key}")
            continue
        if key not in deepmind_rows:
            errors.append(f"missing DeepMind row: {key}")
            continue
        left_row = pytorch_rows[key]
        right_row = deepmind_rows[key]
        left = load_tensor(left_row, args.pytorch_tensor_dir)
        right = load_tensor(right_row, args.deepmind_tensor_dir)
        shape_equal = left.shape == right.shape
        if not shape_equal:
            errors.append(f"{key}: shape mismatch {left.shape} vs {right.shape}")
            continue
        diff = left.astype(np.float64) - right.astype(np.float64)
        max_abs = float(np.max(np.abs(diff)))
        mean_abs = float(np.mean(np.abs(diff)))
        mean_signed = float(np.mean(diff))
        tolerance = args.q_tol if key[1] == "q_values" else args.layer_tol
        passed = max_abs <= tolerance
        argmax_match = None
        argmax_total = None
        if key[1] == "q_values":
            argmax_match, argmax_total = argmax_equal(left, right)
            if argmax_match != argmax_total:
                passed = False
                errors.append(f"{key}: argmax mismatch {argmax_match}/{argmax_total}")
        if not passed:
            errors.append(f"{key}: max_abs_diff {max_abs:.12g} > {tolerance:.12g}")
        result_rows.append(
            {
                "phase": "forward_layer_compare",
                "batch_source": key[0],
                "layer": key[1],
                "shape": [int(dim) for dim in left.shape],
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs,
                "mean_signed_diff": mean_signed,
                "tolerance": tolerance,
                "passed": passed,
                "argmax_match": argmax_match,
                "argmax_total": argmax_total,
            }
        )

    write_jsonl(args.jsonl, result_rows)
    status = "PASS" if not errors else "FAIL"
    q_rows = [row for row in result_rows if row["layer"] == "q_values"]
    max_q = max((row["max_abs_diff"] for row in q_rows), default=float("nan"))
    argmax_match = sum(int(row["argmax_match"] or 0) for row in q_rows)
    argmax_total = sum(int(row["argmax_total"] or 0) for row in q_rows)
    lines = [
        "Stage 5A intermediate forward comparison",
        f"status: {status}",
        f"layer_tolerance: {args.layer_tol}",
        f"q_value_tolerance: {args.q_tol}",
        f"compared_layers: {len(result_rows)}",
        f"q_value_max_abs_diff: {max_q:.12g}",
        f"q_value_argmax_agreement: {argmax_match}/{argmax_total}",
        "",
        "Per-layer diffs:",
    ]
    for row in result_rows:
        lines.append(
            f"{row['batch_source']}.{row['layer']}: "
            f"shape={row['shape']} max_abs={row['max_abs_diff']:.12g} "
            f"mean_abs={row['mean_abs_diff']:.12g} mean_signed={row['mean_signed_diff']:.12g} "
            f"passed={row['passed']}"
            + (
                f" argmax={row['argmax_match']}/{row['argmax_total']}"
                if row["argmax_total"] is not None
                else ""
            )
        )
    if errors:
        lines.append("")
        lines.append("Errors:")
        lines.extend(errors[:100])
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
