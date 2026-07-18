#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import arr_hash, canonical_float, write_jsonl  # noqa: E402


ACTION_COUNT = 4
CLIP_DELTA = np.float32(1.0)


def parse_args() -> argparse.Namespace:
    base = os.getenv("BASE_OUTPUT_DIR", "audit_outputs")
    out = os.getenv("OUT", os.path.join(base, "stage5_loss"))
    parser = argparse.ArgumentParser(description="Trace PyTorch Stage 5C clipped TD-error/loss semantics.")
    parser.add_argument("--fixture-tsv", default=os.getenv("STAGE5C_FIXTURE_TSV", os.path.join(out, "loss_fixture.tsv")))
    parser.add_argument("--out", default=os.path.join(out, "pytorch_loss.jsonl"))
    parser.add_argument("--action-count", type=int, default=int(os.getenv("ACTION_COUNT", str(ACTION_COUNT))))
    return parser.parse_args()


def read_batches(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(row)
    batches: list[dict[str, Any]] = []
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_name.setdefault(row["batch_name"], []).append(row)
    for batch_name, items in by_name.items():
        items.sort(key=lambda item: int(item["batch_position"]))
        batches.append(
            {
                "batch_name": batch_name,
                "batch_kind": items[0]["batch_kind"],
                "batch_size": int(items[0]["batch_size"]),
                "samples": items,
            }
        )
    return batches


def f32(value: Any) -> np.float32:
    return np.float32(float(value))


def float_list(array: np.ndarray | list[float]) -> list[Any]:
    data = np.asarray(array, dtype=np.float32).reshape(-1)
    return [canonical_float(float(value), digits=10) for value in data.tolist()]


def tensor_summary(array: np.ndarray) -> dict[str, Any]:
    data = np.ascontiguousarray(array.astype(np.float32, copy=False))
    return {
        "shape": [int(dim) for dim in data.shape],
        "dtype": str(data.dtype),
        "hash": arr_hash(data),
        "first_values": float_list(data.reshape(-1)[: min(16, data.size)]),
        "abs_sum": canonical_float(float(np.abs(data).sum(dtype=np.float64)), digits=10),
        "nonzero_count": int(np.count_nonzero(data)),
    }


def huber_loss(delta: np.ndarray) -> np.ndarray:
    abs_delta = np.abs(delta)
    return np.where(abs_delta <= CLIP_DELTA, np.float32(0.5) * delta * delta, abs_delta - np.float32(0.5)).astype(np.float32)


def rows_for_batch(batch: dict[str, Any], action_count: int, step_start: int) -> list[dict[str, Any]]:
    samples = batch["samples"]
    batch_size = int(batch["batch_size"])
    selected_q = np.asarray([f32(sample["selected_q"]) for sample in samples], dtype=np.float32)
    targets = np.asarray([f32(sample["target"]) for sample in samples], dtype=np.float32)
    actions = np.asarray([int(sample["action"]) for sample in samples], dtype=np.int64)
    raw_td = (targets - selected_q).astype(np.float32)
    pred_minus_target = (selected_q - targets).astype(np.float32)
    abs_td = np.abs(raw_td).astype(np.float32)
    clipped = np.clip(raw_td, -CLIP_DELTA, CLIP_DELTA).astype(np.float32)
    per_sample_loss = huber_loss(raw_td)
    output_gradient = np.zeros((batch_size, action_count), dtype=np.float32)
    for i, action in enumerate(actions.tolist()):
        output_gradient[i, action] = clipped[i]
    smooth_l1_mean_dloss_dq = (-clipped / np.float32(batch_size)).astype(np.float32)

    rows: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        row_grad = output_gradient[i]
        rows.append(
            {
                "phase": "loss_sample",
                "source": "pytorch",
                "step": step_start + i,
                "batch_name": batch["batch_name"],
                "batch_kind": batch["batch_kind"],
                "batch_size": batch_size,
                "batch_position": i,
                "sample_id": sample["sample_id"],
                "sample_kind": sample["sample_kind"],
                "replay_index": int(sample["replay_index"]),
                "action": int(sample["action"]),
                "action_one_based": int(sample["action_one_based"]),
                "selected_q": canonical_float(float(selected_q[i]), digits=10),
                "target": canonical_float(float(targets[i]), digits=10),
                "raw_td_error": canonical_float(float(raw_td[i]), digits=10),
                "pred_minus_target": canonical_float(float(pred_minus_target[i]), digits=10),
                "abs_td_error": canonical_float(float(abs_td[i]), digits=10),
                "clip_delta": 1.0,
                "clipped_td_error": canonical_float(float(clipped[i]), digits=10),
                "strictly_inside_clip_region": bool(abs_td[i] < CLIP_DELTA),
                "at_clip_threshold": bool(abs_td[i] == CLIP_DELTA),
                "per_sample_huber_loss_proxy": canonical_float(float(per_sample_loss[i]), digits=10),
                "selected_action_output_gradient": canonical_float(float(clipped[i]), digits=10),
                "smooth_l1_mean_dloss_d_selected_q": canonical_float(float(smooth_l1_mean_dloss_dq[i]), digits=10),
                "output_gradient_row": float_list(row_grad),
                "output_gradient_nonzero_count": int(np.count_nonzero(row_grad)),
                "unselected_actions_zero": bool(np.count_nonzero(np.delete(row_grad, int(sample["action"]))) == 0),
                "terminal_flag": sample["terminal_flag"] == "1",
                "true_terminal": sample["true_terminal"] == "1",
                "life_loss_terminal": sample["life_loss_terminal"] == "1",
                "deepmind_source_path": "NeuralQLearner:getQUpdate",
                "scalar_loss_reported_by_deepmind": "none",
            }
        )

    rows.append(
        {
            "phase": "loss_batch",
            "source": "pytorch",
            "step": f"batch:{batch['batch_name']}",
            "batch_name": batch["batch_name"],
            "batch_kind": batch["batch_kind"],
            "batch_size": batch_size,
            "sample_ids": [sample["sample_id"] for sample in samples],
            "actions": [int(value) for value in actions.tolist()],
            "selected_q_values": float_list(selected_q),
            "targets": float_list(targets),
            "raw_td_errors": float_list(raw_td),
            "clipped_td_errors": float_list(clipped),
            "per_sample_huber_losses_proxy": float_list(per_sample_loss),
            "huber_loss_sum_proxy": canonical_float(float(per_sample_loss.sum(dtype=np.float64)), digits=10),
            "huber_loss_mean_proxy": canonical_float(float(per_sample_loss.mean(dtype=np.float64)), digits=10),
            "deepmind_batch_normalization_factor": 1.0,
            "deepmind_effective_reduction": "sparse_clipped_delta_no_batch_mean",
            "scalar_loss_reported_by_deepmind": "none",
            "smooth_l1_mean_scalar_proxy": canonical_float(float(per_sample_loss.mean(dtype=np.float64)), digits=10),
            "smooth_l1_mean_dloss_d_selected_q": float_list(smooth_l1_mean_dloss_dq),
            "output_gradient_tensor": tensor_summary(output_gradient),
            "selected_action_output_gradients": float_list(clipped),
            "unselected_actions_zero": bool(np.count_nonzero(output_gradient) == batch_size - int(np.count_nonzero(clipped == 0))),
        }
    )
    return rows


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    step = 0
    for batch in read_batches(args.fixture_tsv):
        batch_rows = rows_for_batch(batch, args.action_count, step)
        rows.extend(batch_rows)
        step += len(batch_rows)
    write_jsonl(args.out, rows)
    print(f"wrote {args.out}")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
