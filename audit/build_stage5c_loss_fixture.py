#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from common import arr_hash, canonical_float


THRESHOLD_DELTAS = [-2.0, -1.001, -1.0, -0.999, -0.5, 0.0, 0.5, 0.999, 1.0, 1.001, 2.0]
ACTION_COUNT = 4


def parse_args() -> argparse.Namespace:
    base = os.getenv("BASE_OUTPUT_DIR", "audit_outputs")
    out = os.getenv("OUT", os.path.join(base, "stage5_loss"))
    parser = argparse.ArgumentParser(description="Build Stage 5C loss/clipped-delta fixtures from Stage 5B traces.")
    parser.add_argument("--stage5b-dir", default=os.getenv("STAGE5_BELLMAN_DIR", os.path.join(base, "stage5_bellman")))
    parser.add_argument("--out-dir", default=out)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sanitize_delta(delta: float) -> str:
    return str(delta).replace("-", "m").replace(".", "p")


def real_sample(row: dict[str, Any], sample_id: str | None = None) -> dict[str, Any]:
    return {
        "sample_id": sample_id or f"real_{row['batch_name']}_{row['batch_position']}_{row['replay_index']}",
        "sample_kind": "real_stage5b",
        "source_batch_name": row["batch_name"],
        "replay_index": int(row["replay_index"]),
        "batch_position": int(row["batch_position"]),
        "action": int(row["action"]),
        "action_one_based": int(row["action_one_based"]),
        "selected_q": float(row["selected_q"]),
        "target": float(row["bellman_target"]),
        "reward": float(row["reward"]),
        "terminal_flag": bool(row["terminal_flag"]),
        "true_terminal": bool(row["true_terminal"]),
        "life_loss_terminal": bool(row["life_loss_terminal"]),
    }


def synthetic_sample(delta: float, position: int, sample_id: str | None = None) -> dict[str, Any]:
    action = int(position % ACTION_COUNT)
    return {
        "sample_id": sample_id or f"synthetic_delta_{sanitize_delta(delta)}_{position}",
        "sample_kind": "synthetic_threshold",
        "source_batch_name": "synthetic_threshold",
        "replay_index": -1,
        "batch_position": int(position),
        "action": action,
        "action_one_based": action + 1,
        "selected_q": 0.0,
        "target": float(delta),
        "reward": float(delta),
        "terminal_flag": False,
        "true_terminal": False,
        "life_loss_terminal": False,
    }


def make_batch(batch_name: str, batch_kind: str, samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "batch_name": batch_name,
        "batch_kind": batch_kind,
        "batch_size": len(samples),
        "samples": samples,
    }


def build_batches(stage5b_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    real_rows = [row for row in stage5b_rows if row.get("phase") == "bellman_sample"]
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in real_rows:
        by_name.setdefault(str(row["batch_name"]), []).append(row)
    for rows in by_name.values():
        rows.sort(key=lambda item: int(item["batch_position"]))

    required_real = [
        "batch_1",
        "batch_4",
        "batch_32",
        "ordinary_nonterminal",
        "true_terminal",
        "life_loss_terminal",
        "zero_reward",
        "positive_reward",
    ]
    missing = [name for name in required_real if name not in by_name]
    if missing:
        raise RuntimeError(f"Stage 5B trace missing required batches: {missing}")

    batches: list[dict[str, Any]] = [
        make_batch(name, "real_stage5b", [real_sample(row) for row in by_name[name]])
        for name in required_real
    ]

    for row in by_name["batch_4"]:
        batches.append(
            make_batch(
                f"real_single_{row['batch_position']}_{row['replay_index']}",
                "real_stage5b_single",
                [real_sample(row, sample_id=f"real_single_{row['batch_position']}_{row['replay_index']}")],
            )
        )

    threshold_samples = [synthetic_sample(delta, idx) for idx, delta in enumerate(THRESHOLD_DELTAS)]
    batches.extend(
        [
            make_batch("threshold_all", "synthetic_threshold", threshold_samples),
            make_batch(
                "threshold_mixed_4",
                "synthetic_threshold",
                [synthetic_sample(delta, idx, sample_id=f"synthetic_mixed_{idx}") for idx, delta in enumerate([-2.0, -0.5, 0.5, 2.0])],
            ),
            make_batch(
                "threshold_duplicated",
                "synthetic_threshold_duplicated",
                [
                    synthetic_sample(delta, idx, sample_id=f"synthetic_dup_{idx}")
                    for idx, delta in enumerate(THRESHOLD_DELTAS + THRESHOLD_DELTAS)
                ],
            ),
            make_batch(
                "threshold_batch_32",
                "synthetic_threshold_batch_32",
                [
                    synthetic_sample(THRESHOLD_DELTAS[idx % len(THRESHOLD_DELTAS)], idx, sample_id=f"synthetic_b32_{idx}")
                    for idx in range(32)
                ],
            ),
        ]
    )
    for idx, delta in enumerate(THRESHOLD_DELTAS):
        batches.append(
            make_batch(
                f"threshold_single_{sanitize_delta(delta)}",
                "synthetic_threshold_single",
                [synthetic_sample(delta, idx, sample_id=f"synthetic_single_{sanitize_delta(delta)}")],
            )
        )
    return batches


def write_tsv(path: Path, batches: list[dict[str, Any]]) -> None:
    header = [
        "batch_name",
        "batch_kind",
        "batch_size",
        "sample_id",
        "sample_kind",
        "source_batch_name",
        "batch_position",
        "replay_index",
        "action",
        "action_one_based",
        "selected_q",
        "target",
        "reward",
        "terminal_flag",
        "true_terminal",
        "life_loss_terminal",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for batch in batches:
            for position, sample in enumerate(batch["samples"]):
                row = {
                    "batch_name": batch["batch_name"],
                    "batch_kind": batch["batch_kind"],
                    "batch_size": batch["batch_size"],
                    **sample,
                    "batch_position": position,
                    "terminal_flag": "1" if sample["terminal_flag"] else "0",
                    "true_terminal": "1" if sample["true_terminal"] else "0",
                    "life_loss_terminal": "1" if sample["life_loss_terminal"] else "0",
                }
                writer.writerow(row)


def write_threshold_controls(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["raw_td_error", "expected_clipped_td_error", "abs_td_error", "strictly_inside_clip_region", "at_clip_threshold"])
        for delta in THRESHOLD_DELTAS:
            writer.writerow(
                [
                    f"{delta:.17g}",
                    f"{max(-1.0, min(1.0, delta)):.17g}",
                    f"{abs(delta):.17g}",
                    "1" if abs(delta) < 1.0 else "0",
                    "1" if abs(delta) == 1.0 else "0",
                ]
            )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage5b_dir = Path(args.stage5b_dir)
    stage5b_rows = read_jsonl(stage5b_dir / "pytorch_bellman.jsonl")
    batches = build_batches(stage5b_rows)
    fixture = {
        "phase": "stage5c_loss_fixture",
        "source": "stage5b_bellman_trace",
        "stage5b_dir": str(stage5b_dir),
        "action_count": ACTION_COUNT,
        "clip_delta": 1.0,
        "td_error_convention": "target_minus_selected_q",
        "prediction_error_convention": "selected_q_minus_target",
        "batches": batches,
    }
    payload = json.dumps(fixture, indent=2, sort_keys=True)
    fixture["content_hash"] = arr_hash(payload.encode("utf-8"))
    (out_dir / "loss_fixture.json").write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_tsv(out_dir / "loss_fixture.tsv", batches)
    write_threshold_controls(out_dir / "loss_threshold_controls.tsv")
    print(f"wrote {out_dir / 'loss_fixture.json'}")
    print(f"wrote {out_dir / 'loss_fixture.tsv'}")
    print(f"wrote {out_dir / 'loss_threshold_controls.tsv'}")
    print(f"batches: {len(batches)}")
    print(f"samples: {sum(len(batch['samples']) for batch in batches)}")


if __name__ == "__main__":
    main()
