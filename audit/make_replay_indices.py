#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

from common import write_jsonl
from stage4_common import HIST_LEN, index_validity, load_records, sample_at


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs/stage4_replay_sample")
    parser = argparse.ArgumentParser(description="Generate explicit Stage 4 replay sample indices.")
    parser.add_argument("--replay-dir", default=os.path.join(out, "canonical_replay"))
    parser.add_argument("--requested-out", default=os.path.join(out, "requested_indices.txt"))
    parser.add_argument("--valid-out", default=os.path.join(out, "valid_indices.txt"))
    parser.add_argument("--rejected-out", default=os.path.join(out, "rejected_indices.jsonl"))
    return parser.parse_args()


def add_unique(target: list[int], values: list[int], capacity: int) -> None:
    seen = set(target)
    for value in values:
        if value < -8 or value > capacity + 8:
            continue
        if value not in seen:
            target.append(value)
            seen.add(value)


def main() -> None:
    args = parse_args()
    replay_dir = Path(args.replay_dir)
    records = load_records(replay_dir)
    capacity = len(records)
    max_start = capacity - HIST_LEN - 1
    wrap = int(records[0].get("wrap_insert_index", capacity // 2))

    terminal_action_indices = [
        index
        for index in range(1, max_start + 1)
        if bool(records[index + HIST_LEN - 1]["terminal_mask"])
    ]
    terminal_next_indices = [
        index
        for index in range(1, max_start + 1)
        if not records[index + HIST_LEN - 1]["terminal_mask"] and bool(records[index + HIST_LEN]["terminal_mask"])
    ]
    life_next_indices = [
        index
        for index in terminal_next_indices
        if bool(records[index + HIST_LEN]["life_loss_terminal"])
    ]
    true_terminal_next_indices = [
        index
        for index in terminal_next_indices
        if bool(records[index + HIST_LEN]["true_terminal"])
    ]
    history_boundary_indices = [
        index
        for index in range(1, max_start + 1)
        if not records[index + HIST_LEN - 1]["terminal_mask"]
        and any(bool(records[index + offset]["terminal_mask"]) for offset in range(HIST_LEN - 1))
    ]

    requested: list[int] = []
    add_unique(
        requested,
        [
            0,
            1,
            2,
            3,
            max_start - 1,
            max_start,
            max_start + 1,
            capacity,
            wrap - 8,
            wrap - 4,
            wrap - 1,
            wrap,
            wrap + 1,
            wrap + 4,
            wrap + 8,
        ],
        capacity,
    )
    for source in (
        terminal_action_indices[:8],
        [idx - 1 for idx in terminal_action_indices[:8]],
        [idx + 1 for idx in terminal_action_indices[:8]],
        terminal_next_indices[:8],
        life_next_indices[:8],
        true_terminal_next_indices[:8],
        history_boundary_indices[:12],
    ):
        add_unique(requested, source, capacity)

    ordinary_valid = [
        index
        for index in range(1, max_start + 1)
        if index_validity(records, index)[0]
        and index not in requested
        and not sample_at(records, index)["terminal_mask"]
        and not sample_at(records, index)["life_loss_terminal"]
        and not sample_at(records, index)["true_terminal"]
    ]
    stride = max(1, len(ordinary_valid) // 100)
    add_unique(requested, ordinary_valid[::stride][:100], capacity)

    invalid = [index for index in range(-2, capacity + 3) if not index_validity(records, index)[0]]
    add_unique(requested, invalid[:20], capacity)

    valid_indices: list[int] = []
    rejected_rows = []
    for position, index in enumerate(requested):
        accepted, reason = index_validity(records, index)
        if accepted:
            valid_indices.append(index)
        else:
            rejected_rows.append(
                {
                    "phase": "requested_index_rejection",
                    "request_position": position,
                    "requested_replay_index": index,
                    "accepted": False,
                    "rejection_reason": reason,
                }
            )

    Path(args.requested_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.requested_out).write_text("\n".join(str(index) for index in requested) + "\n", encoding="utf-8")
    Path(args.valid_out).write_text("\n".join(str(index) for index in valid_indices) + "\n", encoding="utf-8")
    write_jsonl(args.rejected_out, rejected_rows)
    print(f"wrote {args.requested_out}")
    print(f"wrote {args.valid_out}")
    print(f"wrote {args.rejected_out}")
    print(f"requested: {len(requested)}")
    print(f"valid: {len(valid_indices)}")
    print(f"rejected: {len(rejected_rows)}")


if __name__ == "__main__":
    main()
