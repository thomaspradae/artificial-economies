#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from common import arr_hash, arr_stats, read_jsonl, write_jsonl
from stage4_common import ACTION_TO_ALE, CAPACITY_DEFAULT, TOTAL_INSERTIONS_DEFAULT, FRAME_SHAPE, HIST_LEN


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs/stage4_replay_sample")
    stage3 = os.getenv("STAGE3_REPLAY_DIR", "audit_outputs/stage3_replay")
    parser = argparse.ArgumentParser(description="Build Stage 4 canonical replay fixture from Stage 3 processed frames.")
    parser.add_argument("--stage3-dir", default=stage3)
    parser.add_argument("--out-dir", default=os.path.join(out, "canonical_replay"))
    parser.add_argument("--capacity", type=int, default=CAPACITY_DEFAULT)
    parser.add_argument("--total-insertions", type=int, default=TOTAL_INSERTIONS_DEFAULT)
    return parser.parse_args()


def load_stage3_frames(stage3_dir: Path) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_jsonl(stage3_dir / "canonical" / "transitions.jsonl")
        if row.get("phase") == "canonical_processed_transition"
    ]
    rows.sort(key=lambda row: int(row["transition_index"]))
    if not rows:
        raise RuntimeError(f"missing Stage 3 canonical processed transitions under {stage3_dir}")
    return rows


def frame_for_insertion(stage3_rows: list[dict[str, Any]], insertion_index: int) -> np.ndarray:
    row = stage3_rows[insertion_index % len(stage3_rows)]
    frame = np.asarray(np.load(row["frame_path"]))
    if frame.shape != FRAME_SHAPE:
        raise ValueError(f"expected {FRAME_SHAPE}, got {frame.shape}: {row['frame_path']}")
    return np.ascontiguousarray(frame.astype(np.uint8, copy=False))


def terminal_flags(position_in_episode: int, absolute_index: int) -> tuple[bool, bool]:
    true_terminal = position_in_episode == 172
    life_loss = position_in_episode in {46, 93, 140} and not true_terminal
    if absolute_index % 211 == 17:
        life_loss = True
    return true_terminal, life_loss and not true_terminal


def reward_for(absolute_index: int, position_in_episode: int, true_terminal: bool) -> float:
    if true_terminal:
        return 0.0
    if absolute_index % 37 in {0, 1, 2}:
        return 1.0
    if position_in_episode in {80, 81}:
        return 1.0
    return 0.0


def main() -> None:
    args = parse_args()
    if args.capacity < 32:
        raise ValueError("capacity must be at least 32")
    if args.total_insertions < args.capacity:
        raise ValueError("total insertions must be >= capacity to exercise wraparound")

    out_dir = Path(args.out_dir)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    stage3_rows = load_stage3_frames(Path(args.stage3_dir))

    storage: list[dict[str, Any] | None] = [None] * args.capacity
    episode_id = 0
    frame_position = 0
    insertion_rows: list[dict[str, Any]] = []
    wrap_insert_index = args.total_insertions % args.capacity

    for absolute_index in range(args.total_insertions):
        if frame_position == 0 and absolute_index != 0:
            episode_id += 1
        frame = frame_for_insertion(stage3_rows, absolute_index)
        slot = absolute_index % args.capacity
        true_terminal, life_loss = terminal_flags(frame_position, absolute_index)
        terminal_mask = true_terminal or life_loss
        raw_reward = reward_for(absolute_index, frame_position, true_terminal)
        clipped_reward = 1 if raw_reward > 0 else -1 if raw_reward < 0 else 0
        action = (absolute_index + int(stage3_rows[absolute_index % len(stage3_rows)].get("action_index", 0))) % 4

        record = {
            "phase": "stage4_replay_record",
            "replay_index": slot,
            "absolute_insert_index": absolute_index,
            "storage_slot": slot,
            "episode_id": episode_id,
            "frame_position_in_episode": frame_position,
            "frame_path": str(frame_dir / f"frame_{slot:06d}.npy"),
            "processed_frame_hash": arr_hash(frame),
            "processed_frame": arr_stats(frame),
            "action": int(action),
            "action_one_based": int(action) + 1,
            "ale_action_code": ACTION_TO_ALE[int(action)],
            "raw_reward": raw_reward,
            "clipped_reward": clipped_reward,
            "true_terminal": true_terminal,
            "life_loss_terminal": life_loss,
            "terminal_mask": terminal_mask,
            "wrap_insert_index": wrap_insert_index,
        }
        np.save(record["frame_path"], frame)
        storage[slot] = record
        insertion_rows.append(
            {
                "phase": "stage4_replay_insertion",
                "absolute_insert_index": absolute_index,
                "storage_slot": slot,
                "episode_id": episode_id,
                "frame_position_in_episode": frame_position,
                "action": int(action),
                "action_one_based": int(action) + 1,
                "ale_action_code": ACTION_TO_ALE[int(action)],
                "raw_reward": raw_reward,
                "clipped_reward": clipped_reward,
                "true_terminal": true_terminal,
                "life_loss_terminal": life_loss,
                "terminal_mask": terminal_mask,
                "frame_path": record["frame_path"],
                "processed_frame_hash": record["processed_frame_hash"],
            }
        )

        frame_position += 1
        if true_terminal:
            frame_position = 0

    records = [record for record in storage if record is not None]
    records.sort(key=lambda row: int(row["replay_index"]))
    if len(records) != args.capacity:
        raise RuntimeError("fixture storage was not fully populated")

    for record in records:
        replay_index = int(record["replay_index"])
        sampleable = 1 <= replay_index <= args.capacity - HIST_LEN - 1 and not bool(records[replay_index + HIST_LEN - 1]["terminal_mask"])
        reason = "sampleable" if sampleable else (
            "before_earliest_deepmind_sample_index"
            if replay_index < 1
            else "after_latest_deepmind_sample_index"
            if replay_index > args.capacity - HIST_LEN - 1
            else "terminal_at_action_frame"
        )
        record["sampleable_as_start"] = sampleable
        record["sampleability_reason"] = reason

    write_jsonl(out_dir / "records.jsonl", records)
    write_jsonl(out_dir / "insertions.jsonl", insertion_rows)
    header = [
        "replay_index",
        "absolute_insert_index",
        "frame_path",
        "processed_frame_hash",
        "action",
        "action_one_based",
        "ale_action_code",
        "raw_reward",
        "clipped_reward",
        "true_terminal",
        "life_loss_terminal",
        "terminal_mask",
        "episode_id",
        "frame_position_in_episode",
        "sampleable_as_start",
        "sampleability_reason",
        "wrap_insert_index",
    ]
    lines = ["\t".join(header)]
    for record in records:
        lines.append(
            "\t".join(
                [
                    str(record["replay_index"]),
                    str(record["absolute_insert_index"]),
                    str(record["frame_path"]),
                    str(record["processed_frame_hash"]),
                    str(record["action"]),
                    str(record["action_one_based"]),
                    str(record["ale_action_code"]),
                    f"{float(record['raw_reward']):.17g}",
                    str(record["clipped_reward"]),
                    "1" if record["true_terminal"] else "0",
                    "1" if record["life_loss_terminal"] else "0",
                    "1" if record["terminal_mask"] else "0",
                    str(record["episode_id"]),
                    str(record["frame_position_in_episode"]),
                    "1" if record["sampleable_as_start"] else "0",
                    str(record["sampleability_reason"]),
                    str(wrap_insert_index),
                ]
            )
        )
    (out_dir / "records.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    content_hash = arr_hash("\n".join(record["processed_frame_hash"] for record in records).encode("utf-8"))
    manifest = {
        "phase": "stage4_replay_manifest",
        "source": "stage3_canonical_processed_frames_with_synthetic_boundaries",
        "stage3_dir": str(Path(args.stage3_dir)),
        "out_dir": str(out_dir),
        "capacity": args.capacity,
        "total_insertions": args.total_insertions,
        "wrap_insert_index": wrap_insert_index,
        "record_count": len(records),
        "hist_len": HIST_LEN,
        "frame_shape": [84, 84],
        "frame_dtype": "uint8",
        "content_hash": content_hash,
        "sampleable_count": sum(1 for row in records if row["sampleable_as_start"]),
        "terminal_count": sum(1 for row in records if row["true_terminal"]),
        "life_loss_count": sum(1 for row in records if row["life_loss_terminal"]),
        "reward_count": sum(1 for row in records if row["clipped_reward"] != 0),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'records.jsonl'}")
    print(f"wrote {out_dir / 'records.tsv'}")
    print(f"wrote {out_dir / 'insertions.jsonl'}")
    print(f"wrote {out_dir / 'manifest.json'}")
    print(f"records: {len(records)}")
    print(f"sampleable: {manifest['sampleable_count']}")


if __name__ == "__main__":
    main()
