#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from common import arr_hash, read_jsonl
from stage4_common import HIST_LEN, index_validity, load_records, load_requested_indices, sample_at


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs/stage4_replay_sample")
    parser = argparse.ArgumentParser(description="Validate Stage 4 canonical replay fixture and explicit indices.")
    parser.add_argument("--replay-dir", default=os.path.join(out, "canonical_replay"))
    parser.add_argument("--requested", default=os.path.join(out, "requested_indices.txt"))
    parser.add_argument("--valid", default=os.path.join(out, "valid_indices.txt"))
    parser.add_argument("--rejected", default=os.path.join(out, "rejected_indices.jsonl"))
    parser.add_argument("--report", default=os.path.join(out, "fixture_validation.txt"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    replay_dir = Path(args.replay_dir)
    manifest_path = replay_dir / "manifest.json"
    errors: list[str] = []

    if not manifest_path.exists():
        errors.append(f"missing manifest: {manifest_path}")
        manifest = {}
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    try:
        records = load_records(replay_dir)
    except Exception as exc:
        records = []
        errors.append(f"could not load records: {exc}")

    if records:
        expected_indices = list(range(len(records)))
        actual_indices = [int(row["replay_index"]) for row in records]
        if actual_indices != expected_indices:
            errors.append("replay indices are not unique ordered 0..N-1")
        if len(records) != int(manifest.get("record_count", len(records))):
            errors.append(f"record count mismatch: {len(records)} vs {manifest.get('record_count')}")

    frame_hashes: list[str] = []
    for row in records:
        for field in ("action", "clipped_reward", "true_terminal", "life_loss_terminal", "terminal_mask", "episode_id", "frame_position_in_episode"):
            if field not in row:
                errors.append(f"missing field {field} at replay_index {row.get('replay_index')}")
        frame_path = Path(row.get("frame_path", ""))
        if not frame_path.exists():
            errors.append(f"missing frame at replay_index {row.get('replay_index')}: {frame_path}")
            continue
        frame = np.asarray(np.load(frame_path))
        if frame.shape != (84, 84):
            errors.append(f"frame shape mismatch at replay_index {row.get('replay_index')}: {frame.shape}")
        if frame.dtype != np.uint8:
            errors.append(f"frame dtype mismatch at replay_index {row.get('replay_index')}: {frame.dtype}")
        digest = arr_hash(np.ascontiguousarray(frame))
        frame_hashes.append(digest)
        if digest != row.get("processed_frame_hash"):
            errors.append(f"frame hash mismatch at replay_index {row.get('replay_index')}")

    terminal_count = sum(1 for row in records if row.get("true_terminal"))
    life_loss_count = sum(1 for row in records if row.get("life_loss_terminal"))
    reward_count = sum(1 for row in records if row.get("clipped_reward") != 0)
    if terminal_count == 0:
        errors.append("fixture has no true terminal records")
    if life_loss_count == 0:
        errors.append("fixture has no life-loss terminal records")
    if reward_count == 0:
        errors.append("fixture has no nonzero rewards")

    if Path(args.requested).exists():
        requested = load_requested_indices(args.requested)
    else:
        requested = []
        errors.append(f"missing requested indices: {args.requested}")
    valid_file = load_requested_indices(args.valid) if Path(args.valid).exists() else []
    rejected_rows = read_jsonl(args.rejected) if Path(args.rejected).exists() else []

    if len(requested) != len(set(requested)):
        errors.append("requested indices contain duplicates")

    expected_valid: list[int] = []
    expected_rejected: dict[int, str] = {}
    for index in requested:
        accepted, reason = index_validity(records, index)
        if accepted:
            expected_valid.append(index)
            sample = sample_at(records, index)
            state_indices = [idx for idx in sample["state_source_frame_indices"] if idx is not None]
            next_indices = [idx for idx in sample["next_state_source_frame_indices"] if idx is not None]
            if max(sample["state_raw_frame_indices"]) >= len(records) or max(sample["next_state_raw_frame_indices"]) >= len(records):
                errors.append(f"valid index out of bounds after reconstruction: {index}")
            for idx in state_indices + next_indices:
                if idx < 0 or idx >= len(records):
                    errors.append(f"valid index has bad component {idx}: requested {index}")
        else:
            expected_rejected[index] = reason

    if valid_file != expected_valid:
        errors.append(f"valid_indices mismatch: file={valid_file[:20]} expected={expected_valid[:20]}")
    rejected_by_index = {int(row["requested_replay_index"]): row["rejection_reason"] for row in rejected_rows}
    if rejected_by_index != expected_rejected:
        errors.append("rejected_indices mismatch")

    content_hash = arr_hash("\n".join(frame_hashes).encode("utf-8"))
    if manifest and content_hash != manifest.get("content_hash"):
        errors.append(f"content hash mismatch: {content_hash} != {manifest.get('content_hash')}")

    lines = [
        "Stage 4 replay fixture validation",
        f"replay_dir: {replay_dir}",
        f"records: {len(records)}",
        f"requested_indices: {len(requested)}",
        f"valid_indices: {len(expected_valid)}",
        f"rejected_indices: {len(expected_rejected)}",
        f"true_terminals: {terminal_count}",
        f"life_loss_terminals: {life_loss_count}",
        f"nonzero_rewards: {reward_count}",
        f"content_hash: {content_hash}",
        f"status: {'pass' if not errors else 'fail'}",
    ]
    if errors:
        lines.append("errors:")
        lines.extend(errors[:100])
    report = "\n".join(lines) + "\n"
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(report, end="")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
