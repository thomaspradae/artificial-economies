#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from common import arr_hash, read_jsonl


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs/stage3_replay")
    parser = argparse.ArgumentParser(description="Validate Stage 3 canonical processed-frame tape.")
    parser.add_argument("--tape-dir", default=os.path.join(out, "canonical"))
    parser.add_argument("--report", default=os.path.join(out, "canonical_validation.txt"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tape_dir = Path(args.tape_dir)
    transitions_path = tape_dir / "transitions.jsonl"
    manifest_path = tape_dir / "manifest.json"
    tsv_path = tape_dir / "transitions.tsv"
    frame_paths_path = tape_dir / "frame_paths.txt"
    errors: list[str] = []

    for path in (transitions_path, manifest_path, tsv_path, frame_paths_path):
        if not path.exists():
            errors.append(f"missing required file: {path}")
    if errors:
        report = "\n".join(["Canonical processed tape validation", "status: fail", *errors]) + "\n"
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report, encoding="utf-8")
        print(report, end="")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [row for row in read_jsonl(transitions_path) if row.get("phase") == "canonical_processed_transition"]
    if len(rows) != int(manifest.get("transition_count", -1)):
        errors.append(f"transition count mismatch: jsonl={len(rows)} manifest={manifest.get('transition_count')}")
    if len(rows) != int(manifest.get("frame_count", -1)):
        errors.append(f"frame count mismatch: jsonl={len(rows)} manifest={manifest.get('frame_count')}")

    frame_hashes: list[str] = []
    for expected_index, row in enumerate(rows):
        if int(row.get("transition_index", -1)) != expected_index:
            errors.append(f"transition index mismatch at row {expected_index}: {row.get('transition_index')}")
        if int(row.get("frame_index", -1)) != expected_index:
            errors.append(f"frame index mismatch at row {expected_index}: {row.get('frame_index')}")
        frame_path = Path(row["frame_path"])
        if not frame_path.exists():
            errors.append(f"missing frame {expected_index}: {frame_path}")
            continue
        frame = np.asarray(np.load(frame_path))
        if frame.shape != (84, 84):
            errors.append(f"frame shape mismatch at {expected_index}: {frame.shape}")
        if frame.dtype != np.uint8:
            errors.append(f"frame dtype mismatch at {expected_index}: {frame.dtype}")
        if frame.size and (int(frame.min()) < 0 or int(frame.max()) > 255):
            errors.append(f"frame value range mismatch at {expected_index}: {frame.min()}..{frame.max()}")
        digest = arr_hash(np.ascontiguousarray(frame))
        frame_hashes.append(digest)
        if digest != row.get("processed_frame_hash"):
            errors.append(f"frame hash mismatch at {expected_index}: {digest} != {row.get('processed_frame_hash')}")
        stats = row.get("processed_frame") or {}
        if digest != stats.get("hash"):
            errors.append(f"frame stats hash mismatch at {expected_index}: {digest} != {stats.get('hash')}")

    tsv_rows = [line for line in tsv_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    frame_path_rows = [line for line in frame_paths_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(tsv_rows) - 1 != len(rows):
        errors.append(f"TSV transition count mismatch: {len(tsv_rows)-1} vs {len(rows)}")
    if len(frame_path_rows) != len(rows):
        errors.append(f"frame_paths count mismatch: {len(frame_path_rows)} vs {len(rows)}")

    content_hash = arr_hash("\n".join(frame_hashes).encode("utf-8"))
    if content_hash != manifest.get("content_hash"):
        errors.append(f"content hash mismatch: {content_hash} != {manifest.get('content_hash')}")

    lines = [
        "Canonical processed tape validation",
        f"tape_dir: {tape_dir}",
        f"frames: {len(rows)}",
        f"frame_shape: {manifest.get('frame_shape')}",
        f"frame_dtype: {manifest.get('frame_dtype')}",
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
