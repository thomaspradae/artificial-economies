#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from common import arr_hash, arr_stats, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs/stage3_replay")
    stage2 = os.getenv("STAGE2_PREPROCESS_DIR", "audit_outputs/stage2_preprocess")
    parser = argparse.ArgumentParser(description="Build canonical DeepMind-processed 84x84 frame tape for Stage 3.")
    parser.add_argument("--stage2-dir", default=stage2)
    parser.add_argument("--out-dir", default=os.path.join(out, "canonical"))
    parser.add_argument("--hist-len", type=int, default=4)
    return parser.parse_args()


def clip_reward(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def load_processed_rows(stage2_dir: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(stage2_dir / "deepmind_preprocess.jsonl"):
        if row.get("phase") != "preprocess":
            continue
        rows[int(row["step"])] = row
    return rows


def load_env_rows(stage2_dir: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(stage2_dir / "canonical_frames" / "transitions.jsonl"):
        if row.get("phase") != "transition":
            continue
        rows[int(row["step"])] = row
    return rows


def canonical_frame(path: str | Path) -> np.ndarray:
    frame = np.asarray(np.load(path))
    if frame.shape != (84, 84):
        raise ValueError(f"expected 84x84 processed frame, got {frame.shape}: {path}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def main() -> None:
    args = parse_args()
    stage2_dir = Path(args.stage2_dir)
    out_dir = Path(args.out_dir)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    processed_rows = load_processed_rows(stage2_dir)
    env_rows = load_env_rows(stage2_dir)
    steps = sorted(set(processed_rows).intersection(env_rows))
    if not steps:
        raise RuntimeError(f"no overlapping Stage 2 processed/env rows in {stage2_dir}")

    transition_rows: list[dict[str, Any]] = []
    tsv_lines = [
        "\t".join(
            [
                "transition_index",
                "frame_index",
                "frame_path",
                "processed_frame_hash",
                "action_index",
                "ale_action_code",
                "raw_reward",
                "clipped_reward",
                "true_terminal",
                "life_loss",
                "terminal_mask",
                "lives_before",
                "lives_after",
            ]
        )
    ]

    manifest_hash_items: list[str] = []
    for transition_index, step in enumerate(steps):
        processed = processed_rows[step]
        env = env_rows[step]
        frame = canonical_frame(processed["processed_path"])
        frame_path = frame_dir / f"frame_{transition_index:06d}.npy"
        np.save(frame_path, frame)

        raw_reward = float(env.get("reward", 0))
        clipped = int(env.get("clipped_reward", clip_reward(raw_reward)))
        true_terminal = bool(env.get("terminal", False))
        life_loss = bool(env.get("life_loss", False))
        terminal_mask = bool(true_terminal or life_loss)
        frame_hash = arr_hash(frame)
        manifest_hash_items.append(frame_hash)

        row = {
            "phase": "canonical_processed_transition",
            "transition_index": transition_index,
            "source_step": step,
            "frame_index": transition_index,
            "frame_path": str(frame_path),
            "processed_frame_hash": frame_hash,
            "processed_frame": arr_stats(frame),
            "action_index": int(env.get("action_index", 0)),
            "ale_action_code": int(env.get("ale_action_code", env.get("action_index", 0))),
            "action_meaning": env.get("action_meaning"),
            "raw_reward": raw_reward,
            "clipped_reward": clipped,
            "true_terminal": true_terminal,
            "life_loss": life_loss,
            "terminal_mask": terminal_mask,
            "lives_before": env.get("lives_before"),
            "lives_after": env.get("lives_after"),
            "stage2_processed_path": processed["processed_path"],
            "stage2_pooled_path": env.get("pooled_path"),
        }
        transition_rows.append(row)

        tsv_lines.append(
            "\t".join(
                [
                    str(transition_index),
                    str(transition_index),
                    str(frame_path),
                    frame_hash,
                    str(row["action_index"]),
                    str(row["ale_action_code"]),
                    f"{raw_reward:.17g}",
                    str(clipped),
                    "1" if true_terminal else "0",
                    "1" if life_loss else "0",
                    "1" if terminal_mask else "0",
                    "" if row["lives_before"] is None else str(row["lives_before"]),
                    "" if row["lives_after"] is None else str(row["lives_after"]),
                ]
            )
        )

    write_jsonl(out_dir / "transitions.jsonl", transition_rows)
    (out_dir / "transitions.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
    (out_dir / "frame_paths.txt").write_text(
        "\n".join(f"{row['frame_index']}\t{row['frame_path']}" for row in transition_rows) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "phase": "canonical_processed_manifest",
        "source": "deepmind_stage2_processed_frames",
        "stage2_dir": str(stage2_dir),
        "out_dir": str(out_dir),
        "hist_len": int(args.hist_len),
        "frame_count": len(transition_rows),
        "transition_count": len(transition_rows),
        "frame_shape": [84, 84],
        "frame_dtype": "uint8",
        "value_range": [0, 255],
        "first_frame_hash": transition_rows[0]["processed_frame_hash"],
        "last_frame_hash": transition_rows[-1]["processed_frame_hash"],
        "content_hash": arr_hash("\n".join(manifest_hash_items).encode("utf-8")),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'transitions.jsonl'}")
    print(f"wrote {out_dir / 'transitions.tsv'}")
    print(f"wrote {out_dir / 'manifest.json'}")
    print(f"frames: {len(transition_rows)}")


if __name__ == "__main__":
    main()
