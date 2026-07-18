#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import arr_stats, frame_to_uint8_hwc, read_jsonl, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs")
    parser = argparse.ArgumentParser(description="Trace PyTorch preprocessing on a frozen DeepMind frame tape.")
    parser.add_argument("--tape-dir", default=os.path.join(out, "canonical_frames"))
    parser.add_argument("--out", default=os.path.join(out, "pytorch_preprocess.jsonl"))
    parser.add_argument("--processed-dir", default=os.path.join(out, "pytorch_processed"))
    parser.add_argument(
        "--resize-interpolation",
        default=os.getenv("PYTORCH_RESIZE_INTERPOLATION", "area"),
        choices=("area", "bilinear"),
        help="PyTorch train_nature resize mode; current training default is area.",
    )
    parser.add_argument(
        "--preprocess-source",
        default=os.getenv("PYTORCH_PREPROCESS_SOURCE", "train_nature"),
        choices=("train_nature", "deepmind_clone"),
        help="Use train_nature preprocessing or a generated Stage 2c DeepMind clone.",
    )
    return parser.parse_args()


def load_preprocess(source: str) -> tuple[Callable[..., np.ndarray], str]:
    if source == "deepmind_clone":
        try:
            from pytorch import deepmind_preprocess

            def preprocess(obs: np.ndarray, previous_obs: np.ndarray | None = None, resize_interpolation: str = "area") -> np.ndarray:
                if previous_obs is not None:
                    obs = np.maximum(obs, previous_obs)
                return deepmind_preprocess.preprocess_frame(obs)

            return preprocess, "audit.pytorch.deepmind_preprocess.preprocess_frame"
        except Exception as exc:
            raise RuntimeError("PYTORCH_PREPROCESS_SOURCE=deepmind_clone requested, but generated clone could not be imported") from exc

    try:
        train_nature = importlib.import_module("train_nature")
        return train_nature.preprocess_frame, "train_nature.preprocess_frame"
    except Exception:
        pass

    def fallback(obs: np.ndarray, previous_obs: np.ndarray | None = None, resize_interpolation: str = "area") -> np.ndarray:
        if previous_obs is not None:
            obs = np.maximum(obs, previous_obs)
        try:
            import cv2

            interpolation = cv2.INTER_AREA if resize_interpolation == "area" else cv2.INTER_LINEAR
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2YUV)[:, :, 0]
            return cv2.resize(gray, (84, 84), interpolation=interpolation)
        except Exception:
            from PIL import Image

            rgb = obs[..., :3].astype("float64")
            gray = np.rint(np.clip(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2], 0, 255)).astype(np.uint8)
            resample = Image.Resampling.BOX if resize_interpolation == "area" else Image.Resampling.BILINEAR
            return np.asarray(Image.fromarray(gray).resize((84, 84), resample), dtype=np.uint8)

    return fallback, "audit_fallback_preprocess_frame"


def load_frame(path: str | Path) -> np.ndarray:
    return frame_to_uint8_hwc(np.load(path))


def byte_frame(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))


def main() -> None:
    args = parse_args()
    tape_dir = Path(args.tape_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    preprocess, preprocess_source = load_preprocess(args.preprocess_source)

    rows = []
    for row in read_jsonl(tape_dir / "transitions.jsonl"):
        if row.get("phase") != "transition":
            continue
        step = int(row["step"])
        frame = load_frame(row["pooled_path"])
        processed = byte_frame(preprocess(frame, resize_interpolation=args.resize_interpolation))
        processed_float = np.ascontiguousarray(processed.astype("float32") / 255.0)
        processed_path = processed_dir / f"processed_{step:06d}.npy"
        np.save(processed_path, processed)
        rows.append(
            {
                "phase": "preprocess",
                "source": "pytorch",
                "step": step,
                "input_path": row["pooled_path"],
                "input_hash": row["pooled_frame"]["hash"],
                "processed_path": str(processed_path),
                "processed_frame": arr_stats(processed),
                "processed_float": arr_stats(processed_float),
                "preprocess_source": preprocess_source,
                "preprocess_source_mode": args.preprocess_source,
                "resize_interpolation": args.resize_interpolation,
            }
        )

    write_jsonl(args.out, rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
