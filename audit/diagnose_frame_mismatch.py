#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs")
    parser = argparse.ArgumentParser(description="Diagnose Stage 1 frame mismatch.")
    parser.add_argument("--frames-dir", default=os.path.join(out, "frames"))
    parser.add_argument("--max-step", type=int, default=int(os.getenv("FRAME_DUMP_STEPS", "20")))
    parser.add_argument("--out", default=os.getenv("FRAME_DIAGNOSIS_OUT"))
    return parser.parse_args()


def load_frame(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        numeric = arr.astype("float64", copy=False)
        if numeric.size and numeric.min() >= 0.0 and numeric.max() <= 1.0:
            numeric = numeric * 255.0
        arr = np.rint(np.clip(numeric, 0, 255)).astype(np.uint8)
    return np.ascontiguousarray(arr)


def load_optional(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return load_frame(path)


def sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def luminance(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[-1] >= 3:
        rgb = frame[..., :3].astype("float64")
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        return np.rint(np.clip(gray, 0, 255)).astype(np.uint8)
    return frame[..., 0]


def equal(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    return left is not None and right is not None and left.shape == right.shape and np.array_equal(left, right)


def diff_stats(left: np.ndarray, right: np.ndarray) -> dict[str, float | int | str]:
    if left.shape != right.shape:
        return {
            "shape_mismatch": f"{list(left.shape)} vs {list(right.shape)}",
            "max_abs_diff": -1,
            "mean_abs_diff": -1.0,
            "percent_pixels_equal": 0.0,
        }
    diff = np.abs(left.astype("int16") - right.astype("int16"))
    return {
        "max_abs_diff": int(diff.max()) if diff.size else 0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "percent_pixels_equal": float((left == right).mean() * 100.0) if diff.size else 100.0,
    }


def matching_step(target: np.ndarray, prefix: str, frames_dir: Path, max_step: int) -> str:
    matches = []
    for step in range(max_step + 1):
        candidate = load_optional(frames_dir / f"{prefix}_step_{step:03d}.npy")
        if equal(target, candidate):
            matches.append(str(step))
    return ", ".join(matches) if matches else "none"


def existing_files(frames_dir: Path) -> Iterable[str]:
    for path in sorted(frames_dir.glob("*")):
        if path.is_file():
            yield str(path)


def main() -> int:
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    p_init_path = frames_dir / "pytorch_init.npy"
    d_init_path = frames_dir / "deepmind_init.npy"
    if not p_init_path.exists() or not d_init_path.exists():
        raise SystemExit(
            f"missing init frames: {p_init_path} exists={p_init_path.exists()}, "
            f"{d_init_path} exists={d_init_path.exists()}"
        )

    p_init = load_frame(p_init_path)
    d_init = load_frame(d_init_path)

    rgb_bgr_swap_equal = False
    if p_init.shape == d_init.shape and p_init.ndim == 3 and p_init.shape[-1] >= 3:
        rgb_bgr_swap_equal = bool(np.array_equal(p_init[..., :3], d_init[..., 2::-1]))

    can_flip = p_init.shape == d_init.shape and d_init.ndim >= 2
    vertical_flip_equal = equal(p_init, np.flip(d_init, axis=0)) if can_flip else False
    horizontal_flip_equal = equal(p_init, np.flip(d_init, axis=1)) if can_flip else False
    both_flip_equal = equal(p_init, np.flip(np.flip(d_init, axis=0), axis=1)) if can_flip else False

    p_gray = luminance(p_init)
    d_gray = luminance(d_init)
    grayscale_equal = equal(p_gray, d_gray)
    stats = diff_stats(p_init, d_init)

    p_matches_d_step = matching_step(p_init, "deepmind", frames_dir, args.max_step)
    d_matches_p_step = matching_step(d_init, "pytorch", frames_dir, args.max_step)

    lines = [
        "Initial frame diagnosis",
        "",
        f"pytorch_init_shape: {list(p_init.shape)}",
        f"deepmind_init_shape: {list(d_init.shape)}",
        f"pytorch_init_hash: {sha256(p_init)}",
        f"deepmind_init_hash: {sha256(d_init)}",
        f"pytorch_luminance_hash: {sha256(p_gray)}",
        f"deepmind_luminance_hash: {sha256(d_gray)}",
        "",
        f"exact_equal: {str(equal(p_init, d_init)).lower()}",
        f"rgb_bgr_swap_equal: {str(rgb_bgr_swap_equal).lower()}",
        f"vertical_flip_equal: {str(vertical_flip_equal).lower()}",
        f"horizontal_flip_equal: {str(horizontal_flip_equal).lower()}",
        f"both_axes_flip_equal: {str(both_flip_equal).lower()}",
        f"grayscale_luminance_equal: {str(grayscale_equal).lower()}",
        f"pytorch_init_matches_deepmind_step: {p_matches_d_step}",
        f"deepmind_init_matches_pytorch_step: {d_matches_p_step}",
        f"mean_abs_diff: {stats['mean_abs_diff']}",
        f"max_abs_diff: {stats['max_abs_diff']}",
        f"percent_pixels_equal: {stats['percent_pixels_equal']}",
        "",
        "Conclusion:",
    ]

    if equal(p_init, d_init):
        lines.append("initial raw frames are byte-identical after dump normalization")
    elif rgb_bgr_swap_equal:
        lines.append("raw screen is the same image but RGB/BGR channel order differs")
    elif p_matches_d_step != "none" or d_matches_p_step != "none":
        lines.append("initial frames are offset by reset/no-op/first-step timing")
    elif grayscale_equal:
        lines.append("luminance matches but color channels differ")
    elif p_init.shape != d_init.shape:
        lines.append("frame shape/layout differs before pixel values are comparable")
    elif vertical_flip_equal or horizontal_flip_equal or both_flip_equal:
        lines.append("frame orientation differs")
    else:
        lines.append("not a simple channel-order, flip, or step-offset issue; likely reset/no-op/ALE initialization or backend mismatch")

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
