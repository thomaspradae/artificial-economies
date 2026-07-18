#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import arr_hash, frame_to_uint8_hwc, read_jsonl  # noqa: E402


Array = np.ndarray
VariantFn = Callable[[Array], Array]


@dataclass
class Variant:
    name: str
    fn: VariantFn


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs")
    parser = argparse.ArgumentParser(description="Rank PyTorch preprocessing variants against frozen DeepMind outputs.")
    parser.add_argument("--tape-dir", default=os.path.join(out, "canonical_frames"))
    parser.add_argument("--deepmind", default=os.path.join(out, "deepmind_preprocess.jsonl"))
    parser.add_argument("--intermediate-dir", default=os.path.join(out, "deepmind_preprocess_intermediates"))
    parser.add_argument("--out", default=os.path.join(out, "preprocess_variants_ranked.txt"))
    parser.add_argument("--jsonl", default=os.path.join(out, "preprocess_variants_ranked.jsonl"))
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames.")
    parser.add_argument(
        "--write-matching-implementation",
        default="",
        help="Write audit/pytorch/deepmind_preprocess.py when a variant matches every frame exactly.",
    )
    return parser.parse_args()


def load_frame(path: str | Path) -> Array:
    return frame_to_uint8_hwc(np.load(path))


def quantize(value: Array, mode: str) -> Array:
    arr = np.asarray(value, dtype=np.float64)
    if mode == "trunc":
        out = arr
    elif mode == "floor":
        out = np.floor(arr)
    elif mode == "round_half_up":
        out = np.floor(arr + 0.5)
    elif mode == "rint":
        out = np.rint(arr)
    else:
        raise ValueError(f"unknown quantize mode: {mode}")
    return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))


def quantize_unit(value: Array, mode: str) -> Array:
    return quantize(np.asarray(value, dtype=np.float32) * np.float32(255.0), mode)


def gray_bt601_float(frame: Array) -> Array:
    rgb = frame.astype(np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def gray_bt601_unit_float32(frame: Array) -> Array:
    rgb = np.ascontiguousarray(frame, dtype=np.float32) / np.float32(255.0)
    return (
        np.float32(0.299) * rgb[..., 0]
        + np.float32(0.587) * rgb[..., 1]
        + np.float32(0.114) * rgb[..., 2]
    )


def gray_bt601_video_float(frame: Array) -> Array:
    rgb = frame.astype(np.float64)
    return 16.0 + (65.481 * rgb[..., 0] + 128.553 * rgb[..., 1] + 24.966 * rgb[..., 2]) / 255.0


def gray_rgb_mean_float(frame: Array) -> Array:
    return frame.astype(np.float64).mean(axis=2)


_AREA_WEIGHT_CACHE: dict[tuple[int, int], Array] = {}


def torch_image_area_weights(input_size: int, output_size: int) -> Array:
    """Weights matching Torch image.scaleBilinear's downsample box behavior.

    Torch's historical image.scaleBilinear acts like area resampling when the
    target is smaller than the source. Destination pixel i covers source-space
    interval [i * scale - 0.5, (i + 1) * scale - 0.5], where source pixel k
    covers [k - 0.5, k + 0.5].
    """

    key = (input_size, output_size)
    cached = _AREA_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    scale = float(input_size) / float(output_size)
    weights = np.zeros((output_size, input_size), dtype=np.float32)
    for out_index in range(output_size):
        start = out_index * scale - 0.5
        end = (out_index + 1) * scale - 0.5
        first = max(0, int(math.floor(start + 0.5)))
        last = min(input_size - 1, int(math.ceil(end + 0.5)))
        for in_index in range(first, last + 1):
            left = max(start, in_index - 0.5)
            right = min(end, in_index + 0.5)
            overlap = max(0.0, right - left)
            if overlap:
                weights[out_index, in_index] = np.float32(overlap / scale)
    _AREA_WEIGHT_CACHE[key] = weights
    return weights


def torch_image_area_resize(gray: Array, output_shape: tuple[int, int] = (84, 84)) -> Array:
    gray32 = np.ascontiguousarray(gray, dtype=np.float32)
    out_h, out_w = output_shape
    weights_y = torch_image_area_weights(gray32.shape[0], out_h)
    weights_x = torch_image_area_weights(gray32.shape[1], out_w)
    return np.ascontiguousarray((weights_y @ gray32) @ weights_x.T)


def maybe_cv2() -> Any:
    try:
        import cv2

        return cv2
    except Exception:
        return None


def maybe_pil_image() -> Any:
    try:
        from PIL import Image

        return Image
    except Exception:
        return None


def maybe_skimage_resize() -> Any:
    try:
        from skimage.transform import resize

        return resize
    except Exception:
        return None


def cv2_gray(frame: Array, mode: str) -> Array:
    cv2 = maybe_cv2()
    if cv2 is None:
        raise RuntimeError("cv2 unavailable")
    if mode == "rgb2yuv_y":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)[:, :, 0]
    if mode == "rgb2gray":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    raise ValueError(mode)


def cv2_resize(gray: Array, interpolation_name: str) -> Array:
    cv2 = maybe_cv2()
    if cv2 is None:
        raise RuntimeError("cv2 unavailable")
    interpolation = {
        "area": cv2.INTER_AREA,
        "linear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
        "cubic": cv2.INTER_CUBIC,
    }[interpolation_name]
    return cv2.resize(gray, (84, 84), interpolation=interpolation)


def pil_resize(gray: Array, resample_name: str) -> Array:
    Image = maybe_pil_image()
    if Image is None:
        raise RuntimeError("PIL unavailable")
    resampling = getattr(Image, "Resampling", Image)
    resample = {
        "bilinear": resampling.BILINEAR,
        "box": resampling.BOX,
        "nearest": resampling.NEAREST,
    }[resample_name]
    source = quantize(gray, "round_half_up") if gray.dtype != np.uint8 else gray
    return np.asarray(Image.fromarray(source).resize((84, 84), resample), dtype=np.uint8)


def skimage_resize(gray: Array, anti_aliasing: bool, quantize_mode: str) -> Array:
    resize = maybe_skimage_resize()
    if resize is None:
        raise RuntimeError("skimage unavailable")
    resized = resize(
        np.asarray(gray, dtype=np.float64),
        (84, 84),
        order=1,
        mode="edge",
        preserve_range=True,
        anti_aliasing=anti_aliasing,
    )
    return quantize(resized, quantize_mode)


def add_variant(variants: list[Variant], name: str, fn: VariantFn) -> None:
    variants.append(Variant(name=name, fn=fn))


def build_variants() -> list[Variant]:
    variants: list[Variant] = []

    for channel_order, order_fn in (
        ("rgb", lambda frame: frame),
        ("bgr", lambda frame: frame[..., ::-1]),
    ):
        for cv_gray in ("rgb2yuv_y", "rgb2gray"):
            for interpolation in ("area", "linear"):
                add_variant(
                    variants,
                    f"cv2_{channel_order}_{cv_gray}_{interpolation}_uint8",
                    lambda frame, order_fn=order_fn, cv_gray=cv_gray, interpolation=interpolation: np.ascontiguousarray(
                        cv2_resize(cv2_gray(order_fn(frame), cv_gray), interpolation).astype(np.uint8)
                    ),
                )

        gray_fns: list[tuple[str, Callable[[Array], Array]]] = [
            ("bt601_full_float", gray_bt601_float),
            ("bt601_video_float", gray_bt601_video_float),
            ("rgb_mean_float", gray_rgb_mean_float),
        ]
        for gray_name, gray_fn in gray_fns:
            for interpolation in ("area", "linear"):
                for qmode in ("trunc", "floor", "round_half_up", "rint"):
                    add_variant(
                        variants,
                        f"cv2_{channel_order}_{gray_name}_{interpolation}_{qmode}",
                        lambda frame, order_fn=order_fn, gray_fn=gray_fn, interpolation=interpolation, qmode=qmode: quantize(
                            cv2_resize(gray_fn(order_fn(frame)), interpolation), qmode
                        ),
                    )

        for interpolation in ("area", "linear"):
            for qmode in ("trunc", "floor", "round_half_up", "rint"):
                add_variant(
                    variants,
                    f"cv2_{channel_order}_bt601_unit_float32_{interpolation}_mul255_{qmode}",
                    lambda frame, order_fn=order_fn, interpolation=interpolation, qmode=qmode: quantize_unit(
                        cv2_resize(gray_bt601_unit_float32(order_fn(frame)), interpolation), qmode
                    ),
                )

        for gray_name, gray_fn in (("bt601_full_float", gray_bt601_float), ("rgb_mean_float", gray_rgb_mean_float)):
            for qmode in ("trunc", "floor", "round_half_up", "rint"):
                add_variant(
                    variants,
                    f"torch_area_{channel_order}_{gray_name}_{qmode}",
                    lambda frame, order_fn=order_fn, gray_fn=gray_fn, qmode=qmode: quantize(
                        torch_image_area_resize(gray_fn(order_fn(frame))), qmode
                    ),
                )

        for qmode in ("trunc", "floor", "round_half_up", "rint"):
            add_variant(
                variants,
                f"torch_area_{channel_order}_bt601_unit_float32_mul255_{qmode}",
                lambda frame, order_fn=order_fn, qmode=qmode: quantize_unit(
                    torch_image_area_resize(gray_bt601_unit_float32(order_fn(frame))), qmode
                ),
            )

        for gray_name, gray_fn in (("bt601_full_float", gray_bt601_float), ("bt601_video_float", gray_bt601_video_float)):
            for resample in ("bilinear", "box"):
                add_variant(
                    variants,
                    f"pil_{channel_order}_{gray_name}_{resample}_uint8_input",
                    lambda frame, order_fn=order_fn, gray_fn=gray_fn, resample=resample: pil_resize(gray_fn(order_fn(frame)), resample),
                )

        for gray_name, gray_fn in (("bt601_full_float", gray_bt601_float), ("bt601_video_float", gray_bt601_video_float)):
            for anti_aliasing in (False, True):
                for qmode in ("trunc", "round_half_up"):
                    add_variant(
                        variants,
                        f"skimage_{channel_order}_{gray_name}_linear_aa{int(anti_aliasing)}_{qmode}",
                        lambda frame, order_fn=order_fn, gray_fn=gray_fn, anti_aliasing=anti_aliasing, qmode=qmode: skimage_resize(
                            gray_fn(order_fn(frame)), anti_aliasing, qmode
                        ),
                    )

    return variants


def bbox(diff: Array) -> str:
    coords = np.argwhere(diff)
    if coords.size == 0:
        return "none"
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return f"y={int(y_min)}..{int(y_max)},x={int(x_min)}..{int(x_max)}"


def diff_summary(observed: Array, expected: Array) -> str:
    observed = np.ascontiguousarray(observed)
    expected = np.ascontiguousarray(expected)
    if observed.shape != expected.shape:
        return f"shape_mismatch observed={observed.shape} expected={expected.shape}"
    abs_diff = np.abs(observed.astype(np.int16) - expected.astype(np.int16))
    equal = abs_diff == 0
    percent_equal = 100.0 * float(equal.sum()) / float(expected.size) if expected.size else 100.0
    mean_abs = float(abs_diff.mean()) if abs_diff.size else 0.0
    max_abs = int(abs_diff.max()) if abs_diff.size else 0
    return (
        f"exact={bool(max_abs == 0)} percent_equal={percent_equal:.6f} "
        f"mean_abs_diff={mean_abs:.9f} max_abs_diff={max_abs} bbox={bbox(~equal)}"
    )


def compare_variant(variant: Variant, frames: list[tuple[int, Array, Array]]) -> dict[str, Any]:
    exact_count = 0
    total_pixels = 0
    equal_pixels = 0
    abs_sum = 0.0
    max_abs = 0
    first_mismatch_frame: int | None = None
    first_bbox = "none"
    first_error: str | None = None

    for step, frame, expected in frames:
        try:
            observed = np.ascontiguousarray(variant.fn(frame))
        except Exception as exc:
            first_error = f"{type(exc).__name__}: {exc}"
            break

        if observed.shape != expected.shape:
            first_error = f"shape mismatch at step {step}: {observed.shape} != {expected.shape}"
            break

        observed = quantize(observed, "trunc") if observed.dtype != np.uint8 else observed
        diff = observed.astype(np.int16) - expected.astype(np.int16)
        abs_diff = np.abs(diff)
        is_equal = abs_diff == 0
        total_pixels += int(expected.size)
        equal_pixels += int(is_equal.sum())
        abs_sum += float(abs_diff.sum())
        frame_max = int(abs_diff.max()) if abs_diff.size else 0
        max_abs = max(max_abs, frame_max)
        if frame_max == 0:
            exact_count += 1
        elif first_mismatch_frame is None:
            first_mismatch_frame = step
            first_bbox = bbox(~is_equal)

    percent_equal = 100.0 * equal_pixels / total_pixels if total_pixels else 0.0
    mean_abs_diff = abs_sum / total_pixels if total_pixels else math.inf
    return {
        "variant": variant.name,
        "exact_match_count": exact_count,
        "frame_count": len(frames),
        "percent_equal": percent_equal,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs,
        "first_mismatch_frame": first_mismatch_frame,
        "bbox": first_bbox,
        "error": first_error,
    }


def load_pairs(tape_dir: Path, deepmind_path: Path, max_frames: int) -> list[tuple[int, Array, Array]]:
    transitions = {
        int(row["step"]): row
        for row in read_jsonl(tape_dir / "transitions.jsonl")
        if row.get("phase") == "transition"
    }
    deepmind_rows = [row for row in read_jsonl(deepmind_path) if row.get("phase") == "preprocess"]
    pairs: list[tuple[int, Array, Array]] = []
    for row in deepmind_rows:
        step = int(row["step"])
        if step not in transitions:
            continue
        frame = load_frame(transitions[step]["pooled_path"])
        expected = np.ascontiguousarray(np.load(row["processed_path"]).astype(np.uint8))
        pairs.append((step, frame, expected))
        if max_frames and len(pairs) >= max_frames:
            break
    return pairs


def check_intermediates(frames: list[tuple[int, Array, Array]], intermediate_dir: Path) -> list[str]:
    pairs = {step: (frame, expected) for step, frame, expected in frames}
    rows: list[str] = []
    for raw_path in sorted(intermediate_dir.glob("raw_hwc_*.npy")):
        step = int(raw_path.stem.rsplit("_", 1)[1])
        if step not in pairs:
            continue
        canonical, expected_processed = pairs[step]
        dumped = load_frame(raw_path)
        rows.append(
            f"step {step}: raw_hwc_matches_canonical={bool(np.array_equal(dumped, canonical))} "
            f"dumped_hash={arr_hash(dumped)} canonical_hash={arr_hash(canonical)}"
        )

        luminance_path = intermediate_dir / f"luminance_{step:06d}.npy"
        if luminance_path.exists():
            luminance = np.ascontiguousarray(np.load(luminance_path).astype(np.uint8))
            bt601_trunc = quantize(gray_bt601_float(canonical), "trunc")
            bt601_round = quantize(gray_bt601_float(canonical), "round_half_up")
            rows.append(f"step {step}: luminance_vs_bt601_trunc {diff_summary(bt601_trunc, luminance)}")
            rows.append(f"step {step}: luminance_vs_bt601_round_half_up {diff_summary(bt601_round, luminance)}")

        resized_path = intermediate_dir / f"resized_{step:06d}.npy"
        final_path = intermediate_dir / f"final_network_input_{step:06d}.npy"
        if resized_path.exists() and final_path.exists():
            resized = np.ascontiguousarray(np.load(resized_path).astype(np.uint8))
            final = np.ascontiguousarray(np.load(final_path).astype(np.uint8))
            rows.append(f"step {step}: resized_vs_final {diff_summary(resized, final)}")
            rows.append(f"step {step}: final_vs_deepmind_processed_jsonl {diff_summary(final, expected_processed)}")
    return rows


def format_table(results: list[dict[str, Any]]) -> str:
    headers = [
        "rank",
        "variant",
        "exact",
        "percent_equal",
        "mean_abs_diff",
        "max_abs_diff",
        "first_mismatch",
        "bbox",
        "error",
    ]
    rows = []
    for rank, result in enumerate(results, start=1):
        exact = f"{result['exact_match_count']}/{result['frame_count']}"
        rows.append(
            [
                str(rank),
                result["variant"],
                exact,
                f"{result['percent_equal']:.6f}",
                f"{result['mean_abs_diff']:.9f}",
                str(result["max_abs_diff"]),
                "none" if result["first_mismatch_frame"] is None else str(result["first_mismatch_frame"]),
                result["bbox"],
                result["error"] or "",
            ]
        )
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    lines = ["  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    return "\n".join(lines)


def write_matching_implementation(path: Path, best: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'''#!/usr/bin/env python
"""DeepMind-compatible preprocessing selected by Stage 2b.

Selected variant: {best["variant"]}
This file is generated only when the frozen-frame audit finds an exact match
against DeepMind's net_downsample_2x_full_y output.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pytorch.preprocess_variants import build_variants  # noqa: E402


VARIANT_NAME = {best["variant"]!r}


def preprocess_frame(obs: np.ndarray) -> np.ndarray:
    for variant in build_variants():
        if variant.name == VARIANT_NAME:
            return variant.fn(obs)
    raise RuntimeError(f"Stage 2b selected variant not found: {{VARIANT_NAME}}")
''',
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    tape_dir = Path(args.tape_dir)
    deepmind_path = Path(args.deepmind)
    intermediate_dir = Path(args.intermediate_dir)
    pairs = load_pairs(tape_dir, deepmind_path, args.max_frames)
    if not pairs:
        raise SystemExit("no frozen frame pairs found")

    results = [compare_variant(variant, pairs) for variant in build_variants()]
    results.sort(
        key=lambda row: (
            row["error"] is not None,
            -int(row["exact_match_count"]),
            float(row["mean_abs_diff"]),
            int(row["max_abs_diff"]),
            -float(row["percent_equal"]),
            row["variant"],
        )
    )

    raw_checks = check_intermediates(pairs, intermediate_dir) if intermediate_dir.exists() else []
    table = format_table(results)
    best = results[0]
    match_line = "MATCH" if best["exact_match_count"] == len(pairs) and best["error"] is None else "NO EXACT MATCH"

    output_lines = [
        "Stage 2b preprocessing variant ranking",
        "",
        f"frames_compared: {len(pairs)}",
        f"best_variant: {best['variant']}",
        f"status: {match_line}",
        "",
        "DeepMind intermediate sanity check:",
    ]
    output_lines.extend(raw_checks or ["no raw intermediate dumps found"])
    output_lines.extend(["", table, ""])
    text = "\n".join(output_lines)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    jsonl_path = Path(args.jsonl)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")

    if match_line == "MATCH" and args.write_matching_implementation:
        write_matching_implementation(Path(args.write_matching_implementation), best)
        print(f"wrote matching implementation: {args.write_matching_implementation}")

    print(text)
    return 0 if match_line == "MATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
