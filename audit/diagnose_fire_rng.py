#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs/stage1f")
    parser = argparse.ArgumentParser(description="Diagnose Breakout FIRE RNG/start-transition alignment.")
    parser.add_argument("--out", default=out)
    parser.add_argument("--seed-max", type=int, default=50)
    parser.add_argument("--delays", default="0,1,2,3,4,5,10,20,30")
    return parser.parse_args()


def load_frame(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.load(path)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    return np.ascontiguousarray(arr.astype(np.uint8, copy=False))


def frame_path(run_dir: Path, side: str, step: int, repeat_i: int = 0) -> Path:
    return run_dir / "frames" / f"{side}_step_{step:03d}_repeat_{repeat_i:03d}.npy"


def sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def processed84(frame: np.ndarray) -> np.ndarray:
    arr = frame
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        rgb = arr[..., :3].astype("float64")
        lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    elif arr.ndim == 3:
        lum = arr[..., 0].astype("float64")
    else:
        lum = arr.astype("float64")
    lum = np.rint(np.clip(lum, 0, 255)).astype(np.uint8)
    try:
        from PIL import Image

        return np.asarray(Image.fromarray(lum).resize((84, 84), Image.Resampling.BILINEAR), dtype=np.uint8)
    except Exception:
        y_idx = np.linspace(0, lum.shape[0] - 1, 84).round().astype(int)
        x_idx = np.linspace(0, lum.shape[1] - 1, 84).round().astype(int)
        return lum[np.ix_(y_idx, x_idx)]


def diff_stats(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"shape_mismatch": [list(left.shape), list(right.shape)]}
    abs_diff = np.abs(left.astype("int16") - right.astype("int16"))
    if left.ndim == 3:
        pixel_equal = np.all(left == right, axis=-1)
    else:
        pixel_equal = left == right
    coords = np.argwhere(~pixel_equal)
    bbox = None
    if coords.size:
        y_min, x_min = coords.min(axis=0).tolist()
        y_max, x_max = coords.max(axis=0).tolist()
        bbox = {"y_min": y_min, "y_max": y_max, "x_min": x_min, "x_max": x_max}
    return {
        "exact_equal": bool(np.array_equal(left, right)),
        "mean_abs_diff": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "max_abs_diff": int(abs_diff.max()) if abs_diff.size else 0,
        "percent_equal": float(pixel_equal.mean() * 100.0) if pixel_equal.size else 100.0,
        "differing_pixels": int(coords.shape[0]),
        "bbox": bbox,
    }


def classify_region(bbox: dict[str, int] | None, shape: tuple[int, ...]) -> str:
    if not bbox:
        return "none"
    y_mid = (bbox["y_min"] + bbox["y_max"]) / 2.0
    x_min = bbox["x_min"]
    x_max = bbox["x_max"]
    h, w = shape[0], shape[1]
    if y_mid < 34:
        return "scoreboard/top"
    if x_min < 8 or x_max >= max(w - 8, 0):
        return "border"
    if y_mid >= h * 0.86:
        return "paddle/lower"
    return "playfield"


def load_trace(run_dir: Path, side: str) -> list[dict[str, Any]]:
    path = run_dir / f"{side}_env.jsonl"
    if not path.exists():
        return []
    return read_jsonl(path)


def agent_rows(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["step"]): row for row in rows if row.get("phase") == "agent_step"}


def first_post_fire(run_dir: Path, side: str, fire_step: int = 0) -> np.ndarray | None:
    return load_frame(frame_path(run_dir, side, fire_step, 0))


def write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def seed_grid(base: Path, seed_max: int) -> tuple[list[str], dict[str, Any]]:
    p_frames: dict[int, np.ndarray] = {}
    d_frames: dict[int, np.ndarray] = {}
    for seed in range(seed_max + 1):
        run = base / "seed_grid" / f"seed_{seed:03d}"
        p = first_post_fire(run, "pytorch")
        d = first_post_fire(run, "deepmind")
        if p is not None:
            p_frames[seed] = p
        if d is not None:
            d_frames[seed] = d

    exact_pairs: list[tuple[int, int]] = []
    best: tuple[float, int, int, dict[str, Any]] | None = None
    d1_matches: list[int] = []
    means: list[float] = []
    for p_seed, p_frame in p_frames.items():
        for d_seed, d_frame in d_frames.items():
            stats = diff_stats(p_frame, d_frame)
            mean = float(stats["mean_abs_diff"])
            means.append(mean)
            if stats["exact_equal"]:
                exact_pairs.append((p_seed, d_seed))
                if d_seed == 1:
                    d1_matches.append(p_seed)
            if best is None or mean < best[0]:
                best = (mean, p_seed, d_seed, stats)

    lines = [
        "Stage 1f seed grid",
        f"pytorch_seed_count: {len(p_frames)}",
        f"deepmind_seed_count: {len(d_frames)}",
        f"exact_seed_pairs: {exact_pairs if exact_pairs else 'none'}",
        f"deepmind_seed_1_matches_pytorch_seed: {d1_matches if d1_matches else 'none'}",
    ]
    if means:
        lines.extend(
            [
                f"mean_abs_diff_min: {min(means)}",
                f"mean_abs_diff_median: {float(np.median(np.asarray(means)))}",
                f"mean_abs_diff_max: {max(means)}",
            ]
        )
    if best is not None:
        lines.extend(
            [
                f"best_near_match_seed_pair: pytorch={best[1]} deepmind={best[2]}",
                f"best_near_match_stats: {best[3]}",
            ]
        )
    return lines, {"exact_pairs": exact_pairs, "best": best, "d1_matches": d1_matches}


def delayed_fire(base: Path, delays: list[int]) -> tuple[list[str], dict[str, Any]]:
    lines = ["Stage 1f delayed FIRE"]
    exact_delays: list[int] = []
    mismatches: list[dict[str, Any]] = []
    for delay in delays:
        run = base / "delayed_fire" / f"delay_{delay:03d}"
        p = first_post_fire(run, "pytorch", delay)
        d = first_post_fire(run, "deepmind", delay)
        if p is None or d is None:
            lines.append(f"delay={delay}: missing frame")
            continue
        stats = diff_stats(p, d)
        region = classify_region(stats.get("bbox"), p.shape)
        if stats["exact_equal"]:
            exact_delays.append(delay)
        else:
            mismatches.append({"delay": delay, "bbox": stats.get("bbox"), "region": region})
        lines.append(f"delay={delay}: {stats}, region={region}")
    lines.extend(
        [
            f"mismatch_happens_for_every_delay: {str(len(exact_delays) == 0).lower()}",
            f"exact_delay_values: {exact_delays if exact_delays else 'none'}",
            f"mismatch_bboxes: {mismatches if mismatches else 'none'}",
        ]
    )
    return lines, {"exact_delays": exact_delays}


def repeat_boundary(base: Path) -> tuple[list[str], dict[str, Any]]:
    variants = ["repeat_logic", "one_step_fire", "fire_then_noop", "fire4"]
    lines = ["Stage 1f FIRE repeat boundary"]
    result: dict[str, Any] = {}
    for variant in variants:
        run = base / "repeat_boundary" / variant
        p_rows = agent_rows(load_trace(run, "pytorch"))
        d_rows = agent_rows(load_trace(run, "deepmind"))
        first_mismatch = None
        sequence_lines: list[str] = []
        for step in sorted(set(p_rows) & set(d_rows)):
            repeat_count = min(len(p_rows[step].get("repeats") or []), len(d_rows[step].get("repeats") or []))
            for repeat_i in range(repeat_count):
                p_frame = load_frame(frame_path(run, "pytorch", step, repeat_i))
                d_frame = load_frame(frame_path(run, "deepmind", step, repeat_i))
                if p_frame is None or d_frame is None:
                    continue
                stats = diff_stats(p_frame, d_frame)
                sequence_lines.append(
                    f"variant={variant} step={step} repeat_i={repeat_i} exact={stats['exact_equal']} mean_abs_diff={stats['mean_abs_diff']} bbox={stats['bbox']}"
                )
                if first_mismatch is None and not stats["exact_equal"]:
                    first_mismatch = (step, repeat_i, stats)
        lines.append(f"variant={variant}")
        if sequence_lines:
            lines.extend(sequence_lines)
        else:
            lines.append("missing repeated raw frame sequence")
        lines.append(f"first_mismatch: {first_mismatch if first_mismatch else 'none'}")
        result[variant] = first_mismatch
    return lines, result


def determinism(base: Path) -> tuple[list[str], dict[str, Any]]:
    lines = ["Stage 1f same-side determinism"]
    result: dict[str, Any] = {}
    for side in ("pytorch", "deepmind"):
        a = first_post_fire(base / "determinism" / "run_a", side)
        b = first_post_fire(base / "determinism" / "run_b", side)
        if a is None or b is None:
            lines.append(f"{side}_same_seed_deterministic: unavailable")
            result[side] = None
            continue
        stats = diff_stats(a, b)
        lines.append(f"{side}_same_seed_deterministic: {str(stats['exact_equal']).lower()}")
        lines.append(f"{side}_same_seed_stats: {stats}")
        result[side] = stats["exact_equal"]
    return lines, result


def dqn_relevance(base: Path) -> list[str]:
    run = base / "repeat_boundary" / "repeat_logic"
    p = first_post_fire(run, "pytorch", 0)
    d = first_post_fire(run, "deepmind", 0)
    lines = ["Stage 1f DQN relevance summary"]
    if p is None or d is None:
        return lines + ["missing first post-FIRE frame"]
    raw = diff_stats(p, d)
    p84 = processed84(p)
    d84 = processed84(d)
    proc = diff_stats(p84, d84)
    lines.extend(
        [
            f"raw_exact_equal: {str(raw['exact_equal']).lower()}",
            f"raw_percent_equal: {raw['percent_equal']}",
            f"raw_mean_abs_diff: {raw['mean_abs_diff']}",
            f"raw_max_abs_diff: {raw['max_abs_diff']}",
            f"raw_bbox: {raw['bbox']}",
            f"raw_region: {classify_region(raw.get('bbox'), p.shape)}",
            f"processed_84x84_exact_equal: {str(proc['exact_equal']).lower()}",
            f"processed_84x84_percent_equal: {proc['percent_equal']}",
            f"processed_84x84_mean_abs_diff: {proc['mean_abs_diff']}",
            f"processed_84x84_max_abs_diff: {proc['max_abs_diff']}",
            f"pytorch_processed_hash: {sha(p84)}",
            f"deepmind_processed_hash: {sha(d84)}",
        ]
    )
    return lines


def main() -> int:
    args = parse_args()
    base = Path(args.out)
    delays = [int(item) for item in args.delays.split(",") if item.strip()]

    seed_lines, seed_result = seed_grid(base, args.seed_max)
    delayed_lines, delayed_result = delayed_fire(base, delays)
    repeat_lines, repeat_result = repeat_boundary(base)
    determinism_lines, determinism_result = determinism(base)
    dqn_lines = dqn_relevance(base)

    write(base / "seed_grid.txt", seed_lines)
    write(base / "delayed_fire.txt", delayed_lines)
    write(base / "repeat_boundary.txt", repeat_lines)
    write(base / "determinism.txt", determinism_lines)

    interp: list[str] = ["Stage 1f summary", ""]
    interp.extend(seed_lines[:])
    interp.append("")
    interp.extend(delayed_lines[:])
    interp.append("")
    interp.extend(repeat_lines[:])
    interp.append("")
    interp.extend(determinism_lines[:])
    interp.append("")
    interp.extend(dqn_lines[:])
    interp.append("")
    interp.append("Interpretation")
    if seed_result["d1_matches"]:
        interp.append("PyTorch seed k matches DeepMind seed 1: RNG seeding convention mismatch is alignable.")
    elif seed_result["exact_pairs"]:
        interp.append("Some cross-seed pairs match, but not DeepMind seed 1; RNG state may be alignable with a nontrivial mapping.")
    elif determinism_result.get("pytorch") is False or determinism_result.get("deepmind") is False:
        interp.append("Same-side repeated runs are not deterministic; seeding is broken or incomplete.")
    elif delayed_result["exact_delays"]:
        interp.append("A pre-FIRE delay fixes the mismatch; reset/start timing is implicated.")
    elif repeat_result.get("one_step_fire") is not None:
        interp.append("One raw ALE FIRE act already mismatches; start/FIRE transition semantics differ across wrappers/backends.")
    elif repeat_result.get("repeat_logic") is not None:
        interp.append("One-step FIRE matches but repeated FIRE mismatches; external repeat-loop behavior is implicated.")
    else:
        interp.append("No byte-exact seed/delay alignment found; wrapper/backend FIRE transition likely cannot be byte-aligned.")

    write(base / "summary.txt", interp)
    print("\n".join(interp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
