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
    out = os.getenv("OUT", "audit_outputs")
    parser = argparse.ArgumentParser(description="Diagnose Stage 1e emulator state vs screen mismatches.")
    parser.add_argument("--out", default=out)
    parser.add_argument("--pytorch", default=os.path.join(out, "pytorch_env.jsonl"))
    parser.add_argument("--deepmind", default=os.path.join(out, "deepmind_env.jsonl"))
    parser.add_argument("--frames-dir", default=os.path.join(out, "frames"))
    parser.add_argument("--report", default=os.path.join(out, "report.txt"))
    parser.add_argument("--label", default=Path(out).name)
    parser.add_argument("--max-mismatches", type=int, default=12)
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


def frame_path(frames_dir: Path, side: str, step: int, repeat_i: int) -> Path:
    return frames_dir / f"{side}_step_{step:03d}_repeat_{repeat_i:03d}.npy"


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


def sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def pixel_diff(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"shape_mismatch": [list(left.shape), list(right.shape)]}
    diff = left.astype("int16") - right.astype("int16")
    abs_diff = np.abs(diff)
    if left.ndim == 3:
        pixel_equal = np.all(left == right, axis=-1)
        channel_diff = np.any(left != right, axis=(0, 1)).tolist()
        only_one_channel = sum(bool(item) for item in channel_diff) == 1
    else:
        pixel_equal = left == right
        channel_diff = []
        only_one_channel = False
    unequal = ~pixel_equal
    coords = np.argwhere(unequal)
    bbox = None
    first_coords: list[dict[str, Any]] = []
    zones: dict[str, int] = {}
    if coords.size:
        y_min, x_min = coords.min(axis=0).tolist()
        y_max, x_max = coords.max(axis=0).tolist()
        bbox = {"y_min": y_min, "y_max": y_max, "x_min": x_min, "x_max": x_max}
        for y, x in coords[:20]:
            first_coords.append(
                {
                    "y": int(y),
                    "x": int(x),
                    "pytorch_rgb": left[int(y), int(x)].tolist() if left.ndim == 3 else int(left[int(y), int(x)]),
                    "deepmind_rgb": right[int(y), int(x)].tolist() if right.ndim == 3 else int(right[int(y), int(x)]),
                }
            )
        for y, x in coords:
            zone = classify_region(int(y), int(x), left.shape)
            zones[zone] = zones.get(zone, 0) + 1
    total_pixels = int(pixel_equal.size)
    return {
        "number_differing_pixels": int(coords.shape[0]),
        "percent_pixels_equal": float(pixel_equal.mean() * 100.0) if total_pixels else 100.0,
        "mean_abs_diff": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "max_abs_diff": int(abs_diff.max()) if abs_diff.size else 0,
        "bbox": bbox,
        "first_20_differing_coordinates": first_coords,
        "channel_has_mismatch": channel_diff,
        "mismatch_only_in_one_channel": only_one_channel,
        "region_counts": zones,
    }


def classify_region(y: int, x: int, shape: tuple[int, ...]) -> str:
    h = shape[0]
    w = shape[1]
    if y < 34:
        return "scoreboard/top"
    if x < 8 or x >= max(w - 8, 0):
        return "border"
    if y >= int(h * 0.86):
        return "paddle/lower"
    return "playfield"


def first_phase(rows: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    for row in rows:
        if row.get("phase") == phase:
            return row
    return {}


def agent_rows(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result = {}
    for row in rows:
        if row.get("phase") == "agent_step":
            result[int(row["step"])] = row
    return result


def same_list(left: Any, right: Any) -> bool:
    return list(left or []) == list(right or [])


def repeat_value(row: dict[str, Any], repeat_i: int, key: str) -> Any:
    repeats = row.get("repeats") or []
    if repeat_i < 0 or repeat_i >= len(repeats):
        return None
    return repeats[repeat_i].get(key)


def nested_hash(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("hash")
    return None


def ram_presence(value: Any) -> str:
    if isinstance(value, dict) and value.get("hash") is not None:
        return "available"
    return "unavailable"


def main() -> int:
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    p_rows = read_jsonl(args.pytorch)
    d_rows = read_jsonl(args.deepmind)
    p_steps = agent_rows(p_rows)
    d_steps = agent_rows(d_rows)
    common_steps = sorted(set(p_steps) & set(d_steps))

    action_code_match = True
    reward_lives_terminal_match = True
    first_ram_mismatch: tuple[int, int] | None = None
    first_ram_unavailable: tuple[int, int, str, str] | None = None
    first_raw_mismatch: tuple[int, int] | None = None
    processed_mismatch_for_first_raw: dict[str, Any] | None = None
    raw_mismatch_details: list[tuple[int, int, dict[str, Any]]] = []

    for step in common_steps:
        p_row = p_steps[step]
        d_row = d_steps[step]
        if p_row.get("ale_action_code_used") != d_row.get("ale_action_code_used"):
            action_code_match = False
        repeat_count = min(len(p_row.get("repeats") or []), len(d_row.get("repeats") or []))
        for repeat_i in range(repeat_count):
            p_rep = p_row["repeats"][repeat_i]
            d_rep = d_row["repeats"][repeat_i]
            if (
                p_rep.get("reward") != d_rep.get("reward")
                or p_rep.get("lives_after") != d_rep.get("lives_after")
                or p_rep.get("terminal") != d_rep.get("terminal")
            ):
                reward_lives_terminal_match = False
            p_ram_hash = nested_hash(p_rep.get("ram"))
            d_ram_hash = nested_hash(d_rep.get("ram"))
            p_ram_presence = ram_presence(p_rep.get("ram"))
            d_ram_presence = ram_presence(d_rep.get("ram"))
            if first_ram_unavailable is None and p_ram_presence != d_ram_presence:
                first_ram_unavailable = (step, repeat_i, p_ram_presence, d_ram_presence)
            if first_ram_mismatch is None and p_ram_hash is not None and d_ram_hash is not None and p_ram_hash != d_ram_hash:
                first_ram_mismatch = (step, repeat_i)
            p_raw_hash = nested_hash(p_rep.get("raw_frame"))
            d_raw_hash = nested_hash(d_rep.get("raw_frame"))
            if p_raw_hash != d_raw_hash:
                if first_raw_mismatch is None:
                    first_raw_mismatch = (step, repeat_i)
                if len(raw_mismatch_details) < args.max_mismatches:
                    p_frame = load_frame(frame_path(frames_dir, "pytorch", step, repeat_i))
                    d_frame = load_frame(frame_path(frames_dir, "deepmind", step, repeat_i))
                    if p_frame is not None and d_frame is not None:
                        detail = pixel_diff(p_frame, d_frame)
                        raw_mismatch_details.append((step, repeat_i, detail))
                        if processed_mismatch_for_first_raw is None:
                            p_proc = processed84(p_frame)
                            d_proc = processed84(d_frame)
                            proc_diff = pixel_diff(p_proc, d_proc)
                            proc_diff["pytorch_processed_hash"] = sha(p_proc)
                            proc_diff["deepmind_processed_hash"] = sha(d_proc)
                            proc_diff["exact_processed_equal"] = bool(np.array_equal(p_proc, d_proc))
                            processed_mismatch_for_first_raw = proc_diff

    init = first_phase(p_rows, "init")
    d_init = first_phase(d_rows, "init")
    first_step0 = p_steps.get(0, {})
    noop_only_diverges = first_raw_mismatch is not None
    pytorch_ram_available = any(nested_hash(rep.get("ram")) is not None for row in p_steps.values() for rep in row.get("repeats", []))
    deepmind_ram_available = any(nested_hash(rep.get("ram")) is not None for row in d_steps.values() for rep in row.get("repeats", []))
    ram_available = pytorch_ram_available and deepmind_ram_available
    ram_after_first = "unavailable"
    if first_ram_unavailable is not None:
        ram_after_first = (
            f"unavailable on one side at step {first_ram_unavailable[0]} repeat {first_ram_unavailable[1]} "
            f"(pytorch={first_ram_unavailable[2]}, deepmind={first_ram_unavailable[3]})"
        )
    elif ram_available:
        ram_after_first = "match" if first_ram_mismatch is None else f"mismatch at step {first_ram_mismatch[0]} repeat {first_ram_mismatch[1]}"

    lines = [
        f"Stage 1e state-vs-screen diagnosis: {args.label}",
        "",
        "Versions",
        f"pytorch_gymnasium_version: {init.get('gymnasium_version')}",
        f"pytorch_ale_py_version: {init.get('ale_py_version')}",
        f"deepmind_alewrap_version: {d_init.get('alewrap_version')}",
        f"deepmind_ale_version: {d_init.get('ale_version')}",
        "",
        "Questions",
        f"A. same ALE action code every agent step: {str(action_code_match).lower()}",
        f"B. rewards/lives/terminal match at every raw repeat: {str(reward_lives_terminal_match).lower()}",
        f"C. RAM after first action: {ram_after_first}",
        f"D. RAM matches but screen differs: {str(ram_available and first_ram_mismatch is None and first_raw_mismatch is not None).lower()}",
        f"E. raw mismatch survives into 84x84 luminance input: {str(bool(processed_mismatch_for_first_raw and not processed_mismatch_for_first_raw['exact_processed_equal'])).lower()}",
        f"F. this tape diverges at raw screen level: {str(noop_only_diverges).lower()}",
        "",
        "First events",
        f"first_raw_frame_mismatch: {first_raw_mismatch if first_raw_mismatch is not None else 'none'}",
        f"first_ram_mismatch: {first_ram_mismatch if first_ram_mismatch is not None else 'none'}",
        f"ram_availability: pytorch={pytorch_ram_available}, deepmind={deepmind_ram_available}",
        f"step0_action_meaning: {first_step0.get('action_meaning_used')}",
        f"step0_ale_action_code: {first_step0.get('ale_action_code_used')}",
        "",
    ]

    if processed_mismatch_for_first_raw is not None:
        lines.extend(
            [
                "Canonical 84x84 luminance check for first raw mismatch",
                f"exact_processed_equal: {str(processed_mismatch_for_first_raw['exact_processed_equal']).lower()}",
                f"pytorch_processed_hash: {processed_mismatch_for_first_raw['pytorch_processed_hash']}",
                f"deepmind_processed_hash: {processed_mismatch_for_first_raw['deepmind_processed_hash']}",
                f"mean_abs_diff: {processed_mismatch_for_first_raw['mean_abs_diff']}",
                f"max_abs_diff: {processed_mismatch_for_first_raw['max_abs_diff']}",
                f"percent_pixels_equal: {processed_mismatch_for_first_raw['percent_pixels_equal']}",
                "",
            ]
        )

    if raw_mismatch_details:
        lines.append(f"Raw frame mismatch details, first {len(raw_mismatch_details)}")
        for step, repeat_i, detail in raw_mismatch_details:
            lines.extend(
                [
                    f"mismatch step={step} repeat_i={repeat_i}",
                    f"number_differing_pixels: {detail.get('number_differing_pixels')}",
                    f"percent_pixels_equal: {detail.get('percent_pixels_equal')}",
                    f"mean_abs_diff: {detail.get('mean_abs_diff')}",
                    f"max_abs_diff: {detail.get('max_abs_diff')}",
                    f"bbox: {detail.get('bbox')}",
                    f"first_20_differing_coordinates: {detail.get('first_20_differing_coordinates')}",
                    f"mismatch_only_in_one_channel: {str(detail.get('mismatch_only_in_one_channel')).lower()}",
                    f"channel_has_mismatch: {detail.get('channel_has_mismatch')}",
                    f"region_counts: {detail.get('region_counts')}",
                    "",
                ]
            )
    else:
        lines.append("No raw frame mismatches found.")
        lines.append("")

    if not ram_available:
        conclusion = "RAM unavailable; screen-vs-state cannot be decided from RAM on this backend."
    elif first_ram_mismatch is not None:
        conclusion = "RAM differs immediately or during the trace; actual emulator/backend state mismatch."
    elif first_raw_mismatch is not None and processed_mismatch_for_first_raw and processed_mismatch_for_first_raw["exact_processed_equal"]:
        conclusion = "RAM/reward/lives match and raw screen differs, but first mismatch disappears in canonical 84x84 luminance."
    elif first_raw_mismatch is not None:
        conclusion = "RAM/reward/lives match and raw screen differs; mismatch survives into canonical 84x84 luminance."
    else:
        conclusion = "No state or screen mismatch found on this deterministic tape."
    lines.extend(["Conclusion", conclusion, ""])

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
