#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from common import read_jsonl


FRAME_RE = re.compile(r"^(pytorch|deepmind)_(init|step_(\d{3})_(pre|repeat_(\d{3})|pool_src_(\d{3})|pooled)|step_(\d{3}))$")


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs")
    parser = argparse.ArgumentParser(description="Diagnose temporal frame alignment between Stage 1 traces.")
    parser.add_argument("--frames-dir", default=os.path.join(out, "frames"))
    parser.add_argument("--pytorch-jsonl", default=os.path.join(out, "pytorch_env.jsonl"))
    parser.add_argument("--deepmind-jsonl", default=os.path.join(out, "deepmind_env.jsonl"))
    parser.add_argument("--rom", default=os.getenv("ROM"))
    parser.add_argument("--max-agent-steps", type=int, default=20)
    parser.add_argument("--max-raw-frames", type=int, default=100)
    parser.add_argument("--out", default=os.getenv("TEMPORAL_ALIGNMENT_OUT"))
    return parser.parse_args()


def load_frame(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        numeric = arr.astype("float64", copy=False)
        if numeric.size and numeric.min() >= 0 and numeric.max() <= 1:
            numeric = numeric * 255
        arr = np.rint(np.clip(numeric, 0, 255)).astype(np.uint8)
    return np.ascontiguousarray(arr)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def diff_stats(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "shape_mismatch": f"{list(left.shape)} vs {list(right.shape)}",
            "mean_abs_diff": float("inf"),
            "max_abs_diff": float("inf"),
            "percent_equal": 0.0,
        }
    diff = np.abs(left.astype("int16") - right.astype("int16"))
    return {
        "shape_mismatch": None,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "max_abs_diff": int(diff.max()) if diff.size else 0,
        "percent_equal": float((left == right).mean() * 100.0) if diff.size else 100.0,
    }


def parse_label(path: Path) -> dict[str, Any] | None:
    match = FRAME_RE.match(path.stem)
    if not match:
        return None
    side = match.group(1)
    if match.group(2) == "init":
        return {"side": side, "kind": "init", "step": -1, "repeat": -1, "label": path.stem}
    if match.group(7) is not None:
        return {"side": side, "kind": "final", "step": int(match.group(7)), "repeat": 999, "label": path.stem}
    step = int(match.group(3))
    suffix = match.group(4)
    if suffix == "pre":
        return {"side": side, "kind": "pre", "step": step, "repeat": -1, "label": path.stem}
    if suffix == "pooled":
        return {"side": side, "kind": "pooled", "step": step, "repeat": 1000, "label": path.stem}
    if suffix.startswith("repeat_"):
        return {"side": side, "kind": "repeat", "step": step, "repeat": int(match.group(5)), "label": path.stem}
    if suffix.startswith("pool_src_"):
        return {"side": side, "kind": "pool_src", "step": step, "repeat": int(match.group(6)), "label": path.stem}
    return None


def load_records(frames_dir: Path, side: str, max_agent_steps: int) -> list[dict[str, Any]]:
    records = []
    for path in sorted(frames_dir.glob(f"{side}_*.npy")):
        meta = parse_label(path)
        if meta is None or meta["side"] != side:
            continue
        if meta["step"] > max_agent_steps:
            continue
        arr = load_frame(path)
        meta["path"] = path
        meta["array"] = arr
        meta["hash"] = frame_hash(arr)
        records.append(meta)
    return sorted(records, key=lambda r: (r["step"], r["repeat"], r["kind"], r["label"]))


def first_exact_match(left: list[dict[str, Any]], right: list[dict[str, Any]], include_init: bool = False) -> tuple[str, str] | None:
    for lrec in left:
        if not include_init and lrec["kind"] == "init":
            continue
        for rrec in right:
            if not include_init and rrec["kind"] == "init":
                continue
            if lrec["hash"] == rrec["hash"] and lrec["array"].shape == rrec["array"].shape:
                return lrec["label"], rrec["label"]
    return None


def best_near_match(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    best = None
    for lrec in left:
        for rrec in right:
            stats = diff_stats(lrec["array"], rrec["array"])
            candidate = (stats["mean_abs_diff"], stats["max_abs_diff"], lrec, rrec, stats)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
    assert best is not None
    return best[2], best[3], best[4]


def find_label(records: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    for rec in records:
        if rec["label"] == label:
            return rec
    return None


def match_target(target: dict[str, Any] | None, candidates: list[dict[str, Any]]) -> tuple[str, dict[str, Any] | None]:
    if target is None:
        return "missing target", None
    for candidate in candidates:
        if target["hash"] == candidate["hash"] and target["array"].shape == candidate["array"].shape:
            return candidate["label"], None
    best_l, best_r, stats = best_near_match([target], candidates)
    del best_l
    return "none", {"candidate": best_r["label"], **stats}


def package_versions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if row.get("phase") == "init":
            return {
                "gymnasium_version": row.get("gymnasium_version"),
                "ale_py_version": row.get("ale_py_version"),
                "alewrap_version": row.get("alewrap_version"),
                "ale_version": row.get("ale_version"),
            }
    return {}


def main() -> int:
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    p_records = load_records(frames_dir, "pytorch", args.max_agent_steps)
    d_records = load_records(frames_dir, "deepmind", args.max_agent_steps)
    if not p_records or not d_records:
        raise SystemExit(f"missing frame records in {frames_dir}")

    p_raw = [r for r in p_records if r["kind"] == "repeat"][: args.max_raw_frames]
    d_raw = [r for r in d_records if r["kind"] == "repeat"][: args.max_raw_frames]
    p0r0 = find_label(p_records, "pytorch_step_000_repeat_000")
    d0r0 = find_label(d_records, "deepmind_step_000_repeat_000")
    p0_in_d, p0_best = match_target(p0r0, d_raw)
    d0_in_p, d0_best = match_target(d0r0, p_raw)
    exact_after_init = first_exact_match(p_records, d_records)
    best_l, best_r, best_stats = best_near_match(p_records, d_records)

    p_versions = package_versions(read_jsonl(args.pytorch_jsonl))
    d_versions = package_versions(read_jsonl(args.deepmind_jsonl))

    lines = [
        "Temporal/frame-boundary diagnosis",
        "",
        f"pytorch_frame_count: {len(p_records)}",
        f"deepmind_frame_count: {len(d_records)}",
        f"pytorch_gymnasium_version: {p_versions.get('gymnasium_version')}",
        f"pytorch_ale_py_version: {p_versions.get('ale_py_version')}",
        f"deepmind_alewrap_version: {d_versions.get('alewrap_version')}",
        f"deepmind_ale_version: {d_versions.get('ale_version')}",
    ]

    if args.rom:
        rom_path = Path(args.rom)
        if rom_path.exists():
            lines.extend(
                [
                    f"rom_path: {rom_path}",
                    f"rom_md5: {md5_file(rom_path)}",
                    f"rom_sha256: {sha256_file(rom_path)}",
                ]
            )
        else:
            lines.append(f"rom_path_missing: {rom_path}")

    lines.extend(
        [
            "",
            f"first_exact_frame_match_after_init: {exact_after_init if exact_after_init else 'none'}",
            f"best_near_match: {best_l['label']} vs {best_r['label']}",
            f"best_mean_abs_diff: {best_stats['mean_abs_diff']}",
            f"best_max_abs_diff: {best_stats['max_abs_diff']}",
            f"best_percent_equal: {best_stats['percent_equal']}",
            "",
            f"pytorch_step0_repeat0_equals_deepmind_first_100_raw: {p0_in_d}",
        ]
    )
    if p0_best:
        lines.extend(
            [
                f"pytorch_step0_repeat0_best_deepmind_candidate: {p0_best['candidate']}",
                f"pytorch_step0_repeat0_best_mean_abs_diff: {p0_best['mean_abs_diff']}",
                f"pytorch_step0_repeat0_best_max_abs_diff: {p0_best['max_abs_diff']}",
                f"pytorch_step0_repeat0_best_percent_equal: {p0_best['percent_equal']}",
            ]
        )
    lines.append(f"deepmind_step0_repeat0_equals_pytorch_first_100_raw: {d0_in_p}")
    if d0_best:
        lines.extend(
            [
                f"deepmind_step0_repeat0_best_pytorch_candidate: {d0_best['candidate']}",
                f"deepmind_step0_repeat0_best_mean_abs_diff: {d0_best['mean_abs_diff']}",
                f"deepmind_step0_repeat0_best_max_abs_diff: {d0_best['max_abs_diff']}",
                f"deepmind_step0_repeat0_best_percent_equal: {d0_best['percent_equal']}",
            ]
        )

    lines.extend(["", "Conclusion:"])
    if p0_in_d not in ("none", "missing target"):
        lines.append(f"PyTorch step0 repeat0 aligns with {p0_in_d}; frame-boundary mismatch")
    elif d0_in_p not in ("none", "missing target"):
        lines.append(f"DeepMind step0 repeat0 aligns with {d0_in_p}; frame-boundary mismatch")
    elif exact_after_init:
        lines.append("exact frames reappear later, so mismatch is temporal rather than global")
    elif best_stats["mean_abs_diff"] <= 1.0:
        lines.append("no exact temporal match, but best diff is tiny; suspect read-buffer/color conversion")
    else:
        lines.append("no exact temporal match in the searched window and diffs are not tiny; suspect ALE/backend/state progression mismatch")

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
