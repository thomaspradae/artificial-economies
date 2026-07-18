#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs/stage2c_resize")
    parser = argparse.ArgumentParser(description="Diagnose Torch7 resize fixture mismatches.")
    parser.add_argument("--fixtures", default=os.path.join(out, "fixtures", "fixtures.jsonl"))
    parser.add_argument("--deepmind", default=os.path.join(out, "deepmind_outputs", "run1.jsonl"))
    parser.add_argument("--deepmind-repeat", default=os.path.join(out, "deepmind_outputs", "run2.jsonl"))
    parser.add_argument("--pytorch", default=os.path.join(out, "pytorch_outputs", "pytorch_resize.jsonl"))
    parser.add_argument("--report", default=os.path.join(out, "resize_forensics_report.txt"))
    parser.add_argument("--ranked-jsonl", default=os.path.join(out, "resize_forensics_ranked.jsonl"))
    parser.add_argument("--write-matching-implementation", default="")
    return parser.parse_args()


def load_oracle(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in read_jsonl(path):
        if row.get("phase") == "resize_fixture":
            rows[str(row["fixture_name"])] = row
    return rows


def bbox(diff: np.ndarray) -> str:
    coords = np.argwhere(diff)
    if coords.size == 0:
        return "none"
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return f"y={int(y_min)}..{int(y_max)},x={int(x_min)}..{int(x_max)}"


def first_pixel(diff: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict[str, Any] | None:
    coords = np.argwhere(diff)
    if coords.size == 0:
        return None
    y, x = [int(item) for item in coords[0]]
    return {
        "y": y,
        "x": x,
        "pytorch": int(left[y, x]),
        "deepmind": int(right[y, x]),
        "delta": int(left[y, x]) - int(right[y, x]),
    }


def compare_arrays(candidate_path: str | Path, oracle_path: str | Path) -> dict[str, Any]:
    observed = np.ascontiguousarray(np.load(candidate_path).astype(np.uint8))
    expected = np.ascontiguousarray(np.load(oracle_path).astype(np.uint8))
    if observed.shape != expected.shape:
        return {
            "shape_mismatch": True,
            "observed_shape": list(observed.shape),
            "expected_shape": list(expected.shape),
            "exact": False,
            "equal_pixels": 0,
            "total_pixels": int(expected.size),
            "mean_abs_sum": math.inf,
            "max_abs_diff": math.inf,
            "bbox": "shape_mismatch",
            "first_pixel": None,
        }
    abs_diff = np.abs(observed.astype(np.int16) - expected.astype(np.int16))
    equal = abs_diff == 0
    return {
        "shape_mismatch": False,
        "exact": bool(abs_diff.max() == 0) if abs_diff.size else True,
        "equal_pixels": int(equal.sum()),
        "total_pixels": int(expected.size),
        "mean_abs_sum": float(abs_diff.sum()),
        "max_abs_diff": int(abs_diff.max()) if abs_diff.size else 0,
        "bbox": bbox(~equal),
        "first_pixel": first_pixel(~equal, observed, expected),
    }


def summarize_candidate(rows: list[dict[str, Any]], oracle: dict[str, dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    fixture_count = 0
    exact_count = 0
    synthetic_count = 0
    exact_synthetic = 0
    atari_count = 0
    exact_atari = 0
    equal_pixels = 0
    total_pixels = 0
    abs_sum = 0.0
    max_abs = 0
    first_mismatch_fixture = None
    first_mismatch_group = None
    first_mismatch_bbox = "none"
    first_mismatch_pixel = None
    error_count = 0
    group_totals: dict[str, int] = {}
    group_exact: dict[str, int] = {}
    group_max_abs: dict[str, int] = {}

    for row in rows:
        fixture_name = str(row["fixture_name"])
        group = str(row["fixture_group"])
        group_totals[group] = group_totals.get(group, 0) + 1
        fixture_count += 1
        is_atari = group == "atari"
        if is_atari:
            atari_count += 1
        else:
            synthetic_count += 1

        if row.get("error"):
            error_count += 1
            if first_mismatch_fixture is None:
                first_mismatch_fixture = fixture_name
                first_mismatch_group = group
                first_mismatch_bbox = "error"
                first_mismatch_pixel = {"error": row["error"]}
            continue

        if fixture_name not in oracle:
            error_count += 1
            continue

        diff = compare_arrays(row["output_path"], oracle[fixture_name]["output_path"])
        exact = bool(diff["exact"])
        if exact:
            exact_count += 1
            group_exact[group] = group_exact.get(group, 0) + 1
            if is_atari:
                exact_atari += 1
            else:
                exact_synthetic += 1
        elif first_mismatch_fixture is None:
            first_mismatch_fixture = fixture_name
            first_mismatch_group = group
            first_mismatch_bbox = diff["bbox"]
            first_mismatch_pixel = diff["first_pixel"]

        equal_pixels += int(diff["equal_pixels"])
        total_pixels += int(diff["total_pixels"])
        abs_sum += float(diff["mean_abs_sum"])
        if diff["max_abs_diff"] != math.inf:
            max_abs = max(max_abs, int(diff["max_abs_diff"]))
        group_max_abs[group] = max(group_max_abs.get(group, 0), 0 if diff["max_abs_diff"] == math.inf else int(diff["max_abs_diff"]))

    percent_equal = 100.0 * equal_pixels / total_pixels if total_pixels else 0.0
    mean_abs_diff = abs_sum / total_pixels if total_pixels else math.inf
    return {
        "candidate": first["candidate"],
        "family": first["family"],
        "coordinate": first["coordinate"],
        "range_mode": first["range_mode"],
        "cast_rule": first["cast_rule"],
        "border_rule": first["border_rule"],
        "notes": first.get("notes") or "",
        "fixture_count": fixture_count,
        "exact_count": exact_count,
        "synthetic_count": synthetic_count,
        "exact_synthetic": exact_synthetic,
        "atari_count": atari_count,
        "exact_atari": exact_atari,
        "percent_equal": percent_equal,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs,
        "first_mismatch_fixture": first_mismatch_fixture,
        "first_mismatch_group": first_mismatch_group,
        "first_mismatch_bbox": first_mismatch_bbox,
        "first_mismatch_pixel": first_mismatch_pixel,
        "error_count": error_count,
        "group_totals": group_totals,
        "group_exact": group_exact,
        "group_max_abs": group_max_abs,
    }


def reproducibility(deepmind: dict[str, dict[str, Any]], repeat: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    failures = []
    for name, row in sorted(deepmind.items()):
        other = repeat.get(name)
        if other is None:
            failures.append(f"{name}: missing in repeat")
            continue
        left_hash = (row.get("output_frame") or {}).get("hash")
        right_hash = (other.get("output_frame") or {}).get("hash")
        if left_hash != right_hash:
            failures.append(f"{name}: {left_hash} != {right_hash}")
    return len(failures) == 0, failures


def format_table(rows: list[dict[str, Any]], limit: int = 30) -> str:
    headers = [
        "rank",
        "candidate",
        "exact",
        "synthetic",
        "atari",
        "percent_equal",
        "mean_abs",
        "max_abs",
        "first_mismatch",
    ]
    table_rows = []
    for rank, row in enumerate(rows[:limit], start=1):
        table_rows.append(
            [
                str(rank),
                row["candidate"],
                f"{row['exact_count']}/{row['fixture_count']}",
                f"{row['exact_synthetic']}/{row['synthetic_count']}",
                f"{row['exact_atari']}/{row['atari_count']}",
                f"{row['percent_equal']:.6f}",
                f"{row['mean_abs_diff']:.9f}",
                str(row["max_abs_diff"]),
                str(row["first_mismatch_fixture"]),
            ]
        )
    widths = [len(header) for header in headers]
    for row in table_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    lines = ["  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    for row in table_rows:
        lines.append("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    return "\n".join(lines)


def interpretation(best: dict[str, Any], area_rank: int | None, linear_rank: int | None, reproducible: bool) -> list[str]:
    lines = []
    lines.append(f"DeepMind oracle reproducible: {str(reproducible).lower()}")
    constants_total = best["group_totals"].get("constant", 0)
    constants_exact = best["group_exact"].get("constant", 0)
    lines.append(f"Best candidate constants exact: {constants_exact}/{constants_total}")
    if constants_exact != constants_total:
        lines.append("Conclusion: dtype/range/casting is still suspect because constants do not all match.")
    else:
        lines.append("Conclusion: dtype/range preservation passes on constants for the best candidate.")

    for group in ("ramp", "edge", "impulse", "checker", "small_matrix", "atari"):
        total = best["group_totals"].get(group, 0)
        exact = best["group_exact"].get(group, 0)
        max_abs = best["group_max_abs"].get(group, 0)
        if total:
            lines.append(f"Best candidate {group}: exact {exact}/{total}, max_abs_diff {max_abs}")

    if best["exact_count"] == best["fixture_count"]:
        lines.append(
            "Exact explanation: DeepMind resize matches "
            f"family={best['family']}, coordinate={best['coordinate']}, range={best['range_mode']}, "
            f"cast={best['cast_rule']}, border={best['border_rule']}."
        )
    else:
        if best["max_abs_diff"] <= 1:
            lines.append("Mismatch character: all observed differences for the best candidate are +/-1 or exact; rounding/accumulation/cast precision is the likely remaining gap.")
        elif best["max_abs_diff"] >= 20:
            lines.append("Mismatch character: large differences remain; interpolation family or coordinate convention is still wrong.")
        else:
            lines.append("Mismatch character: moderate differences remain; coordinate/kernel semantics are not fully replicated.")
        if area_rank is not None and linear_rank is not None:
            lines.append(f"OpenCV area best rank: {area_rank}; OpenCV linear best rank: {linear_rank}.")
            if area_rank < linear_rank:
                lines.append("Interpretation: area-like shrinking is closer than standard bilinear on these fixtures.")
            else:
                lines.append("Interpretation: standard bilinear is not beaten by area-like shrinking in this fixture set.")
        lines.append(
            "Unresolved: no Python candidate matched every synthetic and Atari fixture; do not claim byte-faithful preprocessing yet."
        )
    return lines


def write_matching_implementation(path: Path, candidate_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'''#!/usr/bin/env python
"""Generated Stage 2c DeepMind preprocessing clone.

This file is written only when resize forensics finds an exact byte match
against DeepMind Torch7 image.scale(..., "bilinear") for all synthetic and
frozen Atari luminance fixtures.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pytorch.torch7_resize_clone import candidate_registry  # noqa: E402


RESIZE_CANDIDATE = {candidate_name!r}


def _luminance(obs: np.ndarray) -> np.ndarray:
    rgb = np.asarray(obs, dtype=np.float64)
    return np.clip(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2], 0, 255).astype(np.uint8)


def preprocess_frame(obs: np.ndarray) -> np.ndarray:
    y = _luminance(obs)
    for candidate in candidate_registry():
        if candidate.name == RESIZE_CANDIDATE:
            return candidate.fn(y)
    raise RuntimeError(f"Stage 2c resize candidate not found: {{RESIZE_CANDIDATE}}")
''',
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    fixtures = [row for row in read_jsonl(args.fixtures) if row.get("phase") == "resize_fixture"]
    deepmind = load_oracle(args.deepmind)
    repeat = load_oracle(args.deepmind_repeat)
    reproducible, reproducibility_failures = reproducibility(deepmind, repeat)

    rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(args.pytorch):
        if row.get("phase") == "resize_fixture":
            rows_by_candidate.setdefault(str(row["candidate"]), []).append(row)

    ranked = [summarize_candidate(rows, deepmind) for rows in rows_by_candidate.values()]
    ranked.sort(
        key=lambda row: (
            row["error_count"] > 0,
            -int(row["exact_count"]),
            -int(row["exact_atari"]),
            float(row["mean_abs_diff"]),
            int(row["max_abs_diff"]),
            -float(row["percent_equal"]),
            row["candidate"],
        )
    )
    best = ranked[0]
    area_rank = next((i + 1 for i, row in enumerate(ranked) if row["candidate"].startswith("cv2_area")), None)
    linear_rank = next((i + 1 for i, row in enumerate(ranked) if row["candidate"].startswith("cv2_linear")), None)

    report_lines = [
        "Stage 2c Resize Forensics",
        "",
        f"fixtures: {len(fixtures)}",
        f"deepmind_outputs: {len(deepmind)}",
        f"candidate_count: {len(ranked)}",
        f"status: {'MATCH' if best['exact_count'] == best['fixture_count'] and reproducible else 'NO EXACT CLONE'}",
        "",
        "Gate 1: DeepMind Oracle Reproducibility",
        f"reproducible: {str(reproducible).lower()}",
    ]
    if reproducibility_failures:
        report_lines.extend(reproducibility_failures[:20])
    report_lines.extend(
        [
            "",
            "Top Candidates",
            format_table(ranked, limit=30),
            "",
            "Best Candidate Details",
            json.dumps(best, indent=2, sort_keys=True),
            "",
            "Diagnosis",
            *interpretation(best, area_rank, linear_rank, reproducible),
            "",
        ]
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    ranked_path = Path(args.ranked_jsonl)
    ranked_path.parent.mkdir(parents=True, exist_ok=True)
    with ranked_path.open("w", encoding="utf-8") as handle:
        for row in ranked:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")

    if best["exact_count"] == best["fixture_count"] and reproducible and args.write_matching_implementation:
        write_matching_implementation(Path(args.write_matching_implementation), str(best["candidate"]))

    print("\n".join(report_lines))
    return 0 if best["exact_count"] == best["fixture_count"] and reproducible else 1


if __name__ == "__main__":
    raise SystemExit(main())
