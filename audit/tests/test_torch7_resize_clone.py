#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pytorch.torch7_resize_clone import CERTIFIED_BYTE_EXACT, resize_torch7_exact  # noqa: E402


def parse_args() -> argparse.Namespace:
    out = os.getenv("STAGE2D_OUT", os.getenv("OUT", "audit_outputs/stage2d_resize_source"))
    parser = argparse.ArgumentParser(description="Validate Python Torch7 resize clone against installed Torch7 oracle.")
    parser.add_argument("--contract", default=os.getenv("STAGE2D_CONTRACT_JSON", os.path.join(out, "lua_contract.json")))
    parser.add_argument("--representation", default=os.getenv("STAGE2D_REPRESENTATION", "float_0_1"))
    parser.add_argument("--python-output-dir", default=os.path.join(out, "python_outputs"))
    parser.add_argument("--comparison", default=os.path.join(out, "comparison.txt"))
    parser.add_argument("--unresolved", default=os.path.join(out, "UNRESOLVED.md"))
    parser.add_argument("--source-report", default=os.path.join(out, "source_report.txt"))
    parser.add_argument("--write-preprocess", default="")
    return parser.parse_args()


def bbox(diff: np.ndarray) -> str:
    coords = np.argwhere(diff)
    if coords.size == 0:
        return "none"
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return f"y={int(y_min)}..{int(y_max)}, x={int(x_min)}..{int(x_max)}"


def mismatch_details(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if actual.shape != expected.shape:
        return {
            "shape_match": False,
            "dtype_match": actual.dtype == expected.dtype,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "first_diff": None,
            "differing_pixels": None,
            "mean_abs_diff": None,
            "max_abs_diff": None,
            "bbox": "shape_mismatch",
        }
    abs_diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    diff = abs_diff != 0
    first = None
    if diff.any():
        y, x = [int(v) for v in np.argwhere(diff)[0]]
        first = {
            "y": y,
            "x": x,
            "expected": int(expected[y, x]),
            "actual": int(actual[y, x]),
            "diff": int(actual[y, x]) - int(expected[y, x]),
        }
    return {
        "shape_match": True,
        "dtype_match": actual.dtype == expected.dtype,
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "first_diff": first,
        "differing_pixels": int(diff.sum()),
        "mean_abs_diff": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "max_abs_diff": int(abs_diff.max()) if abs_diff.size else 0,
        "bbox": bbox(diff),
    }


def load_contract(path: Path, representation: str) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    selected = []
    for row in records:
        if row.get("phase") != "stage2d_resize_contract":
            continue
        if row.get("input_representation") != representation:
            continue
        selected.append(row)
    return selected


def format_table(rows: list[dict[str, Any]], limit: int = 30) -> str:
    headers = ["fixture", "group", "exact", "diff_pixels", "mean_abs", "max_abs", "bbox", "first_diff"]
    table_rows = []
    for row in rows[:limit]:
        details = row["details"]
        table_rows.append(
            [
                row["fixture_name"],
                row["fixture_group"],
                str(row["exact"]).lower(),
                str(details.get("differing_pixels")),
                "none" if details.get("mean_abs_diff") is None else f"{details['mean_abs_diff']:.9f}",
                str(details.get("max_abs_diff")),
                str(details.get("bbox")),
                json.dumps(details.get("first_diff"), sort_keys=True),
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


def source_summary(source_report: Path) -> dict[str, str]:
    if not source_report.exists():
        return {
            "lua_wrapper": "unknown; source_report.txt missing",
            "native_library": "unknown; source_report.txt missing",
            "native_source": "unknown; source_report.txt missing",
        }
    text = source_report.read_text(encoding="utf-8", errors="replace")
    lua_wrapper = "not found"
    native_library = "not found"
    native_source = "not found"
    saw_native_source_section = False
    saw_image_native_source = False
    for line in text.splitlines():
        if "/share/lua/5.1/image/init.lua" in line and lua_wrapper == "not found":
            lua_wrapper = line.strip()
        if ("/image.so" in line or "libimage" in line) and native_library == "not found":
            native_library = line.strip()
        if line.strip() == "Native source candidates":
            saw_native_source_section = True
        if (
            saw_native_source_section
            and (line.endswith(".c") or line.endswith(".cc") or line.endswith(".cpp"))
            and "/image" in line.lower()
        ):
            native_source = line.strip()
            saw_image_native_source = True
            break
    if "Native source candidates" in text and (native_source == "not found" or not saw_image_native_source):
        native_source = "native image C/C++ source not found under installed Torch tree; only compiled libimage.so and Lua wrapper were available"
    return {
        "lua_wrapper": lua_wrapper,
        "native_library": native_library,
        "native_source": native_source,
    }


def write_unresolved(path: Path, report: list[str], summary: dict[str, Any], source: dict[str, str]) -> None:
    first_failure = summary["failures"][0] if summary["failures"] else None
    lines = [
        "# Stage 2d Unresolved",
        "",
        "The installed Torch7 resize contract was dumped and the Python clone was tested against the oracle, but exact equality did not pass.",
        "",
        "## Installed Source",
        "",
        f"- Lua wrapper: `{source['lua_wrapper']}`",
        f"- Native library: `{source['native_library']}`",
        f"- Native source: `{source['native_source']}`",
        "",
        "## Best Clone Behavior",
        "",
        f"- Representation tested: `{summary['representation']}`",
        f"- Exact fixtures: `{summary['exact_count']} / {summary['fixture_count']}`",
        f"- Equal pixels: `{summary['percent_equal']:.9f}%`",
        f"- Mean abs diff: `{summary['mean_abs_diff']:.12f}`",
        f"- Max abs diff: `{summary['max_abs_diff']}`",
        f"- Certified byte exact flag: `{str(CERTIFIED_BYTE_EXACT).lower()}`",
        "",
        "## First Failure",
        "",
    ]
    if first_failure is None:
        lines.append("No per-fixture failure was recorded, but the aggregate gate did not pass.")
    else:
        details = first_failure["details"]
        lines.extend(
            [
                f"- Fixture: `{first_failure['fixture_name']}`",
                f"- Group: `{first_failure['fixture_group']}`",
                f"- First differing coordinate: `{details.get('first_diff')}`",
                f"- Differing pixels: `{details.get('differing_pixels')}`",
                f"- Mean abs diff: `{details.get('mean_abs_diff')}`",
                f"- Max abs diff: `{details.get('max_abs_diff')}`",
                f"- BBox: `{details.get('bbox')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Likely Remaining Rule",
            "",
            "The remaining mismatch is in the resize operator's coordinate, kernel, accumulation precision, or final cast behavior. Do not replace this with OpenCV `INTER_AREA`; it is only a close candidate, not the installed Torch7 contract.",
            "",
            "## Comparison Excerpt",
            "",
            "```text",
            *report[:80],
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_deepmind_preprocess(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '''#!/usr/bin/env python
"""Generated Stage 2d DeepMind-compatible preprocessing.

This file is generated only after audit/tests/test_torch7_resize_clone.py
passes exact equality against the installed Torch7 resize oracle.
"""
from __future__ import annotations

import numpy as np

from pytorch.preprocess_variants import gray_bt601_unit_float32, quantize_unit
from pytorch.torch7_resize_clone import resize_torch7_exact


def rgb_to_bt601_exact(rgb_frame: np.ndarray) -> np.ndarray:
    return quantize_unit(gray_bt601_unit_float32(np.asarray(rgb_frame)), "trunc")


def preprocess_deepmind_full_y(rgb_frame: np.ndarray) -> np.ndarray:
    luminance = rgb_to_bt601_exact(rgb_frame)
    return resize_torch7_exact(luminance, 84, 84)


def preprocess_frame(rgb_frame: np.ndarray) -> np.ndarray:
    return preprocess_deepmind_full_y(rgb_frame)
''',
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    contract_path = Path(args.contract)
    python_output_dir = Path(args.python_output_dir)
    python_output_dir.mkdir(parents=True, exist_ok=True)

    records = load_contract(contract_path, args.representation)
    if not records:
        raise RuntimeError(f"no Stage 2d contract rows found for representation {args.representation!r} in {contract_path}")

    results: list[dict[str, Any]] = []
    exact_count = 0
    equal_pixels = 0
    total_pixels = 0
    abs_sum = 0.0
    max_abs = 0
    group_totals: dict[str, int] = {}
    group_exact: dict[str, int] = {}

    for row in records:
        fixture_name = str(row["fixture_name"])
        group = str(row["fixture_group"])
        group_totals[group] = group_totals.get(group, 0) + 1
        expected = np.ascontiguousarray(np.load(row["output_path"]).astype(np.uint8))
        source = np.ascontiguousarray(np.load(row["input_path"]).astype(np.uint8))
        actual = resize_torch7_exact(source, expected.shape[0], expected.shape[1])
        actual_path = python_output_dir / f"{fixture_name}.npy"
        np.save(actual_path, actual)

        details = mismatch_details(actual, expected)
        exact = bool(
            details["shape_match"]
            and details["dtype_match"]
            and details["differing_pixels"] == 0
        )
        if exact:
            exact_count += 1
            group_exact[group] = group_exact.get(group, 0) + 1
        if details["shape_match"]:
            pixels = int(expected.size)
            total_pixels += pixels
            differing = int(details["differing_pixels"] or 0)
            equal_pixels += pixels - differing
            abs_sum += float(details["mean_abs_diff"] or 0.0) * pixels
            max_abs = max(max_abs, int(details["max_abs_diff"] or 0))

        results.append(
            {
                "fixture_name": fixture_name,
                "fixture_group": group,
                "expected_path": row["output_path"],
                "actual_path": str(actual_path),
                "exact": exact,
                "details": details,
            }
        )

    failures = [row for row in results if not row["exact"]]
    percent_equal = 100.0 * equal_pixels / total_pixels if total_pixels else 0.0
    mean_abs_diff = abs_sum / total_pixels if total_pixels else float("inf")
    summary = {
        "representation": args.representation,
        "fixture_count": len(results),
        "exact_count": exact_count,
        "percent_equal": percent_equal,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs,
        "group_totals": group_totals,
        "group_exact": group_exact,
        "failures": failures,
    }

    lines = [
        "Stage 2d Torch7 Resize Clone Comparison",
        "",
        f"contract: {contract_path}",
        f"representation: {args.representation}",
        f"status: {'MATCH' if not failures else 'NO EXACT CLONE'}",
        f"CERTIFIED_BYTE_EXACT: {str(CERTIFIED_BYTE_EXACT).lower()}",
        f"fixtures: {exact_count}/{len(results)} exact",
        f"percent_equal: {percent_equal:.9f}",
        f"mean_abs_diff: {mean_abs_diff:.12f}",
        f"max_abs_diff: {max_abs}",
        "",
        "Group Summary",
    ]
    for group in sorted(group_totals):
        lines.append(f"{group}: {group_exact.get(group, 0)}/{group_totals[group]} exact")
    lines.extend(["", "First Failures", format_table(failures, limit=25) if failures else "none", ""])

    comparison_path = Path(args.comparison)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    if failures:
        write_unresolved(Path(args.unresolved), lines, summary, source_summary(Path(args.source_report)))
        return 1

    if args.write_preprocess:
        write_deepmind_preprocess(Path(args.write_preprocess))
    unresolved_path = Path(args.unresolved)
    if unresolved_path.exists():
        unresolved_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
