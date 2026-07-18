#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import arr_stats, frame_to_uint8_hwc, read_jsonl, write_jsonl  # noqa: E402


HEIGHT = 210
WIDTH = 160


def parse_args() -> argparse.Namespace:
    out = os.getenv("OUT", "audit_outputs/stage2c_resize")
    parser = argparse.ArgumentParser(description="Build synthetic grayscale resize fixtures.")
    parser.add_argument("--out-dir", default=os.path.join(out, "fixtures"))
    parser.add_argument(
        "--stage2b-dir",
        default=os.path.join(os.getenv("ATARI_DIR", "."), "audit_outputs", "stage2_preprocess"),
        help="Stage 2b output dir containing canonical_frames/transitions.jsonl.",
    )
    parser.add_argument("--max-atari", type=int, default=64)
    return parser.parse_args()


def fixture_path(out_dir: Path, name: str) -> Path:
    return out_dir / f"{name}.npy"


def save_fixture(rows: list[dict[str, Any]], out_dir: Path, name: str, group: str, arr: np.ndarray) -> None:
    frame = np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))
    path = fixture_path(out_dir, name)
    np.save(path, frame)
    stats = arr_stats(frame)
    rows.append(
        {
            "phase": "resize_fixture",
            "name": name,
            "group": group,
            "path": str(path),
            "shape": stats["shape"],
            "dtype": stats["dtype"],
            "min": stats.get("min"),
            "max": stats.get("max"),
            "mean": stats.get("mean"),
            "hash": stats["hash"],
        }
    )


def horizontal_ramp() -> np.ndarray:
    return np.tile(np.linspace(0, 255, WIDTH, dtype=np.float64), (HEIGHT, 1))


def vertical_ramp() -> np.ndarray:
    return np.tile(np.linspace(0, 255, HEIGHT, dtype=np.float64)[:, None], (1, WIDTH))


def diagonal_ramp() -> np.ndarray:
    x = np.linspace(0, 1, WIDTH, dtype=np.float64)
    y = np.linspace(0, 1, HEIGHT, dtype=np.float64)[:, None]
    return 255.0 * (x + y) / 2.0


def checker(period: int) -> np.ndarray:
    yy, xx = np.indices((HEIGHT, WIDTH))
    return ((yy // period + xx // period) % 2) * 255


def centered_rectangle() -> np.ndarray:
    arr = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    arr[HEIGHT // 4 : 3 * HEIGHT // 4, WIDTH // 4 : 3 * WIDTH // 4] = 255
    return arr


def impulses() -> np.ndarray:
    arr = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for y in range(0, HEIGHT, 10):
        for x in range(0, WIDTH, 10):
            arr[y, x] = 255
    return arr


def small_matrix(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    values = np.arange(h * w, dtype=np.float64).reshape(h, w)
    return np.rint(values * (255.0 / values.max())).astype(np.uint8)


def luminance_from_rgb(frame: np.ndarray) -> np.ndarray:
    rgb = frame_to_uint8_hwc(frame).astype(np.float64)
    return np.clip(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2], 0, 255).astype(np.uint8)


def add_atari_fixtures(rows: list[dict[str, Any]], out_dir: Path, stage2b_dir: Path, max_atari: int) -> None:
    transitions_path = stage2b_dir / "canonical_frames" / "transitions.jsonl"
    if not transitions_path.exists():
        return
    count = 0
    for row in read_jsonl(transitions_path):
        if row.get("phase") != "transition":
            continue
        pooled_path = row.get("pooled_path")
        if not pooled_path:
            continue
        frame = np.load(pooled_path)
        luminance = luminance_from_rgb(frame)
        save_fixture(rows, out_dir, f"atari_luminance_{int(row['step']):06d}", "atari", luminance)
        count += 1
        if count >= max_atari:
            break


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    save_fixture(rows, out_dir, "constant_000", "constant", np.zeros((HEIGHT, WIDTH), dtype=np.uint8))
    save_fixture(rows, out_dir, "constant_128", "constant", np.full((HEIGHT, WIDTH), 128, dtype=np.uint8))
    save_fixture(rows, out_dir, "constant_255", "constant", np.full((HEIGHT, WIDTH), 255, dtype=np.uint8))

    save_fixture(rows, out_dir, "horizontal_ramp", "ramp", horizontal_ramp())
    save_fixture(rows, out_dir, "vertical_ramp", "ramp", vertical_ramp())
    save_fixture(rows, out_dir, "diagonal_ramp", "ramp", diagonal_ramp())

    edge_v = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    edge_v[:, WIDTH // 2 :] = 255
    save_fixture(rows, out_dir, "vertical_edge_x080", "edge", edge_v)

    edge_h = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    edge_h[HEIGHT // 2 :, :] = 255
    save_fixture(rows, out_dir, "horizontal_edge_y105", "edge", edge_h)

    save_fixture(rows, out_dir, "centered_rectangle", "edge", centered_rectangle())

    thin_h = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    thin_h[HEIGHT // 2, :] = 255
    save_fixture(rows, out_dir, "thin_horizontal_line", "edge", thin_h)

    thin_v = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    thin_v[:, WIDTH // 2] = 255
    save_fixture(rows, out_dir, "thin_vertical_line", "edge", thin_v)

    for name, y, x in (
        ("impulse_center", HEIGHT // 2, WIDTH // 2),
        ("impulse_top_left", 1, 1),
        ("impulse_top_right", 1, WIDTH - 2),
        ("impulse_bottom_left", HEIGHT - 2, 1),
        ("impulse_bottom_right", HEIGHT - 2, WIDTH - 2),
    ):
        arr = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        arr[y, x] = 255
        save_fixture(rows, out_dir, name, "impulse", arr)

    save_fixture(rows, out_dir, "impulse_grid_10", "impulse", impulses())

    for period in (1, 2, 4):
        save_fixture(rows, out_dir, f"checker_{period}px", "checker", checker(period))

    save_fixture(rows, out_dir, "small_matrix_4x4", "small_matrix", small_matrix((4, 4)))
    save_fixture(rows, out_dir, "small_matrix_8x8", "small_matrix", small_matrix((8, 8)))

    add_atari_fixtures(rows, out_dir, Path(args.stage2b_dir), args.max_atari)

    write_jsonl(out_dir / "fixtures.jsonl", rows)
    paths_path = out_dir / "fixture_paths.txt"
    with paths_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row['name']}\t{row['group']}\t{row['path']}\n")

    print(f"wrote {out_dir / 'fixtures.jsonl'}")
    print(f"wrote {paths_path}")
    print(f"fixtures: {len(rows)}")


if __name__ == "__main__":
    main()
