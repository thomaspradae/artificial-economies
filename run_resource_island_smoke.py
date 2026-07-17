from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from core.metrics import finite_mean
from worlds.resource_island.benchmarks import efficient_gather_upper_bound
from worlds.resource_island.env import ResourceIslandConfig
from worlds.resource_island.training import (
    SUMMARY_METRICS,
    SUPPORTED_RESOURCE_ISLAND_MINDS,
    summarize_records,
    train_resource_island,
)


DEFAULT_INSTITUTIONS = ("none", "property_rights", "redistribution", "trade_price_controls", "reputation_system")


def specialization_preferences(preset: str, n_agents: int) -> tuple[tuple[float, float], ...] | None:
    if preset == "none":
        return None
    if preset != "complementary":
        raise ValueError(f"unknown specialization preset {preset!r}")
    preferences: list[tuple[float, float]] = []
    for agent_id in range(n_agents):
        if agent_id % 2 == 0:
            preferences.append((1.4, 0.6))
        else:
            preferences.append((0.6, 1.4))
    return tuple(preferences)


def activation_start_positions(preset: str, grid_size: int, n_agents: int) -> tuple[tuple[int, int], ...] | None:
    if preset == "none":
        return None
    if preset != "pressure":
        raise ValueError(f"unknown activation preset {preset!r}")
    center = grid_size // 2
    positions = [(center, center), (center, min(grid_size - 1, center + 1))]
    corners = [(0, 0), (grid_size - 1, grid_size - 1), (0, grid_size - 1), (grid_size - 1, 0)]
    for candidate in corners:
        if len(positions) >= n_agents:
            break
        if candidate not in positions:
            positions.append(candidate)
    return tuple(positions[:n_agents])


def activation_initial_inventory(
    preset: str,
    n_agents: int,
    trade_food_units: int,
    trade_wood_units: int,
) -> tuple[tuple[int, int], ...] | None:
    if preset == "none":
        return None
    if preset != "pressure":
        raise ValueError(f"unknown activation preset {preset!r}")
    inventory = [[0, 0] for _ in range(n_agents)]
    if n_agents >= 2:
        inventory[0][0] = int(trade_food_units)
        inventory[1][1] = int(trade_wood_units)
    return tuple((food, wood) for food, wood in inventory)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]], institutions: list[str]) -> list[dict[str, Any]]:
    aggregate: list[dict[str, Any]] = []
    for institution in institutions:
        subset = [row for row in rows if row["institution"] == institution]
        if not subset:
            continue
        out: dict[str, Any] = {"institution": institution, "n_seeds": len(subset)}
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in subset]
            out[f"{metric}_mean"] = finite_mean(values)
            out[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        aggregate.append(out)
    return aggregate


def run(args: argparse.Namespace) -> dict[str, Path]:
    start = time.time()
    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    resource_layout = (
        "contested"
        if args.activation_preset == "pressure" and args.resource_layout == "random"
        else args.resource_layout
    )
    config = ResourceIslandConfig(
        grid_size=args.grid_size,
        n_agents=args.n_agents,
        max_steps=args.steps,
        initial_resource_units=args.initial_resource_units,
        resource_layout=resource_layout,
        resource_spawn_probability=args.resource_spawn_probability,
        trade_radius=args.trade_radius,
        trade_food_units=args.trade_food_units,
        trade_wood_units=args.trade_wood_units,
        trade_acquisition_reward=args.trade_acquisition_reward,
        resource_preferences=specialization_preferences(args.specialization_preset, args.n_agents),
        start_positions=activation_start_positions(args.activation_preset, args.grid_size, args.n_agents),
        initial_inventory=activation_initial_inventory(
            args.activation_preset,
            args.n_agents,
            args.trade_food_units,
            args.trade_wood_units,
        ),
    )

    rows: list[dict[str, Any]] = []
    for institution in args.institutions:
        print(f"\n=== Institution: {institution} ===", flush=True)
        for seed_index in range(args.n_seeds):
            seed = args.seed_start + seed_index * args.seed_stride
            seed_start_time = time.time()
            print(f"seed_index={seed_index:03d} seed={seed}", flush=True)
            result = train_resource_island(
                steps=args.steps,
                seed=seed,
                institution=institution,
                config=config,
                mind=args.mind,
                obs_radius=args.obs_radius,
            )
            summary = summarize_records(result.records, final_window=args.final_window)
            upper_bound = efficient_gather_upper_bound(
                result.world.resources,
                n_agents=config.n_agents,
                steps=args.steps,
            )
            rows.append(
                {
                    "institution": institution,
                    "seed_index": seed_index,
                    "seed": seed,
                    "steps": args.steps,
                    "final_window": min(args.final_window, args.steps),
                    "efficient_gather_upper_bound_remaining": upper_bound,
                    **summary,
                }
            )
            institution_rows = [row for row in rows if row["institution"] == institution]
            running_welfare = finite_mean(float(row["welfare"]) for row in institution_rows)
            elapsed_seed = time.time() - seed_start_time
            print(
                f"  elapsed_seed={elapsed_seed:.2f}s "
                f"welfare={summary['welfare']:.3f} "
                f"survival={summary['survival_rate']:.3f} "
                f"inequality={summary['inequality_over_time']:.3f} "
                f"sustainability={summary['resource_sustainability']:.3f} "
                f"running_welfare={running_welfare:.3f}",
                flush=True,
            )

    aggregate = aggregate_rows(rows, args.institutions)
    by_seed_path = save_dir / "summary_by_seed.csv"
    aggregate_path = save_dir / "summary_aggregate.csv"
    manifest_path = save_dir / "experiment_manifest.json"
    write_csv(
        by_seed_path,
        rows,
        [
            "institution",
            "seed_index",
            "seed",
            "steps",
            "final_window",
            "efficient_gather_upper_bound_remaining",
            *SUMMARY_METRICS,
        ],
    )
    aggregate_fields = ["institution", "n_seeds"]
    for metric in SUMMARY_METRICS:
        aggregate_fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(aggregate_path, aggregate, aggregate_fields)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - start,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": {
            "steps": args.steps,
            "n_seeds": args.n_seeds,
            "seed_start": args.seed_start,
            "seed_stride": args.seed_stride,
            "final_window": args.final_window,
            "institutions": list(args.institutions),
            "grid_size": args.grid_size,
            "n_agents": args.n_agents,
            "initial_resource_units": args.initial_resource_units,
            "resource_layout": resource_layout,
            "activation_preset": args.activation_preset,
            "resource_spawn_probability": args.resource_spawn_probability,
            "trade_radius": config.trade_radius,
            "trade_food_units": args.trade_food_units,
            "trade_wood_units": args.trade_wood_units,
            "trade_acquisition_reward": args.trade_acquisition_reward,
            "specialization_preset": args.specialization_preset,
            "mind": args.mind,
            "obs_radius": args.obs_radius,
            "save_dir": str(save_dir),
        },
        "world": "resource_island",
        "mind": args.mind,
        "summary_metrics": list(SUMMARY_METRICS),
        "outputs": [str(by_seed_path), str(aggregate_path)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"summary_by_seed": by_seed_path, "summary_aggregate": aggregate_path, "manifest": manifest_path}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short Resource Island smoke experiment.")
    parser.add_argument("--mind", choices=SUPPORTED_RESOURCE_ISLAND_MINDS, default="q_learning")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--final-window", type=int, default=50)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--initial-resource-units", type=int, default=12)
    parser.add_argument("--resource-layout", choices=("random", "contested", "split"), default="random")
    parser.add_argument("--activation-preset", choices=("none", "pressure"), default="none")
    parser.add_argument("--resource-spawn-probability", type=float, default=0.08)
    parser.add_argument("--trade-radius", type=int, default=None)
    parser.add_argument("--trade-food-units", type=int, default=1)
    parser.add_argument("--trade-wood-units", type=int, default=1)
    parser.add_argument("--trade-acquisition-reward", type=float, default=0.0)
    parser.add_argument("--specialization-preset", choices=("none", "complementary"), default="none")
    parser.add_argument("--obs-radius", type=int, default=1)
    parser.add_argument("--institutions", nargs="+", default=list(DEFAULT_INSTITUTIONS), choices=DEFAULT_INSTITUTIONS)
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/resource_island_smoke"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    outputs = run(parse_args(argv))
    for path in outputs.values():
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
