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
from run_resource_island_smoke import (
    activation_initial_inventory,
    activation_start_positions,
    specialization_preferences,
)
from worlds.public_goods.env import PublicGoodsConfig
from worlds.public_goods.training import (
    SUMMARY_METRICS as PUBLIC_GOODS_METRICS,
    summarize_records as summarize_public_goods,
    train_public_goods,
)
from worlds.resource_island.env import ResourceIslandConfig
from worlds.resource_island.training import (
    SUMMARY_METRICS as RESOURCE_ISLAND_METRICS,
    summarize_records as summarize_resource_island,
    train_resource_island,
)


WORLDS = ("public_goods", "resource_island")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{field: row.get(field, "NA") for field in fieldnames} for row in rows])


def flat_tax_params(tax_rate: float) -> dict[str, Any]:
    return {"brackets": ((0.0, float(tax_rate)),), "redistribute": True}


def public_goods_config(args: argparse.Namespace) -> PublicGoodsConfig:
    return PublicGoodsConfig(
        n_agents=args.public_goods_agents,
        max_rounds=args.steps,
        pool_capacity=args.pool_capacity,
        initial_pool=args.initial_pool,
        regeneration_rate=args.regeneration_rate,
    )


def resource_island_config(args: argparse.Namespace) -> ResourceIslandConfig:
    return ResourceIslandConfig(
        grid_size=args.grid_size,
        n_agents=args.resource_island_agents,
        max_steps=args.steps,
        initial_resource_units=args.initial_resource_units,
        resource_layout=args.resource_layout,
        resource_spawn_probability=args.resource_spawn_probability,
        trade_food_units=args.trade_food_units,
        trade_wood_units=args.trade_wood_units,
        trade_acquisition_reward=args.trade_acquisition_reward,
        resource_preferences=specialization_preferences(args.specialization_preset, args.resource_island_agents),
        start_positions=activation_start_positions(args.activation_preset, args.grid_size, args.resource_island_agents),
        initial_inventory=activation_initial_inventory(
            args.activation_preset,
            args.resource_island_agents,
            args.trade_food_units,
            args.trade_wood_units,
        ),
    )


def aggregate_rows(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    aggregate: list[dict[str, Any]] = []
    keys = sorted({(row["world"], float(row["tax_rate"])) for row in rows})
    for world, tax_rate in keys:
        subset = [row for row in rows if row["world"] == world and float(row["tax_rate"]) == tax_rate]
        out: dict[str, Any] = {"world": world, "institution": "tax_schedule", "tax_rate": tax_rate, "n_seeds": len(subset)}
        for metric in metrics:
            if metric not in subset[0]:
                continue
            values = [float(row[metric]) for row in subset]
            out[f"{metric}_mean"] = finite_mean(values)
            out[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        aggregate.append(out)
    return aggregate


def run_world(args: argparse.Namespace, world: str, tax_rate: float) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    params = flat_tax_params(tax_rate)
    rows: list[dict[str, Any]] = []
    if world == "public_goods":
        config = public_goods_config(args)
        metrics = PUBLIC_GOODS_METRICS
        for seed_index in range(args.n_seeds):
            seed = args.seed_start + seed_index * args.seed_stride
            seed_start = time.time()
            result = train_public_goods(
                steps=args.steps,
                seed=seed,
                institution="tax_schedule",
                institution_params=params,
                config=config,
                epsilon_start=args.epsilon_start,
                epsilon_min=args.epsilon_min,
                epsilon_decay=args.epsilon_decay,
            )
            summary = summarize_public_goods(result.records, final_window=args.final_window)
            rows.append(
                {
                    "world": world,
                    "institution": "tax_schedule",
                    "tax_rate": tax_rate,
                    "seed_index": seed_index,
                    "seed": seed,
                    "steps": args.steps,
                    "final_window": min(args.final_window, args.steps),
                    **summary,
                }
            )
            print(
                f"{world} tax_rate={tax_rate:.3f} seed_index={seed_index:03d} "
                f"elapsed_seed={time.time() - seed_start:.2f}s "
                f"welfare={summary['welfare']:.3f} inequality={summary['inequality']:.3f} "
                f"tax_revenue={summary['tax_revenue']:.3f}",
                flush=True,
            )
        return rows, metrics

    if world == "resource_island":
        config = resource_island_config(args)
        metrics = RESOURCE_ISLAND_METRICS
        for seed_index in range(args.n_seeds):
            seed = args.seed_start + seed_index * args.seed_stride
            seed_start = time.time()
            result = train_resource_island(
                steps=args.steps,
                seed=seed,
                institution="tax_schedule",
                institution_params=params,
                config=config,
                epsilon_start=args.epsilon_start,
                epsilon_min=args.epsilon_min,
                epsilon_decay=args.epsilon_decay,
            )
            summary = summarize_resource_island(result.records, final_window=args.final_window)
            rows.append(
                {
                    "world": world,
                    "institution": "tax_schedule",
                    "tax_rate": tax_rate,
                    "seed_index": seed_index,
                    "seed": seed,
                    "steps": args.steps,
                    "final_window": min(args.final_window, args.steps),
                    **summary,
                }
            )
            print(
                f"{world} tax_rate={tax_rate:.3f} seed_index={seed_index:03d} "
                f"elapsed_seed={time.time() - seed_start:.2f}s "
                f"welfare={summary['welfare']:.3f} survival={summary['survival_rate']:.3f} "
                f"inequality={summary['inequality_over_time']:.3f} tax_revenue={summary['tax_revenue']:.3f}",
                flush=True,
            )
        return rows, metrics

    raise ValueError(f"unsupported world {world!r}")


def run(args: argparse.Namespace) -> dict[str, Path]:
    start = time.time()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    metrics_by_world: dict[str, tuple[str, ...]] = {}
    for world in args.worlds:
        for tax_rate in args.tax_rates:
            print(f"\n=== Tax sweep: world={world} tax_rate={tax_rate:.3f} ===", flush=True)
            rows, metrics = run_world(args, world, float(tax_rate))
            all_rows.extend(rows)
            metrics_by_world[world] = metrics

    fieldnames = [
        "world",
        "institution",
        "tax_rate",
        "seed_index",
        "seed",
        "steps",
        "final_window",
        *sorted({metric for metrics in metrics_by_world.values() for metric in metrics}),
    ]
    aggregate_fields = ["world", "institution", "tax_rate", "n_seeds"]
    for metric in sorted({metric for metrics in metrics_by_world.values() for metric in metrics}):
        aggregate_fields.extend([f"{metric}_mean", f"{metric}_std"])

    aggregates: list[dict[str, Any]] = []
    for world, metrics in metrics_by_world.items():
        aggregates.extend(aggregate_rows([row for row in all_rows if row["world"] == world], metrics))

    by_seed_path = args.save_dir / "summary_by_seed.csv"
    aggregate_path = args.save_dir / "summary_aggregate.csv"
    manifest_path = args.save_dir / "experiment_manifest.json"
    write_csv(by_seed_path, all_rows, fieldnames)
    write_csv(aggregate_path, aggregates, aggregate_fields)
    manifest_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.time() - start,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "worlds": list(args.worlds),
                "institution": "tax_schedule",
                "tax_rates": [float(rate) for rate in args.tax_rates],
                "config": vars(args) | {"save_dir": str(args.save_dir)},
                "outputs": [str(by_seed_path), str(aggregate_path)],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    return {"summary_by_seed": by_seed_path, "summary_aggregate": aggregate_path, "manifest": manifest_path}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep flat tax schedules across existing worlds.")
    parser.add_argument("--worlds", nargs="+", choices=WORLDS, default=list(WORLDS))
    parser.add_argument("--tax-rates", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.4])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--final-window", type=int, default=200)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--public-goods-agents", type=int, default=4)
    parser.add_argument("--pool-capacity", type=float, default=20.0)
    parser.add_argument("--initial-pool", type=float, default=10.0)
    parser.add_argument("--regeneration-rate", type=float, default=0.08)
    parser.add_argument("--resource-island-agents", type=int, default=2)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--initial-resource-units", type=int, default=12)
    parser.add_argument("--resource-layout", choices=("random", "contested", "split"), default="contested")
    parser.add_argument("--activation-preset", choices=("none", "pressure"), default="pressure")
    parser.add_argument("--resource-spawn-probability", type=float, default=0.08)
    parser.add_argument("--trade-food-units", type=int, default=2)
    parser.add_argument("--trade-wood-units", type=int, default=1)
    parser.add_argument("--trade-acquisition-reward", type=float, default=0.2)
    parser.add_argument("--specialization-preset", choices=("none", "complementary"), default="complementary")
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/tax_schedule_sweep"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    outputs = run(parse_args(argv))
    for path in outputs.values():
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
