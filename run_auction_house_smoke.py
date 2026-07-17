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
from worlds.auction_house.env import AuctionHouseConfig
from worlds.auction_house.training import (
    SUMMARY_METRICS,
    SUPPORTED_AUCTION_MINDS,
    benchmark_for_config,
    learned_bid_curve,
    summarize_records,
    train_auction_house,
)


SCENARIOS = (
    "second_price",
    "first_price",
    "second_price_reserve",
    "clock",
    "second_price_public_signal",
    "second_price_noisy_signal",
)


def scenario_config(name: str, steps: int, n_bidders: int = 2) -> AuctionHouseConfig:
    grid = tuple(float(value) for value in range(11))
    if name == "second_price":
        return AuctionHouseConfig(
            n_bidders=n_bidders,
            auction_format="second_price",
            max_rounds=steps,
            bid_grid=grid,
            valuation_grid=grid,
            valuation_low=0.0,
            valuation_high=10.0,
            reserve_price=0.0,
        )
    if name == "first_price":
        return AuctionHouseConfig(
            n_bidders=n_bidders,
            auction_format="first_price",
            max_rounds=steps,
            bid_grid=grid,
            valuation_grid=grid,
            valuation_low=0.0,
            valuation_high=10.0,
            reserve_price=0.0,
        )
    if name == "second_price_reserve":
        return AuctionHouseConfig(
            n_bidders=n_bidders,
            auction_format="second_price",
            max_rounds=steps,
            bid_grid=grid,
            valuation_grid=grid,
            valuation_low=0.0,
            valuation_high=10.0,
            reserve_price=4.0,
        )
    if name == "clock":
        return AuctionHouseConfig(
            n_bidders=n_bidders,
            auction_format="clock",
            max_rounds=steps,
            bid_grid=grid,
            valuation_grid=grid,
            valuation_low=0.0,
            valuation_high=10.0,
            reserve_price=0.0,
        )
    if name in ("second_price_public_signal", "second_price_noisy_signal"):
        return AuctionHouseConfig(
            n_bidders=n_bidders,
            auction_format="second_price",
            max_rounds=steps,
            bid_grid=grid,
            valuation_grid=grid,
            valuation_low=0.0,
            valuation_high=10.0,
            reserve_price=0.0,
        )
    raise ValueError(f"unknown Auction House scenario {name!r}")


def scenario_institution(name: str, seed: int) -> tuple[str, dict[str, Any]]:
    if name == "second_price_public_signal":
        return "auction_information_policy", {"public_signal_weight": 1.0, "seed": seed}
    if name == "second_price_noisy_signal":
        return "auction_information_policy", {"noise_bins": 1, "seed": seed}
    return "none", {}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]], scenarios: list[str]) -> list[dict[str, Any]]:
    aggregate: list[dict[str, Any]] = []
    for scenario in scenarios:
        subset = [row for row in rows if row["scenario"] == scenario]
        if not subset:
            continue
        out: dict[str, Any] = {"scenario": scenario, "n_seeds": len(subset)}
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in subset]
            out[f"{metric}_mean"] = finite_mean(values)
            out[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        for key in sorted(subset[0]):
            if not key.startswith("benchmark_"):
                continue
            values = [float(row[key]) for row in subset]
            out[key] = finite_mean(values)
        aggregate.append(out)
    return aggregate


def run(args: argparse.Namespace) -> dict[str, Path]:
    start = time.time()
    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    for scenario in args.scenarios:
        config = scenario_config(scenario, steps=args.steps, n_bidders=args.n_bidders)
        benchmark = benchmark_for_config(config)
        print(f"\n=== Auction scenario: {scenario} ===", flush=True)
        for seed_index in range(args.n_seeds):
            seed = args.seed_start + seed_index * args.seed_stride
            institution, institution_params = scenario_institution(scenario, seed)
            seed_start_time = time.time()
            print(f"seed_index={seed_index:03d} seed={seed}", flush=True)
            result = train_auction_house(
                steps=args.steps,
                seed=seed,
                config=config,
                institution=institution,
                institution_params=institution_params,
                epsilon_start=args.epsilon_start,
                epsilon_min=args.epsilon_min,
                epsilon_decay=args.epsilon_decay,
                mind=args.mind,
            )
            summary = summarize_records(result.records, final_window=args.final_window)
            row = {
                "scenario": scenario,
                "auction_format": config.auction_format,
                "institution": institution,
                "reserve_price": config.reserve_price,
                "seed_index": seed_index,
                "seed": seed,
                "steps": args.steps,
                "final_window": min(args.final_window, args.steps),
                **summary,
            }
            for key, value in benchmark.items():
                row[f"benchmark_{key}"] = float(value)
            rows.append(row)
            for curve in learned_bid_curve(result.agents, config.valuation_grid):
                bidder_action = int(curve["greedy_action"])
                curve_rows.append(
                    {
                        "scenario": scenario,
                        "seed_index": seed_index,
                        "seed": seed,
                        "bidder_id": int(curve["bidder_id"]),
                        "valuation_bin": int(curve["valuation_bin"]),
                        "valuation": curve["valuation"],
                        "greedy_action": bidder_action,
                        "greedy_bid": config.bid_grid[bidder_action],
                    }
                )
            elapsed_seed = time.time() - seed_start_time
            scenario_rows = [source for source in rows if source["scenario"] == scenario]
            running_regret = finite_mean(float(source["ex_post_regret_mean"]) for source in scenario_rows)
            print(
                f"  elapsed_seed={elapsed_seed:.2f}s "
                f"revenue={summary['revenue']:.3f} "
                f"welfare={summary['welfare']:.3f} "
                f"efficiency={summary['allocative_efficiency']:.3f} "
                f"regret={summary['ex_post_regret_mean']:.3f} "
                f"running_regret={running_regret:.3f}",
                flush=True,
            )

    aggregate = aggregate_rows(rows, args.scenarios)
    by_seed_path = save_dir / "summary_by_seed.csv"
    aggregate_path = save_dir / "summary_aggregate.csv"
    curves_path = save_dir / "bid_curves.csv"
    manifest_path = save_dir / "experiment_manifest.json"

    seed_fields = [
        "scenario",
        "auction_format",
        "institution",
        "reserve_price",
        "seed_index",
        "seed",
        "steps",
        "final_window",
        *SUMMARY_METRICS,
        "benchmark_revenue",
        "benchmark_bidder_surplus",
        "benchmark_welfare",
        "benchmark_max_possible_welfare",
        "benchmark_allocative_efficiency",
        "benchmark_welfare_efficiency",
    ]
    write_csv(by_seed_path, rows, seed_fields)

    aggregate_fields = ["scenario", "n_seeds"]
    for metric in SUMMARY_METRICS:
        aggregate_fields.extend([f"{metric}_mean", f"{metric}_std"])
    aggregate_fields.extend(
        [
            "benchmark_revenue",
            "benchmark_bidder_surplus",
            "benchmark_welfare",
            "benchmark_max_possible_welfare",
            "benchmark_allocative_efficiency",
            "benchmark_welfare_efficiency",
        ]
    )
    write_csv(aggregate_path, aggregate, aggregate_fields)
    write_csv(
        curves_path,
        curve_rows,
        ["scenario", "seed_index", "seed", "bidder_id", "valuation_bin", "valuation", "greedy_action", "greedy_bid"],
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - start,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
            "world": "auction_house",
        "mind": args.mind,
        "config": {
            "steps": args.steps,
            "n_seeds": args.n_seeds,
            "seed_start": args.seed_start,
            "seed_stride": args.seed_stride,
            "final_window": args.final_window,
            "n_bidders": args.n_bidders,
            "scenarios": list(args.scenarios),
            "epsilon_start": args.epsilon_start,
            "epsilon_min": args.epsilon_min,
            "epsilon_decay": args.epsilon_decay,
            "save_dir": str(save_dir),
        },
        "summary_metrics": list(SUMMARY_METRICS),
        "outputs": [str(by_seed_path), str(aggregate_path), str(curves_path)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "summary_by_seed": by_seed_path,
        "summary_aggregate": aggregate_path,
        "bid_curves": curves_path,
        "manifest": manifest_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Auction House Q-learning smoke experiments.")
    parser.add_argument("--mind", choices=SUPPORTED_AUCTION_MINDS, default="q_learning")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--final-window", type=int, default=500)
    parser.add_argument("--n-bidders", type=int, default=2)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS)
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/auction_house_smoke"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    outputs = run(parse_args(argv))
    for path in outputs.values():
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
