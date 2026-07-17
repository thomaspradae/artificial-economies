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
from worlds.labor_market.env import LaborMarketConfig
from worlds.labor_market.training import (
    SUMMARY_METRICS,
    SUPPORTED_LABOR_MARKET_MINDS,
    benchmark_for_config,
    summarize_records,
    train_labor_market,
)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: dict[str, Any] = {"institution": "deferred_acceptance", "n_seeds": len(rows)}
    for metric in SUMMARY_METRICS:
        values = [float(row[metric]) for row in rows]
        out[f"{metric}_mean"] = finite_mean(values)
        out[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    for key in sorted(rows[0]):
        if key.startswith("benchmark_"):
            out[key] = finite_mean(float(row[key]) for row in rows)
    return [out]


def run(args: argparse.Namespace) -> dict[str, Path]:
    start = time.time()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    config = LaborMarketConfig(n_workers=args.n_workers, n_employers=args.n_employers, max_rounds=args.steps)
    benchmark = benchmark_for_config(config)
    rows: list[dict[str, Any]] = []
    print("\n=== Labor Market institution: deferred_acceptance ===", flush=True)
    for seed_index in range(args.n_seeds):
        seed = args.seed_start + seed_index * args.seed_stride
        seed_start = time.time()
        result = train_labor_market(
            steps=args.steps,
            seed=seed,
            config=config,
            epsilon_start=args.epsilon_start,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            mind=args.mind,
        )
        summary = summarize_records(result.records, final_window=args.final_window)
        row = {
            "institution": "deferred_acceptance",
            "seed_index": seed_index,
            "seed": seed,
            "steps": args.steps,
            "final_window": min(args.final_window, args.steps),
            **summary,
        }
        for key, value in benchmark.items():
            row[f"benchmark_{key}"] = value
        rows.append(row)
        print(
            f"seed_index={seed_index:03d} seed={seed} "
            f"elapsed_seed={time.time() - seed_start:.2f}s "
            f"match_rate={summary['match_rate']:.3f} "
            f"stability={summary['stability']:.3f} "
            f"truthful={summary['truthful_report_rate']:.3f} "
            f"welfare={summary['total_welfare']:.3f}",
            flush=True,
        )

    aggregate = aggregate_rows(rows)
    by_seed_path = args.save_dir / "summary_by_seed.csv"
    aggregate_path = args.save_dir / "summary_aggregate.csv"
    manifest_path = args.save_dir / "experiment_manifest.json"
    seed_fields = [
        "institution",
        "seed_index",
        "seed",
        "steps",
        "final_window",
        *SUMMARY_METRICS,
        *[f"benchmark_{key}" for key in benchmark],
    ]
    aggregate_fields = ["institution", "n_seeds"]
    for metric in SUMMARY_METRICS:
        aggregate_fields.extend([f"{metric}_mean", f"{metric}_std"])
    aggregate_fields.extend([f"benchmark_{key}" for key in benchmark])
    write_csv(by_seed_path, rows, seed_fields)
    write_csv(aggregate_path, aggregate, aggregate_fields)
    manifest_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.time() - start,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "world": "labor_market",
                "mind": args.mind,
                "config": vars(args) | {"save_dir": str(args.save_dir)},
                "summary_metrics": list(SUMMARY_METRICS),
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
    parser = argparse.ArgumentParser(description="Run Labor Market Q-learning smoke experiments.")
    parser.add_argument("--mind", choices=SUPPORTED_LABOR_MARKET_MINDS, default="q_learning")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--final-window", type=int, default=200)
    parser.add_argument("--n-workers", type=int, default=3)
    parser.add_argument("--n-employers", type=int, default=3)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/labor_market_smoke"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    outputs = run(parse_args(argv))
    for path in outputs.values():
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
