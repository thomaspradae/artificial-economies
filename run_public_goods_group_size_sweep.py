from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from core.metrics import finite_mean
from run_public_goods_smoke import INSTITUTIONS
from worlds.public_goods.env import PublicGoodsConfig
from worlds.public_goods.training import (
    SUMMARY_METRICS,
    SUPPORTED_PUBLIC_GOODS_MINDS,
    benchmark_for_config,
    summarize_records,
    train_public_goods,
)


DEFAULT_AGENT_COUNTS = (2, 4, 8, 16)
DEFAULT_INSTITUTIONS = (
    "none",
    "contribution_matching",
    "public_goods_reputation",
    "tax_schedule",
)
DEFAULT_MINDS = (
    "q_learning",
    "dqn",
    "ppo",
    "independent_dqn",
    "centralized_critic",
)
SCALING_METRICS = (
    "welfare",
    "sustainability",
    "contribution_total",
    "contribution_rate",
    "extraction_total",
    "extraction_rate",
    "collapse_rate",
)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _run_key(row: dict[str, Any]) -> tuple[int, str, str, int]:
    return (
        int(row["n_agents"]),
        str(row["mind"]),
        str(row["institution"]),
        int(row["seed_index"]),
    )


def _group_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return int(row["n_agents"]), str(row["mind"]), str(row["institution"])


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate: list[dict[str, Any]] = []
    keys = sorted({_group_key(row) for row in rows})
    for n_agents, mind, institution in keys:
        subset = [
            row
            for row in rows
            if int(row["n_agents"]) == n_agents
            and row["mind"] == mind
            and row["institution"] == institution
        ]
        if not subset:
            continue
        out: dict[str, Any] = {
            "n_agents": n_agents,
            "mind": mind,
            "institution": institution,
            "n_seeds": len(subset),
            "steps": subset[0]["steps"],
            "final_window": subset[0]["final_window"],
        }
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in subset]
            out[f"{metric}_mean"] = finite_mean(values)
            out[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        for key in sorted(subset[0]):
            if key.startswith("benchmark_"):
                out[key] = finite_mean(float(row[key]) for row in subset)
        aggregate.append(out)
    return aggregate


def scaling_effect_rows(
    aggregate: list[dict[str, Any]],
    baseline_n_agents: int,
    metrics: Iterable[str] = SCALING_METRICS,
) -> list[dict[str, Any]]:
    lookup = {
        (int(row["n_agents"]), str(row["mind"]), str(row["institution"])): row
        for row in aggregate
    }
    effects: list[dict[str, Any]] = []
    for row in aggregate:
        n_agents = int(row["n_agents"])
        mind = str(row["mind"])
        institution = str(row["institution"])
        baseline = lookup.get((baseline_n_agents, mind, institution))
        if baseline is None:
            continue
        out: dict[str, Any] = {
            "baseline_n_agents": baseline_n_agents,
            "n_agents": n_agents,
            "mind": mind,
            "institution": institution,
        }
        for metric in metrics:
            current = float(row[f"{metric}_mean"])
            base = float(baseline[f"{metric}_mean"])
            out[f"{metric}_delta_vs_baseline_n"] = current - base
            out[f"{metric}_ratio_vs_baseline_n"] = current / base if base != 0 else float("nan")
        effects.append(out)
    return effects


def validate_args(args: argparse.Namespace) -> None:
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.n_seeds < 1:
        raise ValueError("--n-seeds must be positive")
    if args.seed_stride < 1:
        raise ValueError("--seed-stride must be positive")
    if args.final_window < 1:
        raise ValueError("--final-window must be positive")
    if args.epsilon_min < 0 or args.epsilon_start < 0:
        raise ValueError("epsilon values cannot be negative")
    if not args.agent_counts:
        raise ValueError("--agent-counts cannot be empty")
    if any(count < 1 for count in args.agent_counts):
        raise ValueError("--agent-counts must contain positive integers")
    if len(set(args.agent_counts)) != len(args.agent_counts):
        raise ValueError("--agent-counts cannot contain duplicates")


def run(args: argparse.Namespace) -> dict[str, Path]:
    validate_args(args)
    start = time.time()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    by_seed_path = args.save_dir / "summary_by_seed.csv"
    aggregate_path = args.save_dir / "summary_aggregate.csv"
    scaling_path = args.save_dir / "scaling_effects.csv"
    manifest_path = args.save_dir / "experiment_manifest.json"
    rows: list[dict[str, Any]] = read_csv(by_seed_path) if args.resume else []
    completed = {_run_key(row) for row in rows}

    def checkpoint() -> None:
        if not rows:
            return
        aggregate = aggregate_rows(rows)
        scaling = scaling_effect_rows(aggregate, baseline_n_agents=min(args.agent_counts))

        benchmark_fields = sorted(key for key in rows[0] if key.startswith("benchmark_"))
        seed_fields = [
            "n_agents",
            "mind",
            "institution",
            "seed_index",
            "seed",
            "steps",
            "final_window",
            *SUMMARY_METRICS,
            *benchmark_fields,
        ]
        aggregate_fields = ["n_agents", "mind", "institution", "n_seeds", "steps", "final_window"]
        for metric in SUMMARY_METRICS:
            aggregate_fields.extend([f"{metric}_mean", f"{metric}_std"])
        aggregate_fields.extend(benchmark_fields)
        scaling_fields = ["baseline_n_agents", "n_agents", "mind", "institution"]
        for metric in SCALING_METRICS:
            scaling_fields.extend([f"{metric}_delta_vs_baseline_n", f"{metric}_ratio_vs_baseline_n"])

        write_csv(by_seed_path, rows, seed_fields)
        write_csv(aggregate_path, aggregate, aggregate_fields)
        write_csv(scaling_path, scaling, scaling_fields)
        manifest_path.write_text(
            json.dumps(
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": time.time() - start,
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                    "world": "public_goods",
                    "question": (
                        "Group-size robustness sweep for commons/free-riding behavior and "
                        "institution effects as n_agents changes."
                    ),
                    "config": vars(args) | {"save_dir": str(args.save_dir)},
                    "summary_metrics": list(SUMMARY_METRICS),
                    "scaling_metrics": list(SCALING_METRICS),
                    "completed_seed_rows": len(rows),
                    "outputs": [str(by_seed_path), str(aggregate_path), str(scaling_path)],
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )

    for n_agents in args.agent_counts:
        config = PublicGoodsConfig(
            n_agents=n_agents,
            max_rounds=args.steps,
            pool_capacity=args.pool_capacity,
            initial_pool=args.initial_pool,
            regeneration_rate=args.regeneration_rate,
        )
        benchmark = benchmark_for_config(config, steps=min(args.steps, args.benchmark_steps))
        for mind in args.minds:
            for institution in args.institutions:
                print(
                    f"\n=== Public Goods n_agents={n_agents} mind={mind} institution={institution} ===",
                    flush=True,
                )
                for seed_index in range(args.n_seeds):
                    seed = args.seed_start + seed_index * args.seed_stride
                    key = (n_agents, mind, institution, seed_index)
                    if key in completed:
                        print(
                            f"skip seed_index={seed_index:03d} seed={seed} "
                            f"n_agents={n_agents} mind={mind} institution={institution}",
                            flush=True,
                        )
                        continue
                    seed_start = time.time()
                    result = train_public_goods(
                        steps=args.steps,
                        seed=seed,
                        institution=institution,
                        config=config,
                        epsilon_start=args.epsilon_start,
                        epsilon_min=args.epsilon_min,
                        epsilon_decay=args.epsilon_decay,
                        mind=mind,
                    )
                    summary = summarize_records(result.records, final_window=args.final_window)
                    row = {
                        "n_agents": n_agents,
                        "mind": mind,
                        "institution": institution,
                        "seed_index": seed_index,
                        "seed": seed,
                        "steps": args.steps,
                        "final_window": min(args.final_window, args.steps),
                        **summary,
                    }
                    for key, value in benchmark.items():
                        row[f"benchmark_{key}"] = value
                    rows.append(row)
                    completed.add(_run_key(row))
                    print(
                        f"seed_index={seed_index:03d} seed={seed} "
                        f"elapsed_seed={time.time() - seed_start:.2f}s "
                        f"welfare={summary['welfare']:.3f} "
                        f"sustainability={summary['sustainability']:.3f} "
                        f"contribution={summary['contribution_total']:.3f} "
                        f"collapse={summary['collapse_rate']:.3f}",
                        flush=True,
                    )
                    checkpoint()

    checkpoint()
    return {
        "summary_by_seed": by_seed_path,
        "summary_aggregate": aggregate_path,
        "scaling_effects": scaling_path,
        "manifest": manifest_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Public Goods group-size sweeps across minds and institutions."
    )
    parser.add_argument("--agent-counts", nargs="+", type=int, default=list(DEFAULT_AGENT_COUNTS))
    parser.add_argument("--minds", nargs="+", choices=SUPPORTED_PUBLIC_GOODS_MINDS, default=list(DEFAULT_MINDS))
    parser.add_argument("--institutions", nargs="+", choices=INSTITUTIONS, default=list(DEFAULT_INSTITUTIONS))
    parser.add_argument("--steps", type=int, default=40_000)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--final-window", type=int, default=1_000)
    parser.add_argument("--benchmark-steps", type=int, default=200)
    parser.add_argument("--pool-capacity", type=float, default=20.0)
    parser.add_argument("--initial-pool", type=float, default=10.0)
    parser.add_argument("--regeneration-rate", type=float, default=0.08)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/public_goods_group_size_sweep"))
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in summary_by_seed.csv.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    outputs = run(parse_args(argv))
    for path in outputs.values():
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
