from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from worlds.pricing_arena.benchmarks import compute_static_benchmarks
from worlds.pricing_arena.training import MECHANISMS, SUPPORTED_MINDS, train_market


DEFAULT_MECHANISMS = tuple(MECHANISMS)
PLOT_METRICS = ("avg_price", "welfare", "consumer_surplus", "collusion_index")
SUMMARY_METRICS = (
    "avg_price",
    "price_dispersion",
    "profit_total",
    "reward_total",
    "consumer_surplus",
    "welfare",
    "collusion_index",
    "penalty_total",
    "quantity_total",
    "audit_rate",
    "mean_market_size",
    "nash_price_gap",
    "abs_nash_price_gap",
)

PLOT_FILENAMES = {
    "avg_price": "avg_price_ci.png",
    "welfare": "welfare_ci.png",
    "consumer_surplus": "consumer_surplus_ci.png",
    "collusion_index": "collusion_ci.png",
}


T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
    40: 2.021,
    50: 2.009,
    60: 2.000,
    80: 1.990,
    100: 1.984,
    120: 1.980,
}


def finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan")
    return float(np.mean(finite))


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def t_critical_975(df: int) -> float:
    """Two-sided 95% Student-t critical value."""
    if df < 1:
        return float("nan")
    if df in T_CRITICAL_975:
        return T_CRITICAL_975[df]
    if df > max(T_CRITICAL_975):
        return 1.96

    points = sorted(T_CRITICAL_975)
    lower = max(point for point in points if point < df)
    upper = min(point for point in points if point > df)
    lower_value = T_CRITICAL_975[lower]
    upper_value = T_CRITICAL_975[upper]
    weight = (df - lower) / (upper - lower)
    return float(lower_value + weight * (upper_value - lower_value))


def mean_ci95(values: Iterable[float]) -> dict[str, float]:
    xs = np.asarray(list(values), dtype=float)
    xs = xs[np.isfinite(xs)]
    n = int(len(xs))
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "ci95_half_width": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }

    mean = float(np.mean(xs))
    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": float("nan"),
            "sem": float("nan"),
            "ci95_half_width": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }

    std = float(np.std(xs, ddof=1))
    sem = std / math.sqrt(n)
    half_width = t_critical_975(n - 1) * sem
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_half_width": half_width,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def final_window_slice(data: dict[str, np.ndarray], final_window: int) -> tuple[slice, int, int]:
    if final_window < 1:
        raise ValueError("final_window must be positive")
    n_steps = len(next(iter(data.values())))
    if n_steps < 1:
        raise ValueError("cannot summarize an empty training run")
    effective_window = min(final_window, n_steps)
    return slice(n_steps - effective_window, n_steps), n_steps, effective_window


def summarize_final_window(
    data: dict[str, np.ndarray],
    benchmarks: dict[str, object],
    final_window: int,
) -> dict[str, float]:
    sl, _, effective_window = final_window_slice(data, final_window)
    avg_price = finite_mean(data["avg_price"][sl])
    nash_price = benchmarks["nash_price"]
    nash_gap = float("nan") if nash_price is None else avg_price - float(nash_price)

    return {
        "effective_final_window": float(effective_window),
        "nash_price": float(nash_price) if nash_price is not None else float("nan"),
        "monopoly_price": float(benchmarks["monopoly_price"]),
        "avg_price": avg_price,
        "price_dispersion": finite_mean(np.abs(data["p1"][sl] - data["p2"][sl])),
        "profit_total": finite_mean(data["profit1"][sl] + data["profit2"][sl]),
        "reward_total": finite_mean(data["reward1"][sl] + data["reward2"][sl]),
        "consumer_surplus": finite_mean(data["consumer_surplus"][sl]),
        "welfare": finite_mean(data["welfare"][sl]),
        "collusion_index": finite_mean(data["collusion_index"][sl]),
        "penalty_total": finite_mean(data["penalty1"][sl] + data["penalty2"][sl]),
        "quantity_total": finite_mean(data["quantity1"][sl] + data["quantity2"][sl]),
        "audit_rate": finite_mean(data["audit_hit"][sl]) if "audit_hit" in data else float("nan"),
        "mean_market_size": finite_mean(data["market_size"][sl]) if "market_size" in data else float("nan"),
        "nash_price_gap": nash_gap,
        "abs_nash_price_gap": abs(nash_gap) if math.isfinite(nash_gap) else float("nan"),
    }


def aggregate_rows(
    rows: list[dict[str, Any]],
    mechanisms: Iterable[str],
    metrics: Iterable[str] = SUMMARY_METRICS,
) -> list[dict[str, Any]]:
    aggregate: list[dict[str, Any]] = []
    for mechanism in mechanisms:
        mech_rows = [row for row in rows if row["mechanism"] == mechanism]
        if not mech_rows:
            continue

        out: dict[str, Any] = {
            "mechanism": mechanism,
            "n_seeds": len(mech_rows),
            "steps": mech_rows[0]["steps"],
            "final_window": mech_rows[0]["final_window"],
            "effective_final_window": mech_rows[0]["effective_final_window"],
            "nash_price": mech_rows[0]["nash_price"],
            "monopoly_price": mech_rows[0]["monopoly_price"],
        }
        for metric in metrics:
            stats = mean_ci95(row[metric] for row in mech_rows)
            out[f"{metric}_n"] = stats["n"]
            out[f"{metric}_mean"] = stats["mean"]
            out[f"{metric}_std"] = stats["std"]
            out[f"{metric}_sem"] = stats["sem"]
            out[f"{metric}_ci95_half_width"] = stats["ci95_half_width"]
            out[f"{metric}_ci95_low"] = stats["ci95_low"]
            out[f"{metric}_ci95_high"] = stats["ci95_high"]
        aggregate.append(out)
    return aggregate


def rank_mechanisms(aggregate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    welfare_sorted = sorted(
        aggregate,
        key=lambda row: (float("-inf") if not math.isfinite(row["welfare_mean"]) else row["welfare_mean"]),
        reverse=True,
    )
    price_sorted = sorted(
        aggregate,
        key=lambda row: (float("inf") if not math.isfinite(row["avg_price_mean"]) else row["avg_price_mean"]),
    )
    collusion_sorted = sorted(
        aggregate,
        key=lambda row: (
            float("inf") if not math.isfinite(row["collusion_index_mean"]) else row["collusion_index_mean"]
        ),
    )

    rank_by_mechanism: dict[str, dict[str, Any]] = {
        row["mechanism"]: {"mechanism": row["mechanism"]} for row in aggregate
    }
    for rank, row in enumerate(welfare_sorted, start=1):
        rank_by_mechanism[row["mechanism"]]["welfare_rank_desc"] = rank
    for rank, row in enumerate(price_sorted, start=1):
        rank_by_mechanism[row["mechanism"]]["avg_price_rank_asc"] = rank
    for rank, row in enumerate(collusion_sorted, start=1):
        rank_by_mechanism[row["mechanism"]]["collusion_rank_asc"] = rank

    for row in aggregate:
        target = rank_by_mechanism[row["mechanism"]]
        target["welfare_mean"] = row["welfare_mean"]
        target["welfare_ci95_low"] = row["welfare_ci95_low"]
        target["welfare_ci95_high"] = row["welfare_ci95_high"]
        target["avg_price_mean"] = row["avg_price_mean"]
        target["collusion_index_mean"] = row["collusion_index_mean"]
    return [rank_by_mechanism[row["mechanism"]] for row in aggregate]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_fieldnames(metrics: Iterable[str] = SUMMARY_METRICS) -> list[str]:
    fields = [
        "mechanism",
        "n_seeds",
        "steps",
        "final_window",
        "effective_final_window",
        "nash_price",
        "monopoly_price",
    ]
    for metric in metrics:
        fields.extend(
            [
                f"{metric}_n",
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_sem",
                f"{metric}_ci95_half_width",
                f"{metric}_ci95_low",
                f"{metric}_ci95_high",
            ]
        )
    return fields


def by_seed_fieldnames(metrics: Iterable[str] = SUMMARY_METRICS) -> list[str]:
    return [
        "mechanism",
        "seed_index",
        "seed",
        "steps",
        "final_window",
        "effective_final_window",
        "nash_price",
        "monopoly_price",
        *metrics,
    ]


def matrix_mean_ci95(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0, ddof=1) if matrix.shape[0] > 1 else np.full(matrix.shape[1], np.nan)

    if matrix.shape[0] < 2:
        return mean, np.full_like(mean, np.nan)

    sem = std / math.sqrt(matrix.shape[0])
    half_width = t_critical_975(matrix.shape[0] - 1) * sem
    return mean, half_width


def plot_metric_curves(
    series_by_mechanism: dict[str, list[np.ndarray]],
    metric: str,
    benchmarks: dict[str, object],
    save_dir: Path,
    smoothing: int,
    show: bool = False,
) -> list[Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(save_dir / ".matplotlib"))

    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for mechanism, seed_series in series_by_mechanism.items():
        if not seed_series:
            continue

        smoothed = [rolling_mean(np.asarray(series, dtype=float), smoothing) for series in seed_series]
        min_len = min(len(series) for series in smoothed)
        matrix = np.vstack([series[:min_len] for series in smoothed])
        mean, half_width = matrix_mean_ci95(matrix)
        step_offset = smoothing - 1 if smoothing > 1 and len(seed_series[0]) >= smoothing else 0
        x_axis = np.arange(min_len) + step_offset

        ax.plot(x_axis, mean, linewidth=1.8, label=mechanism)
        if np.any(np.isfinite(half_width)):
            ax.fill_between(x_axis, mean - half_width, mean + half_width, alpha=0.16, linewidth=0)

    if metric == "avg_price":
        nash_price = benchmarks["nash_price"]
        if nash_price is not None:
            ax.axhline(float(nash_price), color="black", linestyle="--", linewidth=1.0, alpha=0.45, label="Nash")
        ax.axhline(
            float(benchmarks["monopoly_price"]),
            color="black",
            linestyle=":",
            linewidth=1.0,
            alpha=0.45,
            label="Joint-profit",
        )
    elif metric == "collusion_index":
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.35)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0, alpha=0.35)

    ax.set_title(f"{metric} across seeds: mean with 95% Student-t CI")
    ax.set_xlabel("Training step")
    ax.set_ylabel(metric)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    filename = PLOT_FILENAMES.get(metric, f"{metric}_ci.png")
    output_path = save_dir / filename
    fig.savefig(output_path, dpi=180)
    saved = [output_path]

    if metric == "collusion_index":
        compatibility_path = save_dir / "collusion_index_ci.png"
        fig.savefig(compatibility_path, dpi=180)
        saved.append(compatibility_path)

    if show:
        plt.show()
    plt.close(fig)
    return saved


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    benchmarks: dict[str, object],
    elapsed_seconds: float,
    outputs: list[Path],
) -> None:
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": serializable_args,
        "mechanisms": list(args.mechanisms),
        "summary_metrics": list(SUMMARY_METRICS),
        "plot_metrics": list(args.plot_metrics),
        "ci_method": "Student-t two-sided 95% confidence interval over seed-level means",
        "seed_design": "paired seeds across mechanisms: seed = seed_start + seed_index * seed_stride",
        "benchmarks": {
            "nash_pairs": benchmarks["nash_pairs"],
            "nash_price": benchmarks["nash_price"],
            "monopoly_price": benchmarks["monopoly_price"],
        },
        "outputs": [str(path) for path in outputs],
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def validate_args(args: argparse.Namespace) -> None:
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.n_seeds < 1:
        raise ValueError("--n-seeds must be positive")
    if args.seed_stride < 1:
        raise ValueError("--seed-stride must be positive")
    if args.final_window < 1:
        raise ValueError("--final-window must be positive")
    if args.smoothing < 1:
        raise ValueError("--smoothing must be positive")
    unknown = sorted(set(args.mechanisms) - set(MECHANISMS))
    if unknown:
        valid = ", ".join(MECHANISMS)
        raise ValueError(f"Unknown mechanism(s): {', '.join(unknown)}. Valid choices: {valid}")


def run_experiment(args: argparse.Namespace) -> dict[str, Path]:
    validate_args(args)
    start_time = time.time()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    series: dict[str, dict[str, list[np.ndarray]]] = {
        mechanism: {metric: [] for metric in args.plot_metrics}
        for mechanism in args.mechanisms
    }
    final_benchmarks = compute_static_benchmarks(np.linspace(1.0, 10.0, 19))

    for mechanism in args.mechanisms:
        print(f"\n=== Mechanism: {mechanism} ===", flush=True)
        for seed_index in range(args.n_seeds):
            seed = args.seed_start + seed_index * args.seed_stride
            seed_start_time = time.time()
            print(f"seed_index={seed_index:03d} seed={seed}", flush=True)
            data, benchmarks = train_market(mechanism=mechanism, steps=args.steps, seed=seed, mind=args.mind)
            final_benchmarks = benchmarks
            summary = summarize_final_window(data, benchmarks, args.final_window)

            row = {
                "mechanism": mechanism,
                "seed_index": seed_index,
                "seed": seed,
                "steps": args.steps,
                "final_window": args.final_window,
                **summary,
            }
            summary_rows.append(row)

            for metric in args.plot_metrics:
                series[mechanism][metric].append(np.asarray(data[metric], dtype=float))

            elapsed_seed = time.time() - seed_start_time
            mechanism_rows = [item for item in summary_rows if item["mechanism"] == mechanism]
            running_avg_price = finite_mean([item["avg_price"] for item in mechanism_rows])
            print(
                f"  elapsed_seed={elapsed_seed:.2f}s "
                f"avg_price={summary['avg_price']:.3f} "
                f"welfare={summary['welfare']:.3f} "
                f"consumer_surplus={summary['consumer_surplus']:.3f} "
                f"collusion={summary['collusion_index']:.3f} "
                f"nash_gap={summary['nash_price_gap']:.3f} "
                f"running_avg_price={running_avg_price:.3f}",
                flush=True,
            )

    by_seed_path = save_dir / "summary_by_seed.csv"
    write_csv(by_seed_path, summary_rows, by_seed_fieldnames())

    aggregate = aggregate_rows(summary_rows, args.mechanisms)
    aggregate_path = save_dir / "summary_aggregate.csv"
    write_csv(aggregate_path, aggregate, aggregate_fieldnames())

    rankings = rank_mechanisms(aggregate)
    rankings_path = save_dir / "mechanism_rankings.csv"
    write_csv(
        rankings_path,
        rankings,
        [
            "mechanism",
            "welfare_rank_desc",
            "avg_price_rank_asc",
            "collusion_rank_asc",
            "welfare_mean",
            "welfare_ci95_low",
            "welfare_ci95_high",
            "avg_price_mean",
            "collusion_index_mean",
        ],
    )

    outputs = [by_seed_path, aggregate_path, rankings_path]
    if not args.no_plots:
        for metric in args.plot_metrics:
            metric_series = {
                mechanism: series[mechanism][metric]
                for mechanism in args.mechanisms
            }
            outputs.extend(
                plot_metric_curves(
                    metric_series,
                    metric,
                    final_benchmarks,
                    save_dir,
                    args.smoothing,
                    args.show,
                )
            )

    manifest_path = save_dir / "experiment_manifest.json"
    elapsed_seconds = time.time() - start_time
    write_manifest(manifest_path, args, final_benchmarks, elapsed_seconds, outputs)
    outputs.append(manifest_path)

    print()
    for path in outputs:
        print(f"Wrote: {path}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")

    return {
        "summary_by_seed": by_seed_path,
        "summary_aggregate": aggregate_path,
        "mechanism_rankings": rankings_path,
        "manifest": manifest_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run replicated duopoly-learning experiments across paired random seeds."
    )
    parser.add_argument("--steps", type=int, default=40_000, help="Training steps per mechanism and seed.")
    parser.add_argument("--n-seeds", type=int, default=20, help="Number of paired seeds per mechanism.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed in the paired seed schedule.")
    parser.add_argument("--seed-stride", type=int, default=1, help="Stride between paired seeds.")
    parser.add_argument("--final-window", type=int, default=1_000, help="Tail window used for per-seed summaries.")
    parser.add_argument("--smoothing", type=int, default=500, help="Rolling mean window for CI plots.")
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/full_v0_multiseed"))
    parser.add_argument("--mechanisms", nargs="+", choices=MECHANISMS, default=list(DEFAULT_MECHANISMS))
    parser.add_argument("--mind", choices=SUPPORTED_MINDS, default="q_learning")
    parser.add_argument("--plot-metrics", nargs="+", choices=PLOT_METRICS, default=list(PLOT_METRICS))
    parser.add_argument("--no-plots", action="store_true", help="Write CSV/JSON outputs without generating PNGs.")
    parser.add_argument("--show", action="store_true", help="Show Matplotlib windows after saving plots.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
