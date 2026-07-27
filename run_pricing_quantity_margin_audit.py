from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from run_multiseed import mean_ci95, summarize_final_window
from worlds.pricing_arena.training import train_market_with_agents


FULL_RESULT_DIRS = {
    "q_learning": "full_v0_multiseed",
    "random": "random_v0_multiseed",
    "dqn": "dqn_v0_multiseed",
    "ppo": "ppo_v0_multiseed",
    "independent_dqn": "independent_dqn_v0_multiseed_fixed",
    "centralized_critic": "centralized_critic_v0_multiseed",
}

DEFAULT_MINDS = ("q_learning", "dqn", "ppo", "independent_dqn", "centralized_critic")
DELTA_METRICS = ("avg_price", "quantity_total", "profit_total", "welfare", "collusion_index")
MECHANISMS = ("none", "price_cap")

SEED_DELTA_FIELDS = (
    "source",
    "mind",
    "budget_steps",
    "seed_index",
    "seed",
    "final_window",
    "none_avg_price",
    "price_cap_avg_price",
    "delta_avg_price",
    "none_quantity_total",
    "price_cap_quantity_total",
    "delta_quantity_total",
    "none_profit_total",
    "price_cap_profit_total",
    "delta_profit_total",
    "none_welfare",
    "price_cap_welfare",
    "delta_welfare",
    "none_collusion_index",
    "price_cap_collusion_index",
    "delta_collusion_index",
    "price_cap_profit_higher",
)

AGGREGATE_FIELDS = (
    "source",
    "mind",
    "budget_steps",
    "n",
    "positive_profit_delta_count",
    "positive_profit_delta_share",
    "delta_profit_total_mean",
    "delta_profit_total_median",
    "delta_profit_total_std",
    "delta_profit_total_ci95_low",
    "delta_profit_total_ci95_high",
    "delta_quantity_total_mean",
    "delta_avg_price_mean",
    "delta_welfare_mean",
    "delta_collusion_index_mean",
    "min_delta_profit_total",
    "max_delta_profit_total",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _paired_delta_row(
    *,
    source: str,
    mind: str,
    budget_steps: int,
    seed_index: int,
    seed: int,
    final_window: int,
    none: dict[str, Any],
    price_cap: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "source": source,
        "mind": mind,
        "budget_steps": budget_steps,
        "seed_index": seed_index,
        "seed": seed,
        "final_window": final_window,
    }
    for metric in DELTA_METRICS:
        none_value = _float(none, metric)
        cap_value = _float(price_cap, metric)
        out[f"none_{metric}"] = none_value
        out[f"price_cap_{metric}"] = cap_value
        out[f"delta_{metric}"] = cap_value - none_value
    out["price_cap_profit_higher"] = float(out["delta_profit_total"] > 0.0)
    return out


def existing_full_seed_deltas(outputs_dir: Path, minds: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mind in minds:
        alias = FULL_RESULT_DIRS.get(mind)
        if alias is None:
            continue
        path = outputs_dir / alias / "summary_by_seed.csv"
        if not path.exists():
            continue
        source_rows = _read_csv(path)
        by_seed_mech = {
            (int(float(row["seed"])), row["mechanism"]): row
            for row in source_rows
            if row.get("mechanism") in MECHANISMS
        }
        seeds = sorted({seed for seed, mechanism in by_seed_mech if mechanism == "none"})
        for seed_index, seed in enumerate(seeds):
            none = by_seed_mech.get((seed, "none"))
            cap = by_seed_mech.get((seed, "price_cap"))
            if none is None or cap is None:
                continue
            rows.append(
                _paired_delta_row(
                    source="existing_full_n20",
                    mind=mind,
                    budget_steps=int(float(none.get("steps", 0))),
                    seed_index=seed_index,
                    seed=seed,
                    final_window=int(float(none.get("final_window", 0))),
                    none=none,
                    price_cap=cap,
                )
            )
    return rows


def _fresh_summary(mind: str, mechanism: str, steps: int, seed: int, final_window: int) -> dict[str, float]:
    result = train_market_with_agents(
        mechanism=mechanism,
        steps=steps,
        seed=seed,
        mind=mind,
    )
    return summarize_final_window(result.data, result.benchmarks, final_window=min(final_window, steps))


def fresh_seed_deltas(
    *,
    minds: list[str],
    budgets: list[int],
    n_seeds: int,
    seed_start: int,
    seed_stride: int,
    final_window: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, int, int], dict[str, dict[str, float]]] = {}
    for mind in minds:
        for steps in budgets:
            print(f"=== fresh Pricing audit mind={mind} steps={steps} ===", flush=True)
            for seed_index in range(n_seeds):
                seed = seed_start + seed_index * seed_stride
                key = (mind, steps, seed)
                if key not in cache:
                    seed_start_time = time.time()
                    none = _fresh_summary(mind, "none", steps, seed, final_window)
                    cap = _fresh_summary(mind, "price_cap", steps, seed, final_window)
                    cache[key] = {"none": none, "price_cap": cap}
                    print(
                        f"seed_index={seed_index:03d} seed={seed} "
                        f"delta_profit={cap['profit_total'] - none['profit_total']:.6g} "
                        f"delta_qty={cap['quantity_total'] - none['quantity_total']:.6g} "
                        f"elapsed={time.time() - seed_start_time:.2f}s",
                        flush=True,
                    )
                rows.append(
                    _paired_delta_row(
                        source="fresh_local",
                        mind=mind,
                        budget_steps=steps,
                        seed_index=seed_index,
                        seed=seed,
                        final_window=min(final_window, steps),
                        none=cache[key]["none"],
                        price_cap=cache[key]["price_cap"],
                    )
                )
    return rows


def aggregate_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups = sorted({(row["source"], row["mind"], int(row["budget_steps"])) for row in rows})
    for source, mind, budget_steps in groups:
        subset = [
            row
            for row in rows
            if row["source"] == source and row["mind"] == mind and int(row["budget_steps"]) == budget_steps
        ]
        deltas = [float(row["delta_profit_total"]) for row in subset if _finite(float(row["delta_profit_total"]))]
        stats = mean_ci95(deltas)
        out.append(
            {
                "source": source,
                "mind": mind,
                "budget_steps": budget_steps,
                "n": len(deltas),
                "positive_profit_delta_count": int(sum(value > 0.0 for value in deltas)),
                "positive_profit_delta_share": float(sum(value > 0.0 for value in deltas) / len(deltas))
                if deltas
                else float("nan"),
                "delta_profit_total_mean": stats["mean"],
                "delta_profit_total_median": float(np.median(deltas)) if deltas else float("nan"),
                "delta_profit_total_std": stats["std"],
                "delta_profit_total_ci95_low": stats["ci95_low"],
                "delta_profit_total_ci95_high": stats["ci95_high"],
                "delta_quantity_total_mean": mean_ci95(float(row["delta_quantity_total"]) for row in subset)["mean"],
                "delta_avg_price_mean": mean_ci95(float(row["delta_avg_price"]) for row in subset)["mean"],
                "delta_welfare_mean": mean_ci95(float(row["delta_welfare"]) for row in subset)["mean"],
                "delta_collusion_index_mean": mean_ci95(float(row["delta_collusion_index"]) for row in subset)["mean"],
                "min_delta_profit_total": min(deltas) if deltas else float("nan"),
                "max_delta_profit_total": max(deltas) if deltas else float("nan"),
            }
        )
    return out


def budget_ladder_rows(all_seed_rows: list[dict[str, Any]], mind: str) -> list[dict[str, Any]]:
    rows = [row for row in aggregate_delta_rows(all_seed_rows) if row["mind"] == mind]
    source_priority = {"fresh_local": 0, "existing_full_n20": 1}
    return sorted(rows, key=lambda row: (int(row["budget_steps"]), source_priority.get(str(row["source"]), 99)))


def _format_float(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def write_diagnosis(
    path: Path,
    *,
    existing_aggregate: list[dict[str, Any]],
    fresh_aggregate: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    audited_mind: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pricing Arena Quantity-Margin Audit",
        "",
        "Generated by `run_pricing_quantity_margin_audit.py`.",
        "",
        "Question: why did the 1000-step mechanism trace show profit falling under price cap while the full n=20 table shows a DQN profit increase under price cap?",
        "",
        "This file separates three explanations: training-budget mismatch, seed variance, and full-run artifact/outlier sensitivity.",
        "",
        "## Existing Full N=20 Paired Seed Audit",
        "",
    ]
    for row in sorted(existing_aggregate, key=lambda item: item["mind"]):
        lines.append(
            "- "
            f"`{row['mind']}`: delta_profit_mean=`{_format_float(row['delta_profit_total_mean'])}`, "
            f"median=`{_format_float(row['delta_profit_total_median'])}`, "
            f"95% CI=[`{_format_float(row['delta_profit_total_ci95_low'])}`, `{_format_float(row['delta_profit_total_ci95_high'])}`], "
            f"positive_seed_share=`{_format_float(row['positive_profit_delta_share'])}`, "
            f"delta_quantity_mean=`{_format_float(row['delta_quantity_total_mean'])}`, "
            f"delta_price_mean=`{_format_float(row['delta_avg_price_mean'])}`."
        )
    lines.extend(["", f"## `{audited_mind}` Budget Ladder", ""])
    for row in budget_rows:
        lines.append(
            "- "
            f"{row['source']} steps=`{row['budget_steps']}`: "
            f"delta_profit_mean=`{_format_float(row['delta_profit_total_mean'])}`, "
            f"median=`{_format_float(row['delta_profit_total_median'])}`, "
            f"positive_seed_share=`{_format_float(row['positive_profit_delta_share'])}`, "
            f"delta_quantity_mean=`{_format_float(row['delta_quantity_total_mean'])}`."
        )

    full = next(
        (
            row
            for row in existing_aggregate
            if row["mind"] == audited_mind and int(row["budget_steps"]) == max(int(item["budget_steps"]) for item in existing_aggregate if item["mind"] == audited_mind)
        ),
        None,
    )
    early = next(
        (
            row
            for row in fresh_aggregate
            if row["mind"] == audited_mind and int(row["budget_steps"]) == min(int(item["budget_steps"]) for item in fresh_aggregate if item["mind"] == audited_mind)
        ),
        None,
    )
    lines.extend(["", "## Diagnosis", ""])
    if full is None or early is None:
        lines.append("- Diagnosis incomplete: missing early fresh run or existing full run for the audited mind.")
    else:
        early_delta = float(early["delta_profit_total_mean"])
        full_delta = float(full["delta_profit_total_mean"])
        early_share = float(early["positive_profit_delta_share"])
        full_share = float(full["positive_profit_delta_share"])
        if early_delta < 0.0 < full_delta:
            lines.append(
                "- The contradiction is real at the aggregate level, but it is consistent with a training-budget effect: "
                f"`{audited_mind}` is negative at the short fresh budget and positive in the existing full 40k-step paired audit."
            )
        elif full_delta > 0.0 and full_share < 0.5:
            lines.append(
                "- The full-run mean is positive while fewer than half of full-run seeds are positive; treat the headline as outlier-sensitive until inspected."
            )
        elif full_delta > 0.0 and full_share >= 0.5:
            lines.append(
                "- The full-run profit increase is not just one outlier: at least half of paired full-run seeds show positive cap-minus-none profit."
            )
        else:
            lines.append(
                "- The full-run paired audit does not support a positive profit-under-cap finding for the audited mind; revise the headline claim."
            )
        if 0.2 < early_share < 0.8:
            lines.append(
                "- Short-budget seed variance is substantial: positive and negative short-run outcomes both occur, so single-seed traces should never be used as sign evidence."
            )
        else:
            lines.append(
                "- Short-budget seed variance is not balanced around the sign; the 1000-step trace direction is fairly consistent at the tested short budget."
            )
    lines.extend(
        [
            "",
            "## Safe Claim Boundary",
            "",
            "- Use `mechanism_traces_local_full/` as channel evidence only.",
            "- Use this audit's paired full-run seed table for sign claims about DQN profit under price cap.",
            "- If the budget ladder shows a sign flip, write the result as an emergent long-training behavior, not as something visible in short traces.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Path]:
    start = time.time()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    existing_rows = existing_full_seed_deltas(args.outputs_dir, args.minds)
    existing_aggregate = aggregate_delta_rows(existing_rows)

    fresh_minds = sorted(set(args.fresh_minds + [args.audited_mind]))
    fresh_rows = fresh_seed_deltas(
        minds=fresh_minds,
        budgets=sorted(set(args.fresh_steps)),
        n_seeds=args.n_seeds,
        seed_start=args.seed_start,
        seed_stride=args.seed_stride,
        final_window=args.final_window,
    )
    audited_extra_budgets = sorted(set(args.budget_steps) - set(args.fresh_steps))
    if audited_extra_budgets:
        fresh_rows.extend(
            fresh_seed_deltas(
                minds=[args.audited_mind],
                budgets=audited_extra_budgets,
                n_seeds=args.n_seeds,
                seed_start=args.seed_start,
                seed_stride=args.seed_stride,
                final_window=args.final_window,
            )
        )
    fresh_aggregate = aggregate_delta_rows(fresh_rows)
    all_rows = existing_rows + fresh_rows
    budget_rows = budget_ladder_rows(all_rows, args.audited_mind)

    existing_seed_path = args.save_dir / "existing_full_seed_deltas.csv"
    existing_aggregate_path = args.save_dir / "existing_full_aggregate_deltas.csv"
    fresh_seed_path = args.save_dir / "fresh_seed_deltas.csv"
    fresh_aggregate_path = args.save_dir / "fresh_aggregate_deltas.csv"
    budget_path = args.save_dir / f"{args.audited_mind}_budget_ladder.csv"
    diagnosis_path = args.save_dir / "diagnosis.md"
    manifest_path = args.save_dir / "manifest.json"

    _write_csv(existing_seed_path, existing_rows, SEED_DELTA_FIELDS)
    _write_csv(existing_aggregate_path, existing_aggregate, AGGREGATE_FIELDS)
    _write_csv(fresh_seed_path, fresh_rows, SEED_DELTA_FIELDS)
    _write_csv(fresh_aggregate_path, fresh_aggregate, AGGREGATE_FIELDS)
    _write_csv(budget_path, budget_rows, AGGREGATE_FIELDS)
    write_diagnosis(
        diagnosis_path,
        existing_aggregate=existing_aggregate,
        fresh_aggregate=fresh_aggregate,
        budget_rows=budget_rows,
        audited_mind=args.audited_mind,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.time() - start,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "config": {
                    **{key: value for key, value in vars(args).items() if key not in {"save_dir", "outputs_dir"}},
                    "save_dir": str(args.save_dir),
                    "outputs_dir": str(args.outputs_dir),
                },
                "existing_seed_delta_rows": len(existing_rows),
                "fresh_seed_delta_rows": len(fresh_rows),
                "outputs": [
                    str(existing_seed_path),
                    str(existing_aggregate_path),
                    str(fresh_seed_path),
                    str(fresh_aggregate_path),
                    str(budget_path),
                    str(diagnosis_path),
                ],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    return {
        "existing_seed_deltas": existing_seed_path,
        "existing_aggregate_deltas": existing_aggregate_path,
        "fresh_seed_deltas": fresh_seed_path,
        "fresh_aggregate_deltas": fresh_aggregate_path,
        "budget_ladder": budget_path,
        "diagnosis": diagnosis_path,
        "manifest": manifest_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the Pricing Arena price-cap quantity-margin contradiction.")
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/pricing_quantity_margin_audit"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--minds", nargs="+", default=list(DEFAULT_MINDS), choices=tuple(FULL_RESULT_DIRS))
    parser.add_argument("--fresh-minds", nargs="+", default=list(DEFAULT_MINDS), choices=tuple(FULL_RESULT_DIRS))
    parser.add_argument("--audited-mind", default="dqn", choices=tuple(FULL_RESULT_DIRS))
    parser.add_argument("--fresh-steps", nargs="+", type=int, default=[1000])
    parser.add_argument("--budget-steps", nargs="+", type=int, default=[1000, 5000, 10000])
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--final-window", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.n_seeds < 1:
        parser.error("--n-seeds must be positive")
    if any(step < 1 for step in args.fresh_steps + args.budget_steps):
        parser.error("step budgets must be positive")
    if args.final_window < 1:
        parser.error("--final-window must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    outputs = run(parse_args(argv))
    for path in outputs.values():
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
