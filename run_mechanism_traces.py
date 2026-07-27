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
from typing import Any, Callable

import numpy as np

from core.metrics import finite_mean
from run_auction_house_smoke import scenario_config, scenario_institution
from run_resource_island_smoke import (
    activation_initial_inventory,
    activation_start_positions,
    specialization_preferences,
)
from worlds.auction_house.training import benchmark_for_config as auction_benchmark_for_config
from worlds.auction_house.training import summarize_records as summarize_auction_records
from worlds.auction_house.training import train_auction_house
from worlds.labor_market.env import LaborMarketConfig
from worlds.labor_market.training import benchmark_for_config as labor_benchmark_for_config
from worlds.labor_market.training import summarize_records as summarize_labor_records
from worlds.labor_market.training import train_labor_market
from worlds.pricing_arena.training import train_market_with_agents
from worlds.public_goods.env import PublicGoodsConfig
from worlds.public_goods.training import benchmark_for_config as public_goods_benchmark_for_config
from worlds.public_goods.training import summarize_records as summarize_public_goods_records
from worlds.public_goods.training import train_public_goods
from worlds.resource_island.benchmarks import efficient_gather_upper_bound
from worlds.resource_island.env import ResourceIslandConfig
from worlds.resource_island.training import summarize_records as summarize_resource_island_records
from worlds.resource_island.training import train_resource_island


DEFAULT_WORLDS = ("pricing_arena", "auction_house", "public_goods", "labor_market", "resource_island")
DEFAULT_MINDS = ("q_learning", "dqn", "ppo", "independent_dqn", "centralized_critic")

PRICING_MECHANISMS = ("none", "price_cap")
AUCTION_SCENARIOS = ("second_price", "first_price")
PUBLIC_GOODS_INSTITUTIONS = ("none", "contribution_matching", "public_goods_penalty")
RESOURCE_ISLAND_INSTITUTIONS = ("none", "property_rights", "trade_price_controls", "reputation_system")

TRACE_FIELDS = (
    "world",
    "mind",
    "institution",
    "seed",
    "n_agents",
    "step",
    "metric_payload",
)

DECOMPOSITION_FIELDS = (
    "world",
    "mind",
    "institution",
    "seed",
    "n_agents",
    "trace_family",
    "component",
    "value",
    "reference",
    "gap",
    "interpretation",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean(records: list[dict[str, Any]], key: str, final_window: int) -> float:
    window = records[-min(len(records), final_window) :]
    return finite_mean(float(record[key]) for record in window if _finite(record.get(key)))


def _array_mean(data: dict[str, np.ndarray], key: str, final_window: int) -> float:
    values = np.asarray(data[key], dtype=float)
    window = values[-min(len(values), final_window) :]
    return finite_mean(window)


def _decomp_row(
    *,
    world: str,
    mind: str,
    institution: str,
    seed: int,
    n_agents: int,
    trace_family: str,
    component: str,
    value: float,
    reference: float | str = "",
    interpretation: str,
) -> dict[str, Any]:
    gap = ""
    if _finite(value) and _finite(reference):
        gap = float(value) - float(reference)
    return {
        "world": world,
        "mind": mind,
        "institution": institution,
        "seed": seed,
        "n_agents": n_agents,
        "trace_family": trace_family,
        "component": component,
        "value": value,
        "reference": reference,
        "gap": gap,
        "interpretation": interpretation,
    }


def _records_from_arrays(data: dict[str, np.ndarray], max_rows: int | None = None) -> list[dict[str, Any]]:
    length = len(next(iter(data.values())))
    indices = range(length) if max_rows is None else range(max(0, length - max_rows), length)
    rows: list[dict[str, Any]] = []
    for index in indices:
        rows.append({key: _jsonable(value[index]) for key, value in data.items()})
    return rows


def _trace_rows(
    *,
    world: str,
    mind: str,
    institution: str,
    seed: int,
    n_agents: int,
    records: list[dict[str, Any]],
    max_rows: int | None,
) -> list[dict[str, Any]]:
    if max_rows is not None:
        records = records[-min(len(records), max_rows) :]
    rows: list[dict[str, Any]] = []
    for fallback_step, record in enumerate(records):
        step = record.get("step", record.get("train_step", record.get("round", fallback_step)))
        rows.append(
            {
                "world": world,
                "mind": mind,
                "institution": institution,
                "seed": seed,
                "n_agents": n_agents,
                "step": int(float(step)) if _finite(step) else fallback_step,
                "metric_payload": json.dumps(_jsonable(record), sort_keys=True),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _pricing_decomposition(
    *,
    mind: str,
    seed: int,
    n_agents: int,
    mechanism: str,
    data: dict[str, np.ndarray],
    benchmarks: dict[str, Any],
    final_window: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    avg_price = _array_mean(data, "avg_price", final_window)
    quantity = _array_mean(data, "quantity_total", final_window)
    profit = _array_mean(data, "profit_total", final_window)
    welfare = _array_mean(data, "welfare", final_window)
    reward = _array_mean(data, "reward_total", final_window)
    nash_price = benchmarks.get("nash_price")
    monopoly_price = float(benchmarks["monopoly_price"])
    binding_values = []
    for firm_index in range(1, n_agents + 1):
        raw_key = f"raw_p{firm_index}"
        price_key = f"p{firm_index}"
        if raw_key in data and price_key in data:
            binding_values.extend((np.asarray(data[raw_key]) > np.asarray(data[price_key]) + 1e-12).astype(float))
    cap_binding = finite_mean(binding_values) if binding_values else float("nan")
    components = [
        ("avg_price", avg_price, nash_price, "price level relative to the static Nash reference"),
        ("quantity_total", quantity, "", "quantity channel that can offset lower regulated prices"),
        ("profit_total", profit, benchmarks.get("nash_profit", ""), "producer-profit channel"),
        ("reward_total", reward, "", "post-institution agent reward after penalties/taxes"),
        ("welfare", welfare, "", "consumer surplus plus producer profit in the implemented proxy"),
        ("price_cap_binding_rate", cap_binding, 0.0, "fraction of firm-steps where raw price exceeded realized price"),
        ("monopoly_price_reference", monopoly_price, "", "symmetric joint-profit reference price"),
    ]
    for component, value, reference, interpretation in components:
        rows.append(
            _decomp_row(
                world="pricing_arena",
                mind=mind,
                institution=mechanism,
                seed=seed,
                n_agents=n_agents,
                trace_family="price_margin_quantity",
                component=component,
                value=float(value) if _finite(value) else float("nan"),
                reference=float(reference) if _finite(reference) else reference,
                interpretation=interpretation,
            )
        )
    return rows


def run_pricing(args: argparse.Namespace, mind: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace_rows: list[dict[str, Any]] = []
    decomp_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, float]] = {}
    for mechanism in args.pricing_mechanisms:
        result = train_market_with_agents(
            mechanism=mechanism,
            steps=args.steps,
            seed=args.seed,
            mind=mind,
            n_firms=args.n_agents,
        )
        records = _records_from_arrays(result.data, max_rows=args.max_trace_rows)
        trace_rows.extend(
            _trace_rows(
                world="pricing_arena",
                mind=mind,
                institution=mechanism,
                seed=args.seed,
                n_agents=args.n_agents,
                records=records,
                max_rows=None,
            )
        )
        decomp_rows.extend(
            _pricing_decomposition(
                mind=mind,
                seed=args.seed,
                n_agents=args.n_agents,
                mechanism=mechanism,
                data=result.data,
                benchmarks=result.benchmarks,
                final_window=args.final_window,
            )
        )
        summaries[mechanism] = {
            "avg_price": _array_mean(result.data, "avg_price", args.final_window),
            "quantity_total": _array_mean(result.data, "quantity_total", args.final_window),
            "profit_total": _array_mean(result.data, "profit_total", args.final_window),
            "welfare": _array_mean(result.data, "welfare", args.final_window),
        }
    if "none" in summaries and "price_cap" in summaries:
        base = summaries["none"]
        cap = summaries["price_cap"]
        for component in ("avg_price", "quantity_total", "profit_total", "welfare"):
            decomp_rows.append(
                _decomp_row(
                    world="pricing_arena",
                    mind=mind,
                    institution="price_cap_vs_none",
                    seed=args.seed,
                    n_agents=args.n_agents,
                    trace_family="paired_quantity_margin_trace",
                    component=f"delta_{component}",
                    value=float(cap[component] - base[component]),
                    reference=0.0,
                    interpretation=(
                        "paired final-window difference; positive quantity/profit under lower capped prices "
                        "is the quantity-margin channel to inspect"
                    ),
                )
            )
    return trace_rows, decomp_rows


def run_auction(args: argparse.Namespace, mind: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace_rows: list[dict[str, Any]] = []
    decomp_rows: list[dict[str, Any]] = []
    for scenario in args.auction_scenarios:
        config = scenario_config(scenario, steps=args.steps, n_bidders=args.n_agents)
        institution, institution_params = scenario_institution(scenario, args.seed)
        result = train_auction_house(
            steps=args.steps,
            seed=args.seed,
            config=config,
            institution=institution,
            institution_params=institution_params,
            mind=mind,
        )
        trace_rows.extend(
            _trace_rows(
                world="auction_house",
                mind=mind,
                institution=scenario,
                seed=args.seed,
                n_agents=args.n_agents,
                records=result.records,
                max_rows=args.max_trace_rows,
            )
        )
        summary = summarize_auction_records(result.records, args.final_window)
        benchmark = auction_benchmark_for_config(config)
        components = {
            "revenue": ("benchmark revenue under the theory/reference bid strategy", "seller revenue"),
            "welfare": ("benchmark welfare under efficient/reference behavior", "realized total welfare"),
            "allocative_efficiency": ("1.0 for fully efficient allocation", "whether the highest-value bidder wins"),
            "ex_post_regret_mean": ("0.0", "incentive-compatibility / profitable misreport proxy"),
            "truthful_bid_distance_mean": ("0.0", "distance from truthful bidding"),
            "first_price_shading_distance_mean": (
                "0.0",
                "distance from symmetric first-price bid-shading reference",
            ),
            "overbid_rate": ("0.0", "rate of bids above private value"),
            "underbid_rate": ("", "rate of bid shading / below-value bids"),
        }
        for component, (reference_label, interpretation) in components.items():
            reference: float | str
            benchmark_key = component
            if benchmark_key in benchmark:
                reference = float(benchmark[benchmark_key])
            elif reference_label == "0.0":
                reference = 0.0
            elif reference_label.startswith("1.0"):
                reference = 1.0
            else:
                reference = ""
            decomp_rows.append(
                _decomp_row(
                    world="auction_house",
                    mind=mind,
                    institution=scenario,
                    seed=args.seed,
                    n_agents=args.n_agents,
                    trace_family="bid_value_regret",
                    component=component,
                    value=float(summary[component]),
                    reference=reference,
                    interpretation=interpretation,
                )
            )
    return trace_rows, decomp_rows


def run_public_goods(args: argparse.Namespace, mind: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace_rows: list[dict[str, Any]] = []
    decomp_rows: list[dict[str, Any]] = []
    config = PublicGoodsConfig(
        n_agents=args.n_agents,
        max_rounds=args.steps,
        pool_capacity=args.public_goods_pool_capacity,
        initial_pool=args.public_goods_initial_pool,
        regeneration_rate=args.public_goods_regeneration_rate,
    )
    benchmark = public_goods_benchmark_for_config(config, steps=min(args.steps, 200))
    for institution in args.public_goods_institutions:
        result = train_public_goods(
            steps=args.steps,
            seed=args.seed,
            institution=institution,
            config=config,
            mind=mind,
        )
        trace_rows.extend(
            _trace_rows(
                world="public_goods",
                mind=mind,
                institution=institution,
                seed=args.seed,
                n_agents=args.n_agents,
                records=result.records,
                max_rows=args.max_trace_rows,
            )
        )
        summary = summarize_public_goods_records(result.records, args.final_window)
        components = [
            (
                "contribution_total",
                "",
                "agent contributions; near-zero baseline is the free-rider sanity signal",
            ),
            ("extraction_total", "", "realized extraction pressure on the public pool"),
            ("extraction_minus_contribution", 0.0, "positive values indicate net depletion pressure before regeneration"),
            (
                "sustainability",
                benchmark["social_sustainability"],
                "public-stock retention relative to cooperative benchmark",
            ),
            ("collapse_rate", benchmark["social_collapse_rate"], "frequency of collapsed-pool states"),
            ("welfare", benchmark["social_welfare"], "reward plus retained-pool welfare proxy"),
        ]
        for component, reference, interpretation in components:
            if component == "extraction_minus_contribution":
                value = float(summary["extraction_total"] - summary["contribution_total"])
            else:
                value = float(summary[component])
            decomp_rows.append(
                _decomp_row(
                    world="public_goods",
                    mind=mind,
                    institution=institution,
                    seed=args.seed,
                    n_agents=args.n_agents,
                    trace_family="commons_stock_flow",
                    component=component,
                    value=value,
                    reference=reference,
                    interpretation=interpretation,
                )
            )
    return trace_rows, decomp_rows


def run_labor(args: argparse.Namespace, mind: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    n_workers = args.n_agents
    config = LaborMarketConfig(n_workers=n_workers, n_employers=args.n_agents, max_rounds=args.steps)
    benchmark = labor_benchmark_for_config(config)
    result = train_labor_market(steps=args.steps, seed=args.seed, config=config, mind=mind)
    trace_rows = _trace_rows(
        world="labor_market",
        mind=mind,
        institution="deferred_acceptance",
        seed=args.seed,
        n_agents=args.n_agents,
        records=result.records,
        max_rows=args.max_trace_rows,
    )
    summary = summarize_labor_records(result.records, args.final_window)
    decomp_rows = []
    components = {
        "match_rate": benchmark["truthful_match_rate"],
        "stability": 1.0,
        "truthful_report_rate": 1.0,
        "total_welfare": benchmark["truthful_total_welfare"],
        "blocking_pairs": 0.0,
        "manipulation_gain_mean": 0.0,
    }
    for component, reference in components.items():
        decomp_rows.append(
            _decomp_row(
                world="labor_market",
                mind=mind,
                institution="deferred_acceptance",
                seed=args.seed,
                n_agents=args.n_agents,
                trace_family="matching_truthfulness_stability",
                component=component,
                value=float(summary[component]),
                reference=float(reference),
                interpretation="matching channel: separate full matching, stability, truthfulness, and welfare",
            )
        )
    return trace_rows, decomp_rows


def run_resource_island(args: argparse.Namespace, mind: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resource_layout = (
        "contested"
        if args.resource_activation_preset == "pressure" and args.resource_layout == "random"
        else args.resource_layout
    )
    config = ResourceIslandConfig(
        grid_size=args.resource_grid_size,
        n_agents=args.n_agents,
        max_steps=args.steps,
        initial_resource_units=args.resource_initial_units,
        resource_layout=resource_layout,
        resource_spawn_probability=args.resource_spawn_probability,
        trade_radius=args.resource_trade_radius,
        trade_food_units=args.resource_trade_food_units,
        trade_wood_units=args.resource_trade_wood_units,
        trade_acquisition_reward=args.resource_trade_acquisition_reward,
        resource_preferences=specialization_preferences(args.resource_specialization_preset, args.n_agents),
        start_positions=activation_start_positions(
            args.resource_activation_preset,
            args.resource_grid_size,
            args.n_agents,
        ),
        initial_inventory=activation_initial_inventory(
            args.resource_activation_preset,
            args.n_agents,
            args.resource_trade_food_units,
            args.resource_trade_wood_units,
        ),
    )
    trace_rows: list[dict[str, Any]] = []
    decomp_rows: list[dict[str, Any]] = []
    for institution in args.resource_island_institutions:
        result = train_resource_island(
            steps=args.steps,
            seed=args.seed,
            institution=institution,
            config=config,
            mind=mind,
        )
        trace_rows.extend(
            _trace_rows(
                world="resource_island",
                mind=mind,
                institution=institution,
                seed=args.seed,
                n_agents=args.n_agents,
                records=result.records,
                max_rows=args.max_trace_rows,
            )
        )
        summary = summarize_resource_island_records(result.records, args.final_window)
        upper_bound = efficient_gather_upper_bound(result.world.resources, n_agents=args.n_agents, steps=args.steps)
        components = {
            "survival_rate": ("", "survival channel"),
            "welfare": ("", "total reward channel"),
            "trade_count": ("", "successful trade activation"),
            "trade_attempt_count": ("", "trade exploration/offer activation"),
            "trade_inventory_blocked_count": ("", "inventory/coordination failure in trade attempts"),
            "trade_institution_blocked_count": ("", "price-control or other institution binding"),
            "property_opportunities": ("", "non-owner opportunity exposure to claimed cells"),
            "property_violations": (0.0, "property-right rule binding through violations"),
            "specialization_index": ("", "heterogeneous resource-acquisition channel"),
            "resource_sustainability": ("", "resource-stock retention channel"),
        }
        for component, (reference, interpretation) in components.items():
            decomp_rows.append(
                _decomp_row(
                    world="resource_island",
                    mind=mind,
                    institution=institution,
                    seed=args.seed,
                    n_agents=args.n_agents,
                    trace_family="spatial_trade_property",
                    component=component,
                    value=float(summary[component]),
                    reference=reference,
                    interpretation=interpretation,
                )
            )
        decomp_rows.append(
            _decomp_row(
                world="resource_island",
                mind=mind,
                institution=institution,
                seed=args.seed,
                n_agents=args.n_agents,
                trace_family="spatial_trade_property",
                component="efficient_gather_upper_bound_remaining",
                value=float(upper_bound),
                reference="oracle/bracketing",
                interpretation="upper bound reference for remaining resource-gathering capacity",
            )
        )
    return trace_rows, decomp_rows


RUNNERS: dict[str, Callable[[argparse.Namespace, str], tuple[list[dict[str, Any]], list[dict[str, Any]]]]] = {
    "pricing_arena": run_pricing,
    "auction_house": run_auction,
    "public_goods": run_public_goods,
    "labor_market": run_labor,
    "resource_island": run_resource_island,
}


def write_markdown(path: Path, decomp_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Mechanism Trace Summary",
        "",
        "Generated by `run_mechanism_traces.py`.",
        "",
        "Each row is a single algebraic or diagnostic component that explains how an outcome was produced. "
        "These are trace diagnostics, not statistical full-run estimates.",
        "",
    ]
    current = None
    for row in decomp_rows:
        key = (row["world"], row["mind"], row["institution"], row["n_agents"])
        if key != current:
            current = key
            lines.extend(
                [
                    f"## {row['world']} / {row['mind']} / {row['institution']} / n={row['n_agents']}",
                    "",
                ]
            )
        value = row["value"]
        reference = row["reference"]
        gap = row["gap"]
        lines.append(
            f"- `{row['component']}` = `{value}`"
            f"{f' vs `{reference}`' if reference != '' else ''}"
            f"{f' (gap `{gap}`)' if gap != '' else ''}: {row['interpretation']}"
        )
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Path]:
    start = time.time()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    all_trace_rows: list[dict[str, Any]] = []
    all_decomp_rows: list[dict[str, Any]] = []
    worlds = DEFAULT_WORLDS if "all" in args.worlds else tuple(args.worlds)
    minds = DEFAULT_MINDS if "all" in args.minds else tuple(args.minds)
    for world in worlds:
        for mind in minds:
            print(f"=== trace world={world} mind={mind} n={args.n_agents} ===", flush=True)
            trace_rows, decomp_rows = RUNNERS[world](args, mind)
            all_trace_rows.extend(trace_rows)
            all_decomp_rows.extend(decomp_rows)

    trace_path = args.save_dir / "mechanism_trace_steps.csv"
    decomp_path = args.save_dir / "mechanism_decomposition.csv"
    md_path = args.save_dir / "mechanism_trace_summary.md"
    manifest_path = args.save_dir / "manifest.json"
    _write_csv(trace_path, all_trace_rows, TRACE_FIELDS)
    _write_csv(decomp_path, all_decomp_rows, DECOMPOSITION_FIELDS)
    write_markdown(md_path, all_decomp_rows)
    manifest_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.time() - start,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "config": {
                    key: _jsonable(value)
                    for key, value in vars(args).items()
                    if key != "save_dir"
                }
                | {"save_dir": str(args.save_dir)},
                "trace_row_count": len(all_trace_rows),
                "decomposition_row_count": len(all_decomp_rows),
                "outputs": [str(trace_path), str(decomp_path), str(md_path)],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "trace_steps": trace_path,
        "decomposition": decomp_path,
        "summary": md_path,
        "manifest": manifest_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reusable mechanism traces across worlds, minds, and N.")
    parser.add_argument("--worlds", nargs="+", default=["pricing_arena"], choices=("all", *DEFAULT_WORLDS))
    parser.add_argument("--minds", nargs="+", default=["q_learning"], choices=("all", *DEFAULT_MINDS))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--final-window", type=int, default=200)
    parser.add_argument("--max-trace-rows", type=int, default=500)
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/mechanism_traces"))
    parser.add_argument("--pricing-mechanisms", nargs="+", default=list(PRICING_MECHANISMS), choices=PRICING_MECHANISMS)
    parser.add_argument("--auction-scenarios", nargs="+", default=list(AUCTION_SCENARIOS), choices=AUCTION_SCENARIOS)
    parser.add_argument(
        "--public-goods-institutions",
        nargs="+",
        default=list(PUBLIC_GOODS_INSTITUTIONS),
        choices=PUBLIC_GOODS_INSTITUTIONS,
    )
    parser.add_argument("--public-goods-pool-capacity", type=float, default=20.0)
    parser.add_argument("--public-goods-initial-pool", type=float, default=10.0)
    parser.add_argument("--public-goods-regeneration-rate", type=float, default=0.08)
    parser.add_argument(
        "--resource-island-institutions",
        nargs="+",
        default=list(RESOURCE_ISLAND_INSTITUTIONS),
        choices=RESOURCE_ISLAND_INSTITUTIONS,
    )
    parser.add_argument("--resource-grid-size", type=int, default=5)
    parser.add_argument("--resource-initial-units", type=int, default=12)
    parser.add_argument("--resource-layout", choices=("random", "contested", "split"), default="contested")
    parser.add_argument("--resource-activation-preset", choices=("none", "pressure"), default="pressure")
    parser.add_argument("--resource-specialization-preset", choices=("none", "complementary"), default="complementary")
    parser.add_argument("--resource-spawn-probability", type=float, default=0.08)
    parser.add_argument("--resource-trade-radius", type=int, default=None)
    parser.add_argument("--resource-trade-food-units", type=int, default=2)
    parser.add_argument("--resource-trade-wood-units", type=int, default=1)
    parser.add_argument("--resource-trade-acquisition-reward", type=float, default=0.2)
    args = parser.parse_args(argv)
    if args.n_agents < 2:
        parser.error("--n-agents must be at least 2 for cross-agent traces")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.final_window < 1:
        parser.error("--final-window must be positive")
    if args.max_trace_rows is not None and args.max_trace_rows < 1:
        parser.error("--max-trace-rows must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    outputs = run(parse_args(argv))
    for path in outputs.values():
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
