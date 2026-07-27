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

from arena_v0 import MarketConfig, logsumexp
from worlds.auction_house.benchmarks import (
    ex_post_bidder_regret,
    first_price_equilibrium_bid,
    truthful_bid_benchmark,
)
from worlds.labor_market.benchmarks import (
    best_worker_report_gains,
    blocking_pairs,
    canonical_matching_cases,
    preference_order,
    truthful_matching,
)
from worlds.pricing_arena.benchmarks import compute_static_benchmarks
from worlds.public_goods.benchmarks import free_rider_benchmark, social_optimum_benchmark
from worlds.public_goods.env import PublicGoodsConfig
from worlds.resource_island.benchmarks import (
    efficient_gather_upper_bound,
    greedy_full_information_gather_plan,
)
from worlds.resource_island.resources import layout_resource_map


FIELDNAMES = [
    "world",
    "institution",
    "mind",
    "n_agents",
    "check_name",
    "theory_anchor",
    "theory_prediction",
    "observed_metric",
    "expected",
    "observed",
    "gap",
    "status",
    "claim_scope",
    "notes",
]


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _row(
    *,
    world: str,
    institution: str,
    mind: str = "not_applicable",
    n_agents: int | str,
    check_name: str,
    theory_anchor: str,
    theory_prediction: str,
    observed_metric: str,
    expected: float | str,
    observed: float | str,
    gap: float | str,
    status: str,
    claim_scope: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "world": world,
        "institution": institution,
        "mind": mind,
        "n_agents": n_agents,
        "check_name": check_name,
        "theory_anchor": theory_anchor,
        "theory_prediction": theory_prediction,
        "observed_metric": observed_metric,
        "expected": expected,
        "observed": observed,
        "gap": gap,
        "status": status,
        "claim_scope": claim_scope,
        "notes": notes,
    }


def _pricing_demand(prices: np.ndarray, config: MarketConfig) -> np.ndarray:
    utilities = config.quality - config.alpha * prices
    logits = np.concatenate(([0.0], utilities)) / config.tau
    probabilities = np.exp(logits - logsumexp(logits))
    return config.market_size * probabilities[1:]


def _pricing_profit_for_firm(profile: np.ndarray, firm_index: int, config: MarketConfig) -> float:
    quantities = _pricing_demand(profile, config)
    return float((profile[firm_index] - config.cost) * quantities[firm_index])


def pricing_static_nash_rows(n_firms_values: list[int], tolerance: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    price_grid = np.linspace(1.0, 10.0, 19)
    for n_firms in n_firms_values:
        config = MarketConfig(
            mechanism="none",
            price_grid=price_grid,
            quality=np.full(n_firms, 8.0, dtype=float),
        )
        benchmark = compute_static_benchmarks(
            price_grid,
            n_firms=n_firms,
            cost=config.cost,
            market_size=config.market_size,
            alpha=config.alpha,
            tau=config.tau,
            quality=config.quality,
        )
        nash_price = benchmark["nash_price"]
        if nash_price is None:
            rows.append(
                _row(
                    world="pricing_arena",
                    institution="none",
                    n_agents=n_firms,
                    check_name="static_symmetric_nash_exists",
                    theory_anchor="one-shot Bertrand/logit pricing benchmark",
                    theory_prediction="a symmetric grid Nash price exists for the one-shot game",
                    observed_metric="nash_price",
                    expected="finite grid price",
                    observed="none",
                    gap="nan",
                    status="fail",
                    claim_scope="benchmark construction",
                    notes="No symmetric grid Nash price was found; do not use this benchmark as a reference line.",
                )
            )
            continue

        profile = np.full(n_firms, float(nash_price), dtype=float)
        base_profit = _pricing_profit_for_firm(profile, 0, config)
        best_deviation_profit = max(
            _pricing_profit_for_firm(np.asarray([candidate, *profile[1:]], dtype=float), 0, config)
            for candidate in price_grid
        )
        deviation_gain = float(best_deviation_profit - base_profit)
        rows.append(
            _row(
                world="pricing_arena",
                institution="none",
                n_agents=n_firms,
                check_name="static_symmetric_nash_best_response",
                theory_anchor="one-shot Bertrand/logit pricing benchmark",
                theory_prediction="at the reported symmetric Nash price, no firm has a profitable one-shot grid deviation",
                observed_metric="best_deviation_gain",
                expected=f"<= {tolerance:g}",
                observed=deviation_gain,
                gap=deviation_gain,
                status="pass" if deviation_gain <= tolerance else "fail",
                claim_scope="validates payoff/benchmark layer, not repeated-game learning convergence",
                notes=(
                    f"nash_price={float(nash_price):.6g}; monopoly_price={float(benchmark['monopoly_price']):.6g}. "
                    "Repeated learned policies may rationally deviate from this static target."
                ),
            )
        )
    return rows


def auction_known_answer_rows(n_bidders_values: list[int], tolerance: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valuation_grid = (0.0, 5.0, 10.0)
    bid_grid = (0.0, 5.0, 10.0)
    for n_bidders in n_bidders_values:
        max_truthful_regret = 0.0
        for valuations in np.array(np.meshgrid(*([valuation_grid] * n_bidders))).T.reshape(-1, n_bidders):
            bids = truthful_bid_benchmark(tuple(float(v) for v in valuations), bid_grid)
            for bidder in range(n_bidders):
                regret = ex_post_bidder_regret(
                    valuations=tuple(float(v) for v in valuations),
                    bids=bids,
                    bidder=bidder,
                    bid_grid=bid_grid,
                    auction_format="second_price",
                    reserve_price=0.0,
                )
                max_truthful_regret = max(max_truthful_regret, float(regret))
        rows.append(
            _row(
                world="auction_house",
                institution="second_price",
                n_agents=n_bidders,
                check_name="truthful_second_price_zero_grid_regret",
                theory_anchor="Vickrey second-price auction",
                theory_prediction="truthful bidding has no profitable unilateral grid-bid deviation",
                observed_metric="max_ex_post_truthful_regret",
                expected=f"<= {tolerance:g}",
                observed=max_truthful_regret,
                gap=max_truthful_regret,
                status="pass" if max_truthful_regret <= tolerance else "fail",
                claim_scope="exact benchmark check for private-value sealed-bid auction mechanics",
                notes="This is a theorem/mechanism sanity check; learned bidders still need bid-curve and regret inspection.",
            )
        )

        high_value = max(valuation_grid)
        shaded = first_price_equilibrium_bid(
            high_value,
            n_bidders=n_bidders,
            valuation_low=min(valuation_grid),
            bid_grid=bid_grid,
        )
        rows.append(
            _row(
                world="auction_house",
                institution="first_price",
                n_agents=n_bidders,
                check_name="first_price_shading_below_value",
                theory_anchor="symmetric IPV first-price auction benchmark",
                theory_prediction="risk-neutral symmetric bidders shade bids below value",
                observed_metric="shaded_high_value_bid",
                expected=f"< {high_value:g}",
                observed=shaded,
                gap=float(high_value - shaded),
                status="pass" if shaded < high_value else "fail",
                claim_scope="analytical reference check; grid coarseness affects exact bid distance",
                notes="This checks benchmark direction, not learned convergence to the first-price equilibrium.",
            )
        )
    return rows


def public_goods_known_answer_rows(n_agents_values: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_agents in n_agents_values:
        config = PublicGoodsConfig(n_agents=n_agents)
        free = free_rider_benchmark(config, steps=100)
        social = social_optimum_benchmark(config, steps=100)
        sustainability_gap = float(social["sustainability"] - free["sustainability"])
        contribution_gap = float(social["contribution_total"] - free["contribution_total"])
        rows.append(
            _row(
                world="public_goods",
                institution="none",
                n_agents=n_agents,
                check_name="free_rider_vs_cooperative_sustainability_bracket",
                theory_anchor="public goods / commons free-rider benchmark",
                theory_prediction="the cooperative contribution bracket preserves more public stock than all-extract free riding",
                observed_metric="sustainability_gap",
                expected="> 0",
                observed=sustainability_gap,
                gap=sustainability_gap,
                status="pass" if sustainability_gap > 0 else "fail",
                claim_scope="bracketing sanity check, not an equilibrium proof",
                notes=(
                    f"free_sustainability={free['sustainability']:.6g}; "
                    f"cooperative_sustainability={social['sustainability']:.6g}."
                ),
            )
        )
        rows.append(
            _row(
                world="public_goods",
                institution="none",
                n_agents=n_agents,
                check_name="cooperative_profile_has_more_contribution",
                theory_anchor="public goods / commons free-rider benchmark",
                theory_prediction="the cooperative bracket contributes more than the all-extract benchmark",
                observed_metric="contribution_gap",
                expected="> 0",
                observed=contribution_gap,
                gap=contribution_gap,
                status="pass" if contribution_gap > 0 else "fail",
                claim_scope="bracketing sanity check, not an equilibrium proof",
                notes=(
                    f"free_contribution={free['contribution_total']:.6g}; "
                    f"cooperative_contribution={social['contribution_total']:.6g}."
                ),
            )
        )
    return rows


def labor_market_known_answer_rows(tolerance: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = canonical_matching_cases()
    stable = cases["stable_truthful_2x2"]
    stable_bench = truthful_matching(stable["worker_values"], stable["employer_values"])
    rows.append(
        _row(
            world="labor_market",
            institution="deferred_acceptance",
            n_agents=2,
            check_name="truthful_da_stable_hand_case",
            theory_anchor="Gale-Shapley worker-proposing deferred acceptance",
            theory_prediction="truthful deferred acceptance returns a stable matching in the canonical 2x2 case",
            observed_metric="blocking_pair_count",
            expected="0",
            observed=len(stable_bench["blocking_pairs"]),
            gap=len(stable_bench["blocking_pairs"]),
            status="pass" if len(stable_bench["blocking_pairs"]) == 0 else "fail",
            claim_scope="exact matching-mechanism sanity check",
            notes=f"matches={list(map(int, stable_bench['matches']))}.",
        )
    )

    unstable = cases["unstable_forced_2x2"]
    worker_prefs = preference_order(unstable["worker_values"])
    employer_prefs = preference_order(unstable["employer_values"])
    forced_pairs = blocking_pairs(unstable["forced_matches"], worker_prefs, employer_prefs)
    rows.append(
        _row(
            world="labor_market",
            institution="deferred_acceptance",
            n_agents=2,
            check_name="blocking_pair_detector_flags_forced_unstable_case",
            theory_anchor="matching stability benchmark",
            theory_prediction="a deliberately crossed 2x2 matching has blocking pairs",
            observed_metric="blocking_pair_count",
            expected="> 0",
            observed=len(forced_pairs),
            gap=len(forced_pairs),
            status="pass" if len(forced_pairs) > 0 else "fail",
            claim_scope="diagnostic validity check for stability metric",
            notes=f"blocking_pairs={forced_pairs}.",
        )
    )

    contested = cases["contested_strategyproof_3x3"]
    gains = best_worker_report_gains(contested["worker_values"], contested["employer_values"])
    max_gain = float(np.max(gains))
    rows.append(
        _row(
            world="labor_market",
            institution="deferred_acceptance",
            n_agents=3,
            check_name="worker_proposing_da_no_profitable_worker_top_report",
            theory_anchor="strategy-proofness for proposing side under deferred acceptance",
            theory_prediction="workers do not gain from unilateral top-report deviations in the fixed 3x3 case",
            observed_metric="max_worker_report_gain",
            expected=f"<= {tolerance:g}",
            observed=max_gain,
            gap=max_gain,
            status="pass" if max_gain <= tolerance else "fail",
            claim_scope="fixed-profile theory sanity check; does not test employer-side manipulation",
            notes=f"worker_report_gains={gains.tolist()}.",
        )
    )
    return rows


def resource_island_known_answer_rows(n_agents_values: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_agents in n_agents_values:
        resources = layout_resource_map(
            grid_size=5,
            n_resource_types=2,
            resource_capacity=3,
            initial_resource_units=10,
            resource_layout="contested",
        )
        positions = [(0, 0), (4, 4), (0, 4), (4, 0)][:n_agents]
        while len(positions) < n_agents:
            positions.append((2, 2))
        steps = 4
        upper_bound = efficient_gather_upper_bound(resources, n_agents=n_agents, steps=steps)
        greedy = greedy_full_information_gather_plan(positions, resources, steps=steps)
        gap = float(upper_bound - greedy.estimated_gathered)
        rows.append(
            _row(
                world="resource_island",
                institution="none",
                n_agents=n_agents,
                check_name="greedy_planner_bounded_by_oracle_gather_upper_bound",
                theory_anchor="oracle/bracketing benchmark for spatial gather economy",
                theory_prediction="a travel-aware greedy plan cannot exceed the no-travel gather upper bound",
                observed_metric="upper_bound_minus_greedy_gather",
                expected=">= 0",
                observed=gap,
                gap=gap,
                status="pass" if gap >= 0 else "fail",
                claim_scope="engineering/bracketing sanity check; Resource Island has no closed-form equilibrium claim",
                notes=(
                    f"upper_bound={upper_bound}; greedy_estimated_gathered={greedy.estimated_gathered}; "
                    f"assignments={list(greedy.assignments)}."
                ),
            )
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    worlds = []
    for row in rows:
        if row["world"] not in worlds:
            worlds.append(row["world"])
    lines = [
        "# Known-Answer Sanity Checks",
        "",
        "Generated by `run_known_answer_sanity_checks.py`.",
        "",
        "These checks validate theory anchors, payoff accounting, benchmark construction, and diagnostic metrics. "
        "They are not final learned-policy results.",
        "",
    ]
    for world in worlds:
        lines.append(f"## {world}")
        lines.append("")
        for row in [item for item in rows if item["world"] == world]:
            lines.append(
                f"- **{row['status'].upper()}** `{row['check_name']}` "
                f"({row['institution']}, n={row['n_agents']}): "
                f"{row['observed_metric']} observed `{row['observed']}`; expected `{row['expected']}`. "
                f"Scope: {row['claim_scope']}"
            )
            if row["notes"]:
                lines.append(f"  Notes: {row['notes']}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Path]:
    start = time.time()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rows.extend(pricing_static_nash_rows(args.pricing_n_firms, args.tolerance))
    rows.extend(auction_known_answer_rows(args.auction_bidders, args.tolerance))
    rows.extend(public_goods_known_answer_rows(args.public_goods_agents))
    rows.extend(labor_market_known_answer_rows(args.tolerance))
    rows.extend(resource_island_known_answer_rows(args.resource_island_agents))

    csv_path = args.save_dir / "known_answer_sanity_checks.csv"
    md_path = args.save_dir / "known_answer_sanity_checks.md"
    manifest_path = args.save_dir / "manifest.json"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    manifest_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.time() - start,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "config": vars(args) | {"save_dir": str(args.save_dir)},
                "row_count": len(rows),
                "status_counts": {
                    status: sum(1 for row in rows if row["status"] == status)
                    for status in sorted({str(row["status"]) for row in rows})
                },
                "outputs": [str(csv_path), str(md_path)],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    return {"csv": csv_path, "markdown": md_path, "manifest": manifest_path}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run known-answer sanity checks across thesis worlds.")
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/known_answer_sanity_checks"))
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--pricing-n-firms", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--auction-bidders", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--public-goods-agents", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--resource-island-agents", nargs="+", type=int, default=[2, 4])
    args = parser.parse_args(argv)
    if any(value < 2 for value in args.pricing_n_firms):
        parser.error("--pricing-n-firms values must be at least 2")
    if any(value < 2 for value in args.auction_bidders):
        parser.error("--auction-bidders values must be at least 2")
    if any(value < 1 for value in args.public_goods_agents):
        parser.error("--public-goods-agents values must be positive")
    if any(value < 1 for value in args.resource_island_agents):
        parser.error("--resource-island-agents values must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    outputs = run(parse_args(argv))
    for path in outputs.values():
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
