from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from arena_v0 import DuopolyMarket, MarketConfig
from core.metrics import profit_collusion_index


COMBINED_FIELDS = [
    "mind",
    "mechanism",
    "n_seeds_multiseed",
    "n_seeds_exploitability",
    "avg_price_mean",
    "avg_price_ci95_low",
    "avg_price_ci95_high",
    "price_dispersion_mean",
    "profit_total_mean",
    "profit_total_ci95_low",
    "profit_total_ci95_high",
    "quantity_total_mean",
    "welfare_mean",
    "welfare_ci95_low",
    "welfare_ci95_high",
    "consumer_surplus_mean",
    "consumer_surplus_ci95_low",
    "consumer_surplus_ci95_high",
    "collusion_index_mean",
    "collusion_index_ci95_low",
    "collusion_index_ci95_high",
    "profit_collusion_index_mean",
    "profit_collusion_index_ci95_low",
    "profit_collusion_index_ci95_high",
    "exploitability_mean",
    "exploitability_ci95_low",
    "exploitability_ci95_high",
    "victim_loss_mean",
    "victim_loss_ci95_low",
    "victim_loss_ci95_high",
    "welfare_damage_mean",
    "welfare_damage_ci95_low",
    "welfare_damage_ci95_high",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def by_mechanism(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["mechanism"]: row for row in rows}


def get(row: dict[str, str] | None, key: str) -> str:
    if row is None:
        return ""
    return row.get(key, "")


def static_profit_benchmarks() -> dict[str, float]:
    """Return total-profit Nash and joint-profit benchmarks for the base pricing game."""
    config = MarketConfig()
    market = DuopolyMarket(config)
    prices = list(config.price_grid)
    nash_index = prices.index(2.5)
    monopoly_index = prices.index(8.0)
    _, _, nash_info = market.step((nash_index, nash_index))
    _, _, monopoly_info = market.step((monopoly_index, monopoly_index))
    return {
        "nash_profit_total": float(nash_info["profit1"] + nash_info["profit2"]),
        "monopoly_profit_total": float(monopoly_info["profit1"] + monopoly_info["profit2"]),
    }


PROFIT_BENCHMARKS = static_profit_benchmarks()


def profit_index(row: dict[str, str] | None, key: str) -> str:
    if row is None:
        return ""
    value = row.get(key, "")
    if value == "":
        return ""
    return str(
        profit_collusion_index(
            float(value),
            PROFIT_BENCHMARKS["nash_profit_total"],
            PROFIT_BENCHMARKS["monopoly_profit_total"],
        )
    )


def combined_rows(multiseed_dir: Path, exploitability_dir: Path | None, mind: str) -> list[dict[str, Any]]:
    multiseed = by_mechanism(read_rows(multiseed_dir / "summary_aggregate.csv"))
    exploitability = {}
    if exploitability_dir is not None and (exploitability_dir / "summary_aggregate.csv").exists():
        exploitability = by_mechanism(read_rows(exploitability_dir / "summary_aggregate.csv"))

    rows = []
    for mechanism, multi in multiseed.items():
        exploit = exploitability.get(mechanism)
        rows.append(
            {
                "mind": mind,
                "mechanism": mechanism,
                "n_seeds_multiseed": get(multi, "n_seeds"),
                "n_seeds_exploitability": get(exploit, "n_seeds"),
                "avg_price_mean": get(multi, "avg_price_mean"),
                "avg_price_ci95_low": get(multi, "avg_price_ci95_low"),
                "avg_price_ci95_high": get(multi, "avg_price_ci95_high"),
                "price_dispersion_mean": get(multi, "price_dispersion_mean"),
                "profit_total_mean": get(multi, "profit_total_mean"),
                "profit_total_ci95_low": get(multi, "profit_total_ci95_low"),
                "profit_total_ci95_high": get(multi, "profit_total_ci95_high"),
                "quantity_total_mean": get(multi, "quantity_total_mean"),
                "welfare_mean": get(multi, "welfare_mean"),
                "welfare_ci95_low": get(multi, "welfare_ci95_low"),
                "welfare_ci95_high": get(multi, "welfare_ci95_high"),
                "consumer_surplus_mean": get(multi, "consumer_surplus_mean"),
                "consumer_surplus_ci95_low": get(multi, "consumer_surplus_ci95_low"),
                "consumer_surplus_ci95_high": get(multi, "consumer_surplus_ci95_high"),
                "collusion_index_mean": get(multi, "collusion_index_mean"),
                "collusion_index_ci95_low": get(multi, "collusion_index_ci95_low"),
                "collusion_index_ci95_high": get(multi, "collusion_index_ci95_high"),
                "profit_collusion_index_mean": profit_index(multi, "profit_total_mean"),
                "profit_collusion_index_ci95_low": profit_index(multi, "profit_total_ci95_low"),
                "profit_collusion_index_ci95_high": profit_index(multi, "profit_total_ci95_high"),
                "exploitability_mean": get(exploit, "exploitability_mean"),
                "exploitability_ci95_low": get(exploit, "exploitability_ci95_low"),
                "exploitability_ci95_high": get(exploit, "exploitability_ci95_high"),
                "victim_loss_mean": get(exploit, "victim_loss_mean"),
                "victim_loss_ci95_low": get(exploit, "victim_loss_ci95_low"),
                "victim_loss_ci95_high": get(exploit, "victim_loss_ci95_high"),
                "welfare_damage_mean": get(exploit, "welfare_damage_mean"),
                "welfare_damage_ci95_low": get(exploit, "welfare_damage_ci95_low"),
                "welfare_damage_ci95_high": get(exploit, "welfare_damage_ci95_high"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMBINED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiseed and exploitability aggregate result tables.")
    parser.add_argument("--multiseed-dir", type=Path, default=Path("outputs/full_v0_multiseed"))
    parser.add_argument("--exploitability-dir", type=Path, default=Path("outputs/v1_exploitability"))
    parser.add_argument("--mind", default="q_learning")
    parser.add_argument("--random-multiseed-dir", type=Path)
    parser.add_argument("--random-exploitability-dir", type=Path)
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        metavar="MIND:MULTISEED_DIR:EXPLOITABILITY_DIR",
        help="Additional mind result triple to merge. Use an empty exploitability dir field if absent.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/combined_phase1/institution_summary.csv"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = combined_rows(args.multiseed_dir, args.exploitability_dir, args.mind)
    if args.random_multiseed_dir is not None:
        rows.extend(combined_rows(args.random_multiseed_dir, args.random_exploitability_dir, "random"))
    for spec in args.result:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError("--result must have format MIND:MULTISEED_DIR:EXPLOITABILITY_DIR")
        mind, multiseed_dir, exploitability_dir = parts
        rows.extend(
            combined_rows(
                Path(multiseed_dir),
                Path(exploitability_dir) if exploitability_dir else None,
                mind,
            )
        )
    write_csv(args.output, rows)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
