from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


COMPARISON_METRICS = (
    "survival_rate",
    "welfare",
    "specialization_index",
    "inequality_over_time",
    "resource_sustainability",
    "contact_rate",
    "trade_attempt_count",
    "trade_count",
    "trade_inventory_blocked_count",
    "trade_institution_blocked_count",
    "property_claims",
    "property_violations",
)


def _parse_result(value: str) -> tuple[str, Path]:
    parts = value.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("--result must have form mind:path")
    return parts[0], Path(parts[1])


def build_comparison(results: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mind, directory in results:
        aggregate_path = directory / "summary_aggregate.csv"
        if not aggregate_path.exists():
            raise FileNotFoundError(f"missing Resource Island aggregate: {aggregate_path}")
        with aggregate_path.open(newline="") as handle:
            sources = list(csv.DictReader(handle))
        for source in sources:
            out: dict[str, Any] = {
                "mind": mind,
                "institution": source["institution"],
                "n_seeds": int(source["n_seeds"]),
                "source_dir": str(directory),
            }
            for metric in COMPARISON_METRICS:
                mean_col = f"{metric}_mean"
                std_col = f"{metric}_std"
                if mean_col in source:
                    out[mean_col] = float(source[mean_col])
                if std_col in source:
                    out[std_col] = float(source[std_col])
            rows.append(out)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["mind", "institution", "n_seeds", "source_dir"]
    for metric in COMPARISON_METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Resource Island mind x institution comparison table.")
    parser.add_argument("--result", action="append", type=_parse_result, required=True, help="mind:path")
    parser.add_argument("--output", type=Path, default=Path("outputs/resource_island_mind_comparison.csv"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = build_comparison(args.result)
    write_csv(args.output, rows)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
