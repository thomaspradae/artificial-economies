from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from worlds.labor_market.benchmarks import (
    best_worker_report_gains,
    blocking_pairs,
    canonical_matching_cases,
    matching_welfare,
    preference_order,
    truthful_matching,
)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def build_report() -> dict[str, Any]:
    report: dict[str, Any] = {"cases": {}}
    for name, case in canonical_matching_cases().items():
        worker_values = np.asarray(case["worker_values"], dtype=float)
        employer_values = np.asarray(case["employer_values"], dtype=float)
        truthful = truthful_matching(worker_values, employer_values)
        matches = np.asarray(truthful["matches"], dtype=int)
        row: dict[str, Any] = {
            "worker_values": worker_values,
            "employer_values": employer_values,
            "truthful_matches": matches,
            "truthful_blocking_pairs": truthful["blocking_pairs"],
            "truthful_welfare": matching_welfare(matches, worker_values, employer_values),
            "worker_report_gains": best_worker_report_gains(worker_values, employer_values),
        }
        if "forced_matches" in case:
            forced = np.asarray(case["forced_matches"], dtype=int)
            row["forced_matches"] = forced
            row["forced_blocking_pairs"] = blocking_pairs(
                forced,
                preference_order(worker_values),
                preference_order(employer_values),
            )
            row["forced_welfare"] = matching_welfare(forced, worker_values, employer_values)
        report["cases"][name] = row
    return to_jsonable(report)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write fixed Labor Market benchmark case report.")
    parser.add_argument("--output", type=Path, default=Path("outputs/labor_market_benchmark_cases.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_report(), indent=2, sort_keys=True) + "\n")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
