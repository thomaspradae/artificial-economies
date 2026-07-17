from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


STATE_METRICS = (
    "sustainability_mean",
    "contribution_total_mean",
    "extraction_total_mean",
    "collapse_rate_mean",
)
REWARD_METRICS = (
    "welfare_mean",
    "reward_total_mean",
    "penalty_total_mean",
    "reputation_bonus_total_mean",
    "tax_revenue_mean",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def metric_delta(row: dict[str, str], baseline: dict[str, str], key: str) -> float:
    left = as_float(row, key)
    right = as_float(baseline, key)
    if not math.isfinite(left) or not math.isfinite(right):
        return float("nan")
    return left - right


def validate_effects(
    aggregate_path: Path,
    state_threshold: float = 0.01,
    reward_threshold: float = 0.01,
) -> dict[str, Any]:
    rows = read_rows(aggregate_path)
    by_institution = {row["institution"]: row for row in rows}
    if "none" not in by_institution:
        raise ValueError("Public Goods aggregate must contain baseline institution 'none'")
    baseline = by_institution["none"]
    institutions: dict[str, Any] = {}
    for institution, row in by_institution.items():
        if institution == "none":
            continue
        state_deltas = {metric: metric_delta(row, baseline, metric) for metric in STATE_METRICS}
        reward_deltas = {metric: metric_delta(row, baseline, metric) for metric in REWARD_METRICS}
        state_changed = any(
            math.isfinite(value) and abs(value) >= state_threshold
            for value in state_deltas.values()
        )
        reward_changed = any(
            math.isfinite(value) and abs(value) >= reward_threshold
            for value in reward_deltas.values()
        )
        if state_changed and reward_changed:
            classification = "state_and_reward"
        elif state_changed:
            classification = "state_only"
        elif reward_changed:
            classification = "reward_or_accounting_only"
        else:
            classification = "near_baseline"
        institutions[institution] = {
            "classification": classification,
            "state_changed": state_changed,
            "reward_changed": reward_changed,
            "state_deltas": state_deltas,
            "reward_deltas": reward_deltas,
        }
    return {
        "source": str(aggregate_path),
        "baseline": "none",
        "state_threshold": state_threshold,
        "reward_threshold": reward_threshold,
        "institutions": institutions,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate whether Public Goods institutions change state metrics.")
    parser.add_argument("--aggregate", type=Path, default=Path("outputs/public_goods_full/summary_aggregate.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/public_goods_full/institution_effect_validation.json"))
    parser.add_argument("--state-threshold", type=float, default=0.01)
    parser.add_argument("--reward-threshold", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = validate_effects(
        args.aggregate,
        state_threshold=args.state_threshold,
        reward_threshold=args.reward_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote: {args.output}")
    for institution, result in report["institutions"].items():
        print(f"{institution}: {result['classification']}")


if __name__ == "__main__":
    main()
