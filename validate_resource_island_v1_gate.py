from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_PROTOCOL = {
    "resource_layout": "contested",
    "activation_preset": "pressure",
    "specialization_preset": "complementary",
    "trade_food_units": 2,
    "trade_wood_units": 1,
    "trade_acquisition_reward": 0.2,
    "trade_radius": 8,
    "institutions": ["none", "property_rights", "trade_price_controls", "reputation_system"],
}


def parse_result(value: str) -> tuple[str, Path]:
    mind, separator, path = value.partition(":")
    if not separator or not mind or not path:
        raise argparse.ArgumentTypeError("--result must have form mind:path")
    return mind, Path(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_result(mind: str, directory: Path, steps: int, n_seeds: int) -> dict[str, Any]:
    errors: list[str] = []
    required = [
        directory / "summary_by_seed.csv",
        directory / "summary_aggregate.csv",
        directory / "experiment_manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return {"mind": mind, "directory": str(directory), "status": "fail", "errors": missing}

    aggregate = read_rows(directory / "summary_aggregate.csv")
    by_seed = read_rows(directory / "summary_by_seed.csv")
    manifest = json.loads((directory / "experiment_manifest.json").read_text())
    config = manifest.get("config", manifest)
    rows = {row["institution"]: row for row in aggregate}
    expected_institutions = EXPECTED_PROTOCOL["institutions"]
    if set(rows) != set(expected_institutions):
        errors.append(f"institutions={sorted(rows)} expected={expected_institutions}")
    if len(by_seed) != len(expected_institutions) * n_seeds:
        errors.append(f"by_seed rows={len(by_seed)} expected={len(expected_institutions) * n_seeds}")
    if config.get("steps") != steps or config.get("n_seeds") != n_seeds:
        errors.append(
            f"manifest steps/n_seeds={config.get('steps')}/{config.get('n_seeds')} expected={steps}/{n_seeds}"
        )
    for key, expected in EXPECTED_PROTOCOL.items():
        if config.get(key) != expected:
            errors.append(f"manifest {key}={config.get(key)!r} expected={expected!r}")

    for row in aggregate + by_seed:
        for key, value in row.items():
            if value is None or value == "":
                errors.append(f"blank value in {key}")
                break
            try:
                number = float(value)
            except ValueError:
                continue
            if not math.isfinite(number):
                errors.append(f"non-finite value in {key}")
                break

    property_opportunities = float(rows.get("property_rights", {}).get("property_opportunities_mean", 0.0))
    trade_attempts = sum(float(row.get("trade_attempt_count_mean", 0.0)) for row in aggregate)
    successful_trades = sum(float(row.get("trade_count_mean", 0.0)) for row in aggregate)
    controls_bound = float(
        rows.get("trade_price_controls", {}).get("trade_institution_blocked_count_mean", 0.0)
    )
    if property_opportunities <= 0.0:
        errors.append("property channel did not activate")
    if trade_attempts <= 0.0:
        errors.append("trade action channel did not activate")
    if successful_trades <= 0.0 and controls_bound <= 0.0:
        errors.append("neither successful trade nor binding trade controls were observed")

    return {
        "mind": mind,
        "directory": str(directory),
        "status": "pass" if not errors else "fail",
        "property_opportunities_mean": property_opportunities,
        "trade_attempts_sum_mean": trade_attempts,
        "successful_trades_sum_mean": successful_trades,
        "trade_controls_blocked_mean": controls_bound,
        "errors": sorted(set(errors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate exact-protocol Resource Island v1 gates.")
    parser.add_argument("--result", action="append", type=parse_result, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--n-seeds", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [validate_result(mind, path, args.steps, args.n_seeds) for mind, path in args.result]
    payload = {
        "status": "pass" if all(report["status"] == "pass" for report in reports) else "fail",
        "steps": args.steps,
        "n_seeds": args.n_seeds,
        "results": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
