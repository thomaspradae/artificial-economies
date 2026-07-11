from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MECHANISMS = (
    "none",
    "price_cap",
    "tax_high_price",
    "random_audit",
    "anti_collusion",
    "demand_shock",
)

PHASE3_MINDS = ("dqn", "ppo", "independent_dqn", "centralized_critic")
COMBINED_MINDS = ("q_learning", "random", *PHASE3_MINDS)
STRING_FIELDS = {"mind", "mechanism", "selection_metric"}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise AssertionError(f"missing required CSV: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AssertionError(f"missing required JSON: {path}")
    with path.open() as handle:
        return json.load(handle)


def assert_finite_numeric_table(path: Path, rows: list[dict[str, str]]) -> None:
    for row_index, row in enumerate(rows, start=1):
        for field, value in row.items():
            if field in STRING_FIELDS:
                continue
            if value == "":
                raise AssertionError(f"{path}: empty numeric value at row {row_index}, field {field}")
            try:
                number = float(value)
            except ValueError as exc:
                raise AssertionError(f"{path}: nonnumeric value {value!r} at row {row_index}, field {field}") from exc
            if not math.isfinite(number):
                raise AssertionError(f"{path}: nonfinite value {value!r} at row {row_index}, field {field}")


def assert_mechanism_counts(path: Path, rows: list[dict[str, str]], expected_per_mechanism: int) -> None:
    counts = Counter(row["mechanism"] for row in rows)
    expected = {mechanism: expected_per_mechanism for mechanism in MECHANISMS}
    if dict(counts) != expected:
        raise AssertionError(f"{path}: mechanism counts {dict(counts)} != expected {expected}")


def assert_manifest_config(path: Path, expected: dict[str, Any]) -> None:
    manifest = read_json(path)
    config = manifest.get("config", {})
    for key, value in expected.items():
        if config.get(key) != value:
            raise AssertionError(f"{path}: config[{key!r}]={config.get(key)!r}, expected {value!r}")


def validate_multiseed(mind: str, root: Path) -> dict[str, Any]:
    path = root / f"{mind}_v0_multiseed"
    by_seed = read_csv(path / "summary_by_seed.csv")
    aggregate = read_csv(path / "summary_aggregate.csv")

    if len(by_seed) != len(MECHANISMS) * 20:
        raise AssertionError(f"{path}: summary_by_seed row count {len(by_seed)} != 120")
    if len(aggregate) != len(MECHANISMS):
        raise AssertionError(f"{path}: summary_aggregate row count {len(aggregate)} != 6")

    assert_mechanism_counts(path / "summary_by_seed.csv", by_seed, 20)
    assert_mechanism_counts(path / "summary_aggregate.csv", aggregate, 1)
    assert_finite_numeric_table(path / "summary_by_seed.csv", by_seed)
    assert_finite_numeric_table(path / "summary_aggregate.csv", aggregate)
    assert_manifest_config(
        path / "experiment_manifest.json",
        {
            "mind": mind,
            "steps": 40_000,
            "n_seeds": 20,
            "mechanisms": list(MECHANISMS),
        },
    )
    return {"path": str(path), "summary_by_seed_rows": len(by_seed), "summary_aggregate_rows": len(aggregate)}


def validate_exploitability(mind: str, root: Path) -> dict[str, Any]:
    path = root / f"{mind}_v1_exploitability"
    restarts = read_csv(path / "restarts_by_seed.csv")
    by_seed = read_csv(path / "summary_by_seed.csv")
    aggregate = read_csv(path / "summary_aggregate.csv")

    if len(restarts) != len(MECHANISMS) * 20 * 3:
        raise AssertionError(f"{path}: restarts_by_seed row count {len(restarts)} != 360")
    if len(by_seed) != len(MECHANISMS) * 20:
        raise AssertionError(f"{path}: summary_by_seed row count {len(by_seed)} != 120")
    if len(aggregate) != len(MECHANISMS):
        raise AssertionError(f"{path}: summary_aggregate row count {len(aggregate)} != 6")

    assert_mechanism_counts(path / "restarts_by_seed.csv", restarts, 60)
    assert_mechanism_counts(path / "summary_by_seed.csv", by_seed, 20)
    assert_mechanism_counts(path / "summary_aggregate.csv", aggregate, 1)
    assert_finite_numeric_table(path / "restarts_by_seed.csv", restarts)
    assert_finite_numeric_table(path / "summary_by_seed.csv", by_seed)
    assert_finite_numeric_table(path / "summary_aggregate.csv", aggregate)
    assert_manifest_config(
        path / "experiment_manifest.json",
        {
            "incumbent_mind": mind,
            "incumbent_steps": 40_000,
            "adversary_steps": 20_000,
            "evaluation_steps": 5_000,
            "n_seeds": 20,
            "adversary_restarts": 3,
            "mechanisms": list(MECHANISMS),
        },
    )
    return {
        "path": str(path),
        "restarts_by_seed_rows": len(restarts),
        "summary_by_seed_rows": len(by_seed),
        "summary_aggregate_rows": len(aggregate),
    }


def validate_combined(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    expected_rows = len(COMBINED_MINDS) * len(MECHANISMS)
    if len(rows) != expected_rows:
        raise AssertionError(f"{path}: combined row count {len(rows)} != {expected_rows}")

    by_pair = Counter((row["mind"], row["mechanism"]) for row in rows)
    expected_pairs = {(mind, mechanism): 1 for mind in COMBINED_MINDS for mechanism in MECHANISMS}
    if dict(by_pair) != expected_pairs:
        raise AssertionError(f"{path}: combined mind/mechanism coverage is incomplete")

    assert_finite_numeric_table(path, rows)
    return {"path": str(path), "rows": len(rows), "minds": list(COMBINED_MINDS)}


def build_report(root: Path, combined_path: Path) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "multiseed": {mind: validate_multiseed(mind, root) for mind in PHASE3_MINDS},
        "exploitability": {mind: validate_exploitability(mind, root) for mind in PHASE3_MINDS},
        "combined": validate_combined(combined_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate full Phase 3 n=20 PyTorch/MARL outputs.")
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--combined", type=Path, default=Path("outputs/phase3_full/mind_comparison.csv"))
    parser.add_argument("--report", type=Path, default=Path("outputs/phase3_full/validation_report.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        report = build_report(args.root, args.combined)
    except Exception as exc:
        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "fail",
            "error": str(exc),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        raise

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(f"Wrote: {args.report}")


if __name__ == "__main__":
    main()
