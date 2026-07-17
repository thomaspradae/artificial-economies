from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import stdev
from typing import Any

from build_world_mind_comparison import MINDS, auto_results
from world_metric_schemas import WORLD_SCHEMAS


REQUIRED_MINDS = {"q_learning", "dqn", "ppo", "independent_dqn", "centralized_critic"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def numeric_values_are_finite(rows: list[dict[str, str]]) -> bool:
    for row in rows:
        for value in row.values():
            if value is None or value.strip() == "":
                return False
            try:
                number = float(value)
            except ValueError:
                continue
            if not math.isfinite(number):
                return False
    return True


def validate_world(world: str, outputs_dir: Path) -> dict[str, Any]:
    group_column = str(WORLD_SCHEMAS[world]["group_column"])
    results = auto_results(world, outputs_dir)
    minds = {mind for mind, _ in results}
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    missing = sorted(REQUIRED_MINDS - minds)
    if missing:
        errors.append(f"missing minds: {', '.join(missing)}")

    for mind, directory in results:
        seed_path = directory / "summary_by_seed.csv"
        aggregate_path = directory / "summary_aggregate.csv"
        manifest_path = directory / "experiment_manifest.json"
        if not seed_path.exists() or not aggregate_path.exists() or not manifest_path.exists():
            errors.append(f"{mind}: missing seed/aggregate/manifest artifact in {directory}")
            continue
        seed_rows = read_rows(seed_path)
        aggregate_rows = read_rows(aggregate_path)
        seed_column = "seed" if seed_rows and "seed" in seed_rows[0] else "seed_index"
        counts = Counter(row[group_column] for row in seed_rows)
        seed_counts = {
            group: len({row[seed_column] for row in seed_rows if row[group_column] == group})
            for group in counts
        }
        manifest = json.loads(manifest_path.read_text())
        config = manifest.get("config", manifest)
        check = {
            "mind": mind,
            "source_dir": str(directory),
            "seed_rows": len(seed_rows),
            "aggregate_rows": len(aggregate_rows),
            "groups": sorted(counts),
            "rows_per_group": dict(sorted(counts.items())),
            "distinct_seeds_per_group": dict(sorted(seed_counts.items())),
            "finite": numeric_values_are_finite(seed_rows) and numeric_values_are_finite(aggregate_rows),
            "manifest_steps": config.get("steps"),
            "manifest_n_seeds": config.get("n_seeds"),
        }
        checks.append(check)
        if not check["finite"]:
            errors.append(f"{mind}: blank/NaN/inf value")
        if any(count != 20 for count in counts.values()) or any(count != 20 for count in seed_counts.values()):
            errors.append(f"{mind}: expected 20 rows/distinct seeds per group")
        if len(aggregate_rows) != len(counts):
            errors.append(f"{mind}: aggregate/group count mismatch")
        if config.get("steps") != 40000 or config.get("n_seeds") != 20:
            errors.append(f"{mind}: manifest is not steps=40000, n_seeds=20")

    sanity: dict[str, Any] = {}
    if world == "labor_market":
        cc_dir = next((directory for mind, directory in results if mind == "centralized_critic"), None)
        if cc_dir:
            rows = read_rows(cc_dir / "summary_by_seed.csv")
            for metric in ("truthful_report_rate", "stability", "total_welfare"):
                values = [float(row[metric]) for row in rows]
                sanity[metric] = {
                    "min": min(values),
                    "max": max(values),
                    "std": stdev(values),
                    "unique": len(set(values)),
                }
                if len(set(values)) < 2 or stdev(values) == 0.0:
                    errors.append(f"centralized_critic: degenerate {metric}")

    return {
        "world": world,
        "status": "pass" if not errors else "fail",
        "minds": sorted(minds),
        "checks": checks,
        "labor_centralized_critic_sanity": sanity,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate all recovered full ladder result artifacts.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/full_ladder_validation.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worlds = sorted(WORLD_SCHEMAS)
    reports = [validate_world(world, args.outputs_dir) for world in worlds]
    payload = {
        "status": "pass" if all(report["status"] == "pass" for report in reports) else "fail",
        "worlds": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote: {args.output}")
    for report in reports:
        print(f"{report['world']}: {report['status']} minds={','.join(report['minds'])}")
        for error in report["errors"]:
            print(f"  ERROR: {error}")
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
