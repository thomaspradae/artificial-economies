from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from world_metric_schemas import WORLD_SCHEMAS

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt


CAPABILITY_TIERS = {
    "q_learning": 0,
    "random": -1,
    "dqn": 1,
    "ppo": 2,
    "independent_dqn": 3,
    "centralized_critic": 4,
}

KEY_METRICS = {
    "pricing_arena": ("welfare_mean", "profit_collusion_index_mean", "exploitability_mean"),
    "resource_island": ("welfare_mean", "survival_rate_mean", "trade_count_mean"),
    "auction_house": ("welfare_mean", "allocative_efficiency_mean", "ex_post_regret_mean_mean"),
    "public_goods": ("welfare_mean", "sustainability_mean", "collapse_rate_mean"),
    "labor_market": ("total_welfare_mean", "stability_mean", "truthful_report_rate_mean"),
}

DEFAULT_INPUTS = {
    "pricing_arena": Path("outputs/pricing_arena_phase3_full/mind_comparison.csv"),
    "resource_island": Path("outputs/resource_island_phase3_full/mind_comparison.csv"),
    "auction_house": Path("outputs/auction_house_phase3_full/mind_comparison.csv"),
    "public_goods": Path("outputs/public_goods_phase3_full/mind_comparison.csv"),
    "labor_market": Path("outputs/labor_market_phase3_full/mind_comparison.csv"),
}

# Fields that define the environment/protocol rather than the learning mind.
# Institution lists are reported separately as coverage because older baseline
# runs legitimately contain a subset of later variants.
PROTOCOL_KEYS = {
    "pricing_arena": ["steps", "n_seeds", "final_window", "seed_start", "seed_stride"],
    "resource_island": [
        "steps",
        "n_seeds",
        "grid_size",
        "n_agents",
        "obs_radius",
        "initial_resource_units",
        "resource_spawn_probability",
        "trade_radius",
        "resource_layout",
        "activation_preset",
        "specialization_preset",
        "trade_food_units",
        "trade_wood_units",
        "trade_acquisition_reward",
    ],
    "auction_house": ["steps", "n_seeds", "final_window", "n_bidders", "seed_start", "seed_stride"],
    "public_goods": [
        "steps",
        "n_seeds",
        "final_window",
        "n_agents",
        "initial_pool",
        "pool_capacity",
        "regeneration_rate",
        "seed_start",
        "seed_stride",
    ],
    "labor_market": [
        "steps",
        "n_seeds",
        "final_window",
        "n_workers",
        "n_employers",
        "seed_start",
        "seed_stride",
    ],
}

PROTOCOL_DEFAULTS = {
    "resource_island": {
        "resource_layout": "random",
        "activation_preset": "none",
        "specialization_preset": "none",
        "trade_food_units": 1,
        "trade_wood_units": 1,
        "trade_acquisition_reward": 0.0,
    }
}


def _parse_input(value: str) -> tuple[str, Path]:
    parts = value.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("--input must have form world:path")
    return parts[0], Path(parts[1])


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return float("nan")


def _group_value(row: dict[str, str]) -> str:
    return row.get("institution") or row.get("mechanism") or row.get("scenario") or "baseline"


def load_synthesis_rows(inputs: list[tuple[str, Path]], *, require_all: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for world, path in inputs:
        if not path.exists():
            if require_all:
                raise FileNotFoundError(f"missing canonical full comparison for {world}: {path}")
            continue
        metrics = KEY_METRICS.get(world, ())
        with path.open(newline="") as handle:
            for source in csv.DictReader(handle):
                mind = source["mind"]
                out: dict[str, Any] = {
                    "world": world,
                    "institution": _group_value(source),
                    "mind": mind,
                    "capability_tier": CAPABILITY_TIERS.get(mind, -1),
                    "source_path": str(path),
                    "raw_source_dir": source.get("source_dir", ""),
                }
                for index, metric in enumerate(metrics, start=1):
                    out[f"key_metric_{index}_name"] = metric
                    out[f"key_metric_{index}"] = _float(source, metric)
                rows.append(out)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "world",
        "institution",
        "mind",
        "capability_tier",
        "key_metric_1_name",
        "key_metric_1",
        "key_metric_2_name",
        "key_metric_2",
        "key_metric_3_name",
        "key_metric_3",
        "source_path",
        "raw_source_dir",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _monotonic(values: list[float]) -> str:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) < 2:
        return "insufficient"
    nondecreasing = all(a <= b + 1e-12 for a, b in zip(finite, finite[1:]))
    nonincreasing = all(a + 1e-12 >= b for a, b in zip(finite, finite[1:]))
    if nondecreasing and nonincreasing:
        return "flat"
    if nondecreasing:
        return "nondecreasing"
    if nonincreasing:
        return "nonincreasing"
    return "non_monotonic"


def build_monotonicity_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Keep institutions separate. Averaging unmatched institution sets across
    # minds creates a false capability curve when historical coverage differs.
    grouped: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        tier = int(row["capability_tier"])
        if tier < 0:
            continue
        for index in (1, 2, 3):
            name = row.get(f"key_metric_{index}_name")
            value = row.get(f"key_metric_{index}")
            if name and isinstance(value, (float, int)) and math.isfinite(float(value)):
                grouped[(row["world"], row["institution"], name)][tier].append(float(value))

    report: dict[str, Any] = {}
    for (world, institution, metric), tier_values in sorted(grouped.items()):
        ordered_tiers = sorted(tier_values)
        means = [_mean(tier_values[tier]) for tier in ordered_tiers]
        report.setdefault(world, {}).setdefault(institution, {})[metric] = {
            "tiers": ordered_tiers,
            "tier_means": means,
            "classification": _monotonic(means),
        }
    return report


def build_coverage_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_world_mind: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        by_world_mind[row["world"]][row["mind"]].add(row["institution"])
    report: dict[str, Any] = {}
    for world, mind_groups in sorted(by_world_mind.items()):
        group_sets = list(mind_groups.values())
        common = set.intersection(*group_sets) if group_sets else set()
        union = set.union(*group_sets) if group_sets else set()
        report[world] = {
            "institutions_by_mind": {mind: sorted(groups) for mind, groups in sorted(mind_groups.items())},
            "common_institutions": sorted(common),
            "all_institutions": sorted(union),
            "balanced": all(groups == group_sets[0] for groups in group_sets[1:]) if group_sets else True,
        }
    return report


def build_protocol_comparability_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sources: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        if row.get("raw_source_dir"):
            sources[row["world"]][row["mind"]] = str(row["raw_source_dir"])

    report: dict[str, Any] = {}
    for world, mind_sources in sorted(sources.items()):
        keys = PROTOCOL_KEYS[world]
        signatures: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        for mind, source_dir in sorted(mind_sources.items()):
            manifest_path = Path(source_dir) / "experiment_manifest.json"
            if not manifest_path.exists():
                errors.append(f"{mind}: missing {manifest_path}")
                continue
            manifest = json.loads(manifest_path.read_text())
            config = manifest.get("config", manifest)
            defaults = PROTOCOL_DEFAULTS.get(world, {})
            signatures[mind] = {key: config.get(key, defaults.get(key)) for key in keys}
        baseline = signatures.get("q_learning")
        matches = {
            mind: baseline is not None and signature == baseline
            for mind, signature in signatures.items()
            if mind != "q_learning"
        }
        mismatches = sorted(mind for mind, matched in matches.items() if not matched)
        report[world] = {
            "status": "pass" if not errors and not mismatches else "mismatch",
            "baseline_mind": "q_learning",
            "selected_protocol_fields": keys,
            "signatures_by_mind": signatures,
            "matches_q_learning": matches,
            "mismatched_minds": mismatches,
            "errors": errors,
            "cross_mind_capability_claims_valid": not errors and not mismatches,
        }
    return report


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    by_world: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        tier = int(row["capability_tier"])
        value = row.get("key_metric_1")
        baseline = str(WORLD_SCHEMAS[row["world"]]["baseline"])
        if (
            row["institution"] == baseline
            and tier >= 0
            and isinstance(value, (float, int))
            and math.isfinite(float(value))
        ):
            by_world[row["world"]][tier].append(float(value))
    if not by_world:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for world, tier_values in sorted(by_world.items()):
        tiers = sorted(tier_values)
        means = [_mean(tier_values[tier]) for tier in tiers]
        finite = [value for value in means if math.isfinite(value)]
        if not finite:
            continue
        min_v, max_v = min(finite), max(finite)
        denom = max(max_v - min_v, 1e-9)
        normalized = [(value - min_v) / denom if math.isfinite(value) else float("nan") for value in means]
        plt.plot(tiers, normalized, marker="o", label=world)
    plt.xlabel("Capability tier")
    plt.ylabel("Normalized baseline-institution primary metric")
    plt.xticks(sorted(set(CAPABILITY_TIERS.values())))
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cross-world capability synthesis outputs.")
    parser.add_argument("--input", action="append", type=_parse_input, default=None, help="world:path")
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/cross_world_synthesis"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    inputs = args.input if args.input else list(DEFAULT_INPUTS.items())
    rows = load_synthesis_rows(inputs)
    present_worlds = {row["world"] for row in rows}
    missing_worlds = sorted(set(DEFAULT_INPUTS) - present_worlds)
    if missing_worlds:
        raise SystemExit(f"missing synthesis worlds: {', '.join(missing_worlds)}")
    for world in sorted(present_worlds):
        minds = {row["mind"] for row in rows if row["world"] == world}
        expected = {"q_learning", "dqn", "ppo", "independent_dqn", "centralized_critic"}
        missing_minds = sorted(expected - minds)
        if missing_minds:
            raise SystemExit(f"{world} missing required full minds: {', '.join(missing_minds)}")
    table_path = args.save_dir / "synthesis_table.csv"
    baseline_path = args.save_dir / "baseline_capability_table.csv"
    report_path = args.save_dir / "monotonicity_report.json"
    coverage_path = args.save_dir / "coverage_report.json"
    protocol_path = args.save_dir / "protocol_comparability_report.json"
    figure_path = args.save_dir / "capability_ladder.png"
    write_csv(table_path, rows)
    baseline_rows = [
        row for row in rows if row["institution"] == str(WORLD_SCHEMAS[row["world"]]["baseline"])
    ]
    write_csv(baseline_path, baseline_rows)
    report = build_monotonicity_report(rows)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    coverage_path.write_text(json.dumps(build_coverage_report(rows), indent=2, sort_keys=True) + "\n")
    protocol_path.write_text(
        json.dumps(build_protocol_comparability_report(rows), indent=2, sort_keys=True) + "\n"
    )
    write_plot(figure_path, rows)
    print(f"Wrote: {table_path}")
    print(f"Wrote: {baseline_path}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {coverage_path}")
    print(f"Wrote: {protocol_path}")
    if figure_path.exists():
        print(f"Wrote: {figure_path}")


if __name__ == "__main__":
    main()
