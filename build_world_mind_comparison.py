from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from world_metric_schemas import WORLD_SCHEMAS


MINDS = ["q_learning", "random", "dqn", "ppo", "independent_dqn", "centralized_critic"]
REQUIRED_LADDER_MINDS = ["q_learning", "dqn", "ppo", "independent_dqn", "centralized_critic"]

# Historical output names are preserved.  The values are relative to --outputs-dir.
DIRECTORY_ALIASES: dict[tuple[str, str], str | tuple[str, ...]] = {
    ("pricing_arena", "q_learning"): "full_v0_multiseed",
    ("pricing_arena", "random"): "random_v0_multiseed",
    ("pricing_arena", "dqn"): "dqn_v0_multiseed",
    ("pricing_arena", "ppo"): "ppo_v0_multiseed",
    ("pricing_arena", "independent_dqn"): "independent_dqn_v0_multiseed_fixed",
    ("pricing_arena", "centralized_critic"): "centralized_critic_v0_multiseed",
    ("resource_island", "q_learning"): "resource_island_v1_full",
    ("resource_island", "dqn"): ("resource_island_v1_dqn_full", "resource_island_dqn_full"),
    ("resource_island", "ppo"): ("resource_island_v1_ppo_full", "resource_island_ppo_full"),
    ("resource_island", "independent_dqn"): (
        "resource_island_v1_independent_dqn_full",
        "resource_island_independent_dqn_full_fixed",
    ),
    ("resource_island", "centralized_critic"): (
        "resource_island_v1_centralized_critic_full",
        "resource_island_centralized_critic_full",
    ),
    ("auction_house", "q_learning"): "auction_house_full",
    ("public_goods", "q_learning"): "public_goods_full",
    ("labor_market", "q_learning"): "labor_market_full",
}

EXPLOITABILITY_ALIASES: dict[str, str] = {
    "q_learning": "v1_exploitability",
    "random": "random_v1_exploitability",
    "dqn": "dqn_v1_exploitability",
    "ppo": "ppo_v1_exploitability",
    "independent_dqn": "independent_dqn_v1_exploitability_fixed",
    "centralized_critic": "centralized_critic_v1_exploitability",
}

DEFAULT_OUTPUT_DIRS = {
    "pricing_arena": "pricing_arena_phase3_full",
    "resource_island": "resource_island_phase3_full",
    "auction_house": "auction_house_phase3_full",
    "public_goods": "public_goods_phase3_full",
    "labor_market": "labor_market_phase3_full",
}


def _parse_result(value: str) -> tuple[str, Path]:
    parts = value.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("--result must have form mind:path")
    return parts[0], Path(parts[1])


def _coerce(value: str) -> Any:
    try:
        if value.strip() == "":
            return value
        return float(value)
    except ValueError:
        return value


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_dir(world: str, mind: str, outputs_dir: Path) -> Path | None:
    alias = DIRECTORY_ALIASES.get((world, mind))
    if alias:
        aliases = (alias,) if isinstance(alias, str) else alias
        for name in aliases:
            candidate = outputs_dir / name
            if all((candidate / artifact).exists() for artifact in (
                "summary_by_seed.csv",
                "summary_aggregate.csv",
                "experiment_manifest.json",
            )):
                return candidate
        return None
    nested = outputs_dir / f"{world}_phase3_full" / mind
    if nested.exists():
        return nested
    flat = outputs_dir / f"{world}_{mind}_full"
    return flat if flat.exists() else None


def auto_results(world: str, outputs_dir: Path) -> list[tuple[str, Path]]:
    return [(mind, directory) for mind in MINDS if (directory := resolve_dir(world, mind, outputs_dir))]


def comparison_columns(
    rows: list[dict[str, Any]], preferred: list[str] | None = None, *, include_dynamic: bool = True
) -> list[str]:
    fixed = ["world", "mind", "institution", "scenario", "mechanism", "n_seeds", "source_dir"]
    dynamic: list[str] = []
    for key in preferred or []:
        if any(key in row for row in rows) and key not in dynamic:
            dynamic.append(key)
    if include_dynamic:
        for row in rows:
            for key in row:
                if key not in fixed and key not in dynamic:
                    dynamic.append(key)
    return [key for key in fixed if any(key in row for row in rows)] + dynamic


def _pricing_rows(results: list[tuple[str, Path]], outputs_dir: Path) -> list[dict[str, Any]]:
    from build_combined_table import combined_rows

    rows: list[dict[str, Any]] = []
    for mind, multiseed_dir in results:
        exploit_name = EXPLOITABILITY_ALIASES.get(mind)
        exploit_dir = outputs_dir / exploit_name if exploit_name else None
        if exploit_dir is not None and not exploit_dir.exists():
            exploit_dir = None
        for source in combined_rows(multiseed_dir, exploit_dir, mind):
            source.update(
                {
                    "world": "pricing_arena",
                    "institution": source["mechanism"],
                    "source_dir": str(multiseed_dir),
                }
            )
            rows.append(source)
    return rows


def build_comparison(
    *,
    world: str,
    results: list[tuple[str, Path]] | None = None,
    group_column: str | None = None,
    outputs_dir: Path = Path("outputs"),
) -> list[dict[str, Any]]:
    schema = WORLD_SCHEMAS[world]
    group = group_column or str(schema["group_column"])
    resolved = results if results is not None else auto_results(world, outputs_dir)
    if not resolved:
        raise FileNotFoundError(f"no result directories found for world={world}")
    if world == "pricing_arena":
        return _pricing_rows(resolved, outputs_dir)

    rows: list[dict[str, Any]] = []
    for mind, directory in resolved:
        aggregate_path = directory / "summary_aggregate.csv"
        if not aggregate_path.exists():
            raise FileNotFoundError(f"missing aggregate: {aggregate_path}")
        sources = read_rows(aggregate_path)
        for source in sources:
            if group not in source:
                raise KeyError(f"{aggregate_path} lacks group column {group!r}")
            out: dict[str, Any] = {
                "world": world,
                "mind": mind,
                "institution": source[group],
                "source_dir": str(directory),
            }
            if "n_seeds" in source:
                out["n_seeds"] = int(float(source["n_seeds"]))
            if "scenario" in source:
                out["scenario"] = source["scenario"]
            if "mechanism" in source:
                out["mechanism"] = source["mechanism"]
            for key, value in source.items():
                if key in {group, "n_seeds", "scenario", "mechanism"}:
                    continue
                if key.endswith(("_mean", "_std", "_ci95_low", "_ci95_high")) or key.startswith("benchmark_"):
                    out[key] = _coerce(value)
            rows.append(out)
    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    preferred: list[str] | None = None,
    *,
    include_dynamic: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = comparison_columns(rows, preferred, include_dynamic=include_dynamic)
    if not fieldnames:
        fieldnames = preferred or [
            "world",
            "mind",
            "baseline_institution",
            "institution",
            "metric",
            "n",
            "mean",
            "std",
            "ci95_low",
            "ci95_high",
            "ci95_half_width",
            "source_dir",
        ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary(values: list[float]) -> dict[str, float | int]:
    from run_multiseed import t_critical_975

    n = len(values)
    value_mean = mean(values)
    value_std = stdev(values) if n > 1 else 0.0
    half = t_critical_975(n - 1) * value_std / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": value_mean,
        "std": value_std,
        "ci95_low": value_mean - half,
        "ci95_high": value_mean + half,
        "ci95_half_width": half,
    }


def _benchmark_value(row: dict[str, str], spec: dict[str, Any]) -> float:
    if "constant" in spec:
        return float(spec["constant"])
    return float(row[str(spec["column"])])


def build_uncertainty(
    *, world: str, results: list[tuple[str, Path]], outputs_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    schema = WORLD_SCHEMAS[world]
    group_column = str(schema["group_column"])
    metrics = list(schema["seed_metrics"])
    benchmark_specs = dict(schema["benchmark_metrics"])
    uncertainty: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []

    for mind, directory in results:
        seed_path = directory / "summary_by_seed.csv"
        if not seed_path.exists():
            continue
        rows = read_rows(seed_path)
        if not rows:
            continue
        if world == "pricing_arena":
            from build_combined_table import PROFIT_BENCHMARKS
            from core.metrics import profit_collusion_index

            for row in rows:
                row["profit_collusion_index"] = str(
                    profit_collusion_index(
                        float(row["profit_total"]),
                        PROFIT_BENCHMARKS["nash_profit_total"],
                        PROFIT_BENCHMARKS["monopoly_profit_total"],
                    )
                )
        seed_col = "seed" if "seed" in rows[0] else "seed_index"
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        by_group_seed: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        for row in rows:
            group = row[group_column]
            for metric in metrics:
                if row.get(metric, "") != "":
                    value = float(row[metric])
                    grouped[(group, metric)].append(value)
                    by_group_seed[(group, metric)][row[seed_col]] = value
            for metric, spec in benchmark_specs.items():
                if row.get(metric, "") == "":
                    continue
                gap = float(row[metric]) - _benchmark_value(row, spec)
                grouped[(group, f"{metric}__benchmark_gap")].append(gap)

        for (group, metric_key), values in sorted(grouped.items()):
            is_gap = metric_key.endswith("__benchmark_gap")
            metric = metric_key.removesuffix("__benchmark_gap")
            stats = _summary(values)
            uncertainty.append(
                {
                    "world": world,
                    "mind": mind,
                    "institution": group,
                    "metric": metric,
                    "estimate_type": "benchmark_gap" if is_gap else "observed",
                    **stats,
                    "source_dir": str(directory),
                }
            )

        if world == "pricing_arena":
            exploit_name = EXPLOITABILITY_ALIASES.get(mind)
            exploit_path = outputs_dir / exploit_name / "summary_by_seed.csv" if exploit_name else None
            if exploit_path is not None and exploit_path.exists():
                exploit_rows = read_rows(exploit_path)
                exploit_seed_col = (
                    "seed" if exploit_rows and "seed" in exploit_rows[0] else "seed_index"
                )
                exploit_by_group_seed: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
                for metric in ("exploitability", "victim_loss", "welfare_damage"):
                    values_by_group: dict[str, list[float]] = defaultdict(list)
                    for row in exploit_rows:
                        if row.get(metric, "") != "":
                            group = row[group_column]
                            value = float(row[metric])
                            values_by_group[group].append(value)
                            exploit_by_group_seed[(group, metric)][row[exploit_seed_col]] = value
                    for group, values in sorted(values_by_group.items()):
                        uncertainty.append(
                            {
                                "world": world,
                                "mind": mind,
                                "institution": group,
                                "metric": metric,
                                "estimate_type": "observed",
                                **_summary(values),
                                "source_dir": str(exploit_path.parent),
                            }
                        )
                        uncertainty.append(
                            {
                                "world": world,
                                "mind": mind,
                                "institution": group,
                                "metric": metric,
                                "estimate_type": "benchmark_gap",
                                **_summary(values),
                                "source_dir": str(exploit_path.parent),
                            }
                        )
                baseline = str(schema["baseline"])
                for group in sorted({row[group_column] for row in exploit_rows}):
                    if group == baseline:
                        continue
                    for metric in ("exploitability", "victim_loss", "welfare_damage"):
                        base = exploit_by_group_seed.get((baseline, metric), {})
                        current = exploit_by_group_seed.get((group, metric), {})
                        common = sorted(set(base) & set(current))
                        if not common:
                            continue
                        paired.append(
                            {
                                "world": world,
                                "mind": mind,
                                "baseline_institution": baseline,
                                "institution": group,
                                "metric": metric,
                                **_summary([current[seed] - base[seed] for seed in common]),
                                "source_dir": str(exploit_path.parent),
                            }
                        )

        baseline = str(schema["baseline"])
        for group in sorted({row[group_column] for row in rows}):
            if group == baseline:
                continue
            for metric in metrics:
                base = by_group_seed.get((baseline, metric), {})
                current = by_group_seed.get((group, metric), {})
                common = sorted(set(base) & set(current))
                if not common:
                    continue
                diffs = [current[seed] - base[seed] for seed in common]
                paired.append(
                    {
                        "world": world,
                        "mind": mind,
                        "baseline_institution": baseline,
                        "institution": group,
                        "metric": metric,
                        **_summary(diffs),
                        "source_dir": str(directory),
                    }
                )
    return uncertainty, paired


def build_paired_mind_effects(
    *, world: str, results: list[tuple[str, Path]], outputs_dir: Path
) -> list[dict[str, Any]]:
    """Pair each learned mind with Q-learning on common institution/seed rows."""
    schema = WORLD_SCHEMAS[world]
    group_column = str(schema["group_column"])
    metrics = list(schema["seed_metrics"])
    values: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    sources = {mind: directory for mind, directory in results}

    for mind, directory in results:
        seed_path = directory / "summary_by_seed.csv"
        if not seed_path.exists():
            continue
        rows = read_rows(seed_path)
        if not rows:
            continue
        if world == "pricing_arena":
            from build_combined_table import PROFIT_BENCHMARKS
            from core.metrics import profit_collusion_index

            for row in rows:
                row["profit_collusion_index"] = str(
                    profit_collusion_index(
                        float(row["profit_total"]),
                        PROFIT_BENCHMARKS["nash_profit_total"],
                        PROFIT_BENCHMARKS["monopoly_profit_total"],
                    )
                )
        seed_col = "seed" if "seed" in rows[0] else "seed_index"
        for row in rows:
            for metric in metrics:
                if row.get(metric, "") != "":
                    values[(mind, row[group_column], metric)][row[seed_col]] = float(row[metric])

        if world == "pricing_arena":
            exploit_name = EXPLOITABILITY_ALIASES.get(mind)
            exploit_path = outputs_dir / exploit_name / "summary_by_seed.csv" if exploit_name else None
            if exploit_path is not None and exploit_path.exists():
                exploit_rows = read_rows(exploit_path)
                if exploit_rows:
                    seed_col = "seed" if "seed" in exploit_rows[0] else "seed_index"
                    for row in exploit_rows:
                        for metric in ("exploitability", "victim_loss", "welfare_damage"):
                            if row.get(metric, "") != "":
                                values[(mind, row[group_column], metric)][row[seed_col]] = float(
                                    row[metric]
                                )

    baseline_mind = "q_learning"
    paired: list[dict[str, Any]] = []
    groups_metrics = sorted({(group, metric) for _, group, metric in values})
    for mind, directory in results:
        if mind == baseline_mind:
            continue
        for group, metric in groups_metrics:
            base = values.get((baseline_mind, group, metric), {})
            current = values.get((mind, group, metric), {})
            common = sorted(set(base) & set(current))
            if not common:
                continue
            paired.append(
                {
                    "world": world,
                    "baseline_mind": baseline_mind,
                    "mind": mind,
                    "institution": group,
                    "metric": metric,
                    **_summary([current[seed] - base[seed] for seed in common]),
                    "baseline_source_dir": str(sources[baseline_mind]),
                    "source_dir": str(directory),
                }
            )
    return paired


def default_output(world: str, outputs_dir: Path) -> Path:
    return outputs_dir / DEFAULT_OUTPUT_DIRS[world] / "mind_comparison.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a canonical world x mind comparison table.")
    parser.add_argument("--world", required=True, choices=sorted(WORLD_SCHEMAS))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--result", action="append", type=_parse_result, default=None, help="mind:path")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-uncertainty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results = args.result if args.result is not None else auto_results(args.world, args.outputs_dir)
    rows = build_comparison(
        world=args.world,
        results=results,
        group_column=args.group_column,
        outputs_dir=args.outputs_dir,
    )
    output = args.output or default_output(args.world, args.outputs_dir)
    preferred = list(WORLD_SCHEMAS[args.world]["aggregate_metrics"])
    write_csv(output, rows, preferred, include_dynamic=False)
    minds = sorted({str(row["mind"]) for row in rows})
    print(f"Wrote: {output} — {len(rows)} rows; minds={','.join(minds)}")

    missing_metrics = [metric for metric in preferred if not any(metric in row for row in rows)]
    if missing_metrics:
        print(f"WARNING: missing expected metrics: {','.join(missing_metrics)}")

    # Random is a historical Pricing Arena reference, not part of the five-mind
    # capability ladder.  Report only missing ladder minds as incomplete.
    missing = [mind for mind in REQUIRED_LADDER_MINDS if mind not in minds]
    if missing:
        print(f"WARNING: missing expected minds: {','.join(missing)}")
    if not args.skip_uncertainty:
        uncertainty, paired = build_uncertainty(world=args.world, results=results, outputs_dir=args.outputs_dir)
        paired_minds = build_paired_mind_effects(
            world=args.world, results=results, outputs_dir=args.outputs_dir
        )
        uncertainty_path = output.with_name("mind_comparison_uncertainty.csv")
        paired_path = output.with_name("paired_institution_effects.csv")
        paired_minds_path = output.with_name("paired_mind_effects.csv")
        write_csv(uncertainty_path, uncertainty)
        write_csv(paired_path, paired)
        write_csv(paired_minds_path, paired_minds)
        print(f"Wrote: {uncertainty_path} — {len(uncertainty)} rows")
        print(f"Wrote: {paired_path} — {len(paired)} rows")
        print(f"Wrote: {paired_minds_path} — {len(paired_minds)} rows")


if __name__ == "__main__":
    main()
