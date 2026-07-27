from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import numpy as np

from build_world_mind_comparison import EXPLOITABILITY_ALIASES, auto_results, read_rows
from core.metrics import finite_mean, profit_collusion_index
from run_multiseed import t_critical_975
from world_metric_schemas import WORLD_SCHEMAS


WORLDS = ("pricing_arena", "resource_island", "auction_house", "public_goods", "labor_market")
THESIS_MINDS = ("q_learning", "dqn", "ppo", "independent_dqn", "centralized_critic")

PRICING_PROFIT_BENCHMARKS = {
    "nash_profit_total": 149.9796959107128,
    "monopoly_profit_total": 603.4324325204249,
}

OUTPUT_FIELDS = (
    "world",
    "audit_type",
    "mind",
    "baseline",
    "institution",
    "metric",
    "n",
    "mean",
    "median",
    "std",
    "ci95_low",
    "ci95_high",
    "positive_share",
    "negative_share",
    "min",
    "max",
    "classification",
    "claim_boundary",
    "source_dir",
)

CONFLICT_FIELDS = (
    "world",
    "conflict_type",
    "mind",
    "institution",
    "primary_metric",
    "primary_effect",
    "secondary_metric",
    "secondary_effect",
    "classification",
    "interpretation",
)

TRACE_FIELDS = (
    "world",
    "mind",
    "institution",
    "component",
    "trace_value",
    "full_mean",
    "trace_sign",
    "full_sign",
    "sign_agrees",
    "interpretation",
)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary(values: list[float]) -> dict[str, Any]:
    xs = [float(value) for value in values if _finite(value)]
    if not xs:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "positive_share": float("nan"),
            "negative_share": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    n = len(xs)
    value_mean = mean(xs)
    value_std = stdev(xs) if n > 1 else 0.0
    half = t_critical_975(n - 1) * value_std / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": value_mean,
        "median": median(xs),
        "std": value_std,
        "ci95_low": value_mean - half,
        "ci95_high": value_mean + half,
        "positive_share": sum(value > 0.0 for value in xs) / n,
        "negative_share": sum(value < 0.0 for value in xs) / n,
        "min": min(xs),
        "max": max(xs),
    }


def _sign(value: float, tolerance: float = 1e-12) -> int:
    if not _finite(value) or abs(float(value)) <= tolerance:
        return 0
    return 1 if float(value) > 0.0 else -1


def _classify(stats: dict[str, Any]) -> str:
    n = int(stats["n"])
    if n == 0:
        return "missing"
    mean_value = float(stats["mean"])
    median_value = float(stats["median"])
    ci_low = float(stats["ci95_low"])
    ci_high = float(stats["ci95_high"])
    positive_share = float(stats["positive_share"])
    if n < 5:
        return "smoke_scale"
    if ci_low > 0.0 and positive_share >= 0.75:
        return "robust_positive"
    if ci_high < 0.0 and positive_share <= 0.25:
        return "robust_negative"
    if _sign(mean_value) != 0 and _sign(median_value) != 0 and _sign(mean_value) != _sign(median_value):
        return "outlier_sensitive"
    if 0.35 <= positive_share <= 0.65:
        return "mixed_seed_sign"
    if ci_low <= 0.0 <= ci_high:
        return "ci_crosses_zero"
    return "directional_weak"


def _claim_boundary(world: str, audit_type: str, classification: str) -> str:
    if classification in {"robust_positive", "robust_negative"}:
        return "full-run paired effect is seed-robust enough for a directional thesis claim"
    if classification == "outlier_sensitive":
        return "mean is not enough; inspect per-seed/outlier mechanism before writing a sign claim"
    if classification == "mixed_seed_sign":
        return "seed heterogeneity is large; write as fragile or conditional, not stable"
    if classification == "ci_crosses_zero":
        return "direction is suggestive but not CI-clean; avoid strong claims"
    if classification == "smoke_scale":
        return "smoke-scale only; execution evidence, not thesis evidence"
    return f"{world} {audit_type} needs manual interpretation before being cited"


def _group_seed_values(
    rows: list[dict[str, str]], group_column: str, metrics: list[str]
) -> dict[tuple[str, str], dict[str, float]]:
    values: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    if not rows:
        return values
    seed_col = "seed" if "seed" in rows[0] else "seed_index"
    for row in rows:
        group = row.get(group_column, "")
        seed = row.get(seed_col, "")
        if not group or seed == "":
            continue
        for metric in metrics:
            if row.get(metric, "") != "":
                values[(group, metric)][seed] = _float(row[metric])
    return values


def _pricing_seed_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    for row in rows:
        if row.get("profit_total", ""):
            row["profit_collusion_index"] = str(
                profit_collusion_index(
                    _float(row["profit_total"]),
                    PRICING_PROFIT_BENCHMARKS["nash_profit_total"],
                    PRICING_PROFIT_BENCHMARKS["monopoly_profit_total"],
                )
            )
    return rows


def _audit_row(
    *,
    world: str,
    audit_type: str,
    mind: str,
    baseline: str,
    institution: str,
    metric: str,
    values: list[float],
    source_dir: Path | str,
) -> dict[str, Any]:
    stats = _summary(values)
    classification = _classify(stats)
    return {
        "world": world,
        "audit_type": audit_type,
        "mind": mind,
        "baseline": baseline,
        "institution": institution,
        "metric": metric,
        **stats,
        "classification": classification,
        "claim_boundary": _claim_boundary(world, audit_type, classification),
        "source_dir": str(source_dir),
    }


def paired_institution_audits(outputs_dir: Path) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for world in WORLDS:
        schema = WORLD_SCHEMAS[world]
        group_column = str(schema["group_column"])
        baseline = str(schema["baseline"])
        metrics = list(schema["seed_metrics"])
        for mind, directory in auto_results(world, outputs_dir):
            if mind not in THESIS_MINDS:
                continue
            rows = _read_csv(directory / "summary_by_seed.csv")
            if world == "pricing_arena":
                rows = _pricing_seed_rows(rows)
            values = _group_seed_values(rows, group_column, metrics)
            groups = sorted({group for group, _metric in values})
            for group in groups:
                if group == baseline:
                    continue
                for metric in metrics:
                    base = values.get((baseline, metric), {})
                    current = values.get((group, metric), {})
                    common = sorted(set(base) & set(current))
                    if not common:
                        continue
                    audits.append(
                        _audit_row(
                            world=world,
                            audit_type="paired_institution_effect",
                            mind=mind,
                            baseline=baseline,
                            institution=group,
                            metric=metric,
                            values=[current[seed] - base[seed] for seed in common],
                            source_dir=directory,
                        )
                    )
            if world == "pricing_arena":
                audits.extend(_pricing_exploitability_audits(outputs_dir, mind, baseline))
    return audits


def _pricing_exploitability_audits(outputs_dir: Path, mind: str, baseline: str) -> list[dict[str, Any]]:
    alias = EXPLOITABILITY_ALIASES.get(mind)
    if alias is None:
        return []
    directory = outputs_dir / alias
    rows = _read_csv(directory / "summary_by_seed.csv")
    values = _group_seed_values(rows, "mechanism", ["exploitability", "victim_loss", "welfare_damage"])
    audits: list[dict[str, Any]] = []
    groups = sorted({group for group, _metric in values})
    for group in groups:
        if group == baseline:
            continue
        for metric in ("exploitability", "victim_loss", "welfare_damage"):
            base = values.get((baseline, metric), {})
            current = values.get((group, metric), {})
            common = sorted(set(base) & set(current))
            if common:
                audits.append(
                    _audit_row(
                        world="pricing_arena",
                        audit_type="paired_institution_exploitability_effect",
                        mind=mind,
                        baseline=baseline,
                        institution=group,
                        metric=metric,
                        values=[current[seed] - base[seed] for seed in common],
                        source_dir=directory,
                    )
                )
    return audits


def benchmark_gap_audits(outputs_dir: Path) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for world in WORLDS:
        schema = WORLD_SCHEMAS[world]
        benchmark_specs = dict(schema["benchmark_metrics"])
        if not benchmark_specs:
            continue
        group_column = str(schema["group_column"])
        for mind, directory in auto_results(world, outputs_dir):
            if mind not in THESIS_MINDS:
                continue
            rows = _read_csv(directory / "summary_by_seed.csv")
            if not rows:
                continue
            for metric, spec in benchmark_specs.items():
                grouped: dict[str, list[float]] = defaultdict(list)
                for row in rows:
                    if row.get(metric, "") == "":
                        continue
                    benchmark = float(spec["constant"]) if "constant" in spec else _float(row.get(str(spec["column"])))
                    if not _finite(benchmark):
                        continue
                    grouped[row[group_column]].append(_float(row[metric]) - benchmark)
                for group, values in grouped.items():
                    audits.append(
                        _audit_row(
                            world=world,
                            audit_type="benchmark_gap",
                            mind=mind,
                            baseline="benchmark",
                            institution=group,
                            metric=metric,
                            values=values,
                            source_dir=directory,
                        )
                    )
    return audits


def _audit_lookup(audits: list[dict[str, Any]]) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    return {
        (
            str(row["world"]),
            str(row["audit_type"]),
            str(row["mind"]),
            str(row["institution"]),
            str(row["metric"]),
        ): row
        for row in audits
    }


def conflict_audits(audits: list[dict[str, Any]], outputs_dir: Path) -> list[dict[str, Any]]:
    lookup = _audit_lookup(audits)
    conflicts: list[dict[str, Any]] = []
    for mind in THESIS_MINDS:
        profit = lookup.get(("pricing_arena", "paired_institution_effect", mind, "price_cap", "profit_total"))
        exploit = lookup.get(
            ("pricing_arena", "paired_institution_exploitability_effect", mind, "price_cap", "exploitability")
        )
        if profit and exploit and _float(profit["mean"]) > 0.0 and _float(exploit["mean"]) < 0.0:
            conflicts.append(
                {
                    "world": "pricing_arena",
                    "conflict_type": "guardrail_metric_reversal",
                    "mind": mind,
                    "institution": "price_cap",
                    "primary_metric": "profit_total",
                    "primary_effect": profit["mean"],
                    "secondary_metric": "exploitability",
                    "secondary_effect": exploit["mean"],
                    "classification": "metric_conflict",
                    "interpretation": "price cap reduces exploitability while profit rises; write as metric-dependent institutional effect",
                }
            )

    for mind in THESIS_MINDS:
        for institution in ("contribution_matching", "public_goods_penalty", "reputation", "tax_schedule"):
            welfare = lookup.get(("public_goods", "paired_institution_effect", mind, institution, "welfare"))
            sustain = lookup.get(("public_goods", "paired_institution_effect", mind, institution, "sustainability"))
            contrib = lookup.get(("public_goods", "paired_institution_effect", mind, institution, "contribution_total"))
            if welfare and sustain and _float(welfare["mean"]) > 0.0 and _float(sustain["mean"]) <= 0.0:
                conflicts.append(
                    {
                        "world": "public_goods",
                        "conflict_type": "reward_state_metric_lie",
                        "mind": mind,
                        "institution": institution,
                        "primary_metric": "welfare",
                        "primary_effect": welfare["mean"],
                        "secondary_metric": "sustainability",
                        "secondary_effect": sustain["mean"],
                        "classification": "metric_conflict",
                        "interpretation": "welfare/reward rises without a sustainability improvement; do not call it a commons solution",
                    }
                )
            if welfare and contrib and _float(welfare["mean"]) > 0.0 and _float(contrib["mean"]) <= 0.0:
                conflicts.append(
                    {
                        "world": "public_goods",
                        "conflict_type": "reward_contribution_metric_lie",
                        "mind": mind,
                        "institution": institution,
                        "primary_metric": "welfare",
                        "primary_effect": welfare["mean"],
                        "secondary_metric": "contribution_total",
                        "secondary_effect": contrib["mean"],
                        "classification": "metric_conflict",
                        "interpretation": "welfare/reward rises without contribution increasing; inspect reward accounting",
                    }
                )

    mind_effects_path = outputs_dir / "labor_market_phase3_full" / "paired_mind_effects.csv"
    mind_effects = _read_csv(mind_effects_path)
    labor_by_mind_metric = {
        (row["mind"], row["metric"]): _float(row["mean"])
        for row in mind_effects
        if row.get("institution") == "deferred_acceptance"
    }
    for mind in THESIS_MINDS:
        welfare = labor_by_mind_metric.get((mind, "total_welfare"))
        stability = labor_by_mind_metric.get((mind, "stability"))
        truth = labor_by_mind_metric.get((mind, "truthful_report_rate"))
        if welfare is not None and stability is not None and welfare > 0.0 and stability < 0.0:
            conflicts.append(
                {
                    "world": "labor_market",
                    "conflict_type": "welfare_stability_tradeoff",
                    "mind": mind,
                    "institution": "deferred_acceptance",
                    "primary_metric": "total_welfare_vs_q_learning",
                    "primary_effect": welfare,
                    "secondary_metric": "stability_vs_q_learning",
                    "secondary_effect": stability,
                    "classification": "metric_conflict",
                    "interpretation": "welfare improves while stability falls; this is a mechanism-design tradeoff, not a clean improvement",
                }
            )
        if welfare is not None and truth is not None and welfare > 0.0 and truth < 0.0:
            conflicts.append(
                {
                    "world": "labor_market",
                    "conflict_type": "welfare_truthfulness_tradeoff",
                    "mind": mind,
                    "institution": "deferred_acceptance",
                    "primary_metric": "total_welfare_vs_q_learning",
                    "primary_effect": welfare,
                    "secondary_metric": "truthfulness_vs_q_learning",
                    "secondary_effect": truth,
                    "classification": "metric_conflict",
                    "interpretation": "welfare improves while truthful reporting falls; do not treat welfare alone as mechanism success",
                }
            )

    auction_pairs = _read_csv(outputs_dir / "auction_house_phase3_full" / "paired_institution_effects.csv")
    auction_lookup = {(row["mind"], row["institution"], row["metric"]): _float(row["mean"]) for row in auction_pairs}
    for mind in THESIS_MINDS:
        for institution in ("first_price", "second_price_reserve", "clock", "second_price_public_signal", "second_price_noisy_signal"):
            revenue = auction_lookup.get((mind, institution, "revenue"))
            efficiency = auction_lookup.get((mind, institution, "allocative_efficiency"))
            regret = auction_lookup.get((mind, institution, "ex_post_regret_mean"))
            if revenue is not None and efficiency is not None and revenue > 0.0 and efficiency < 0.0:
                conflicts.append(
                    {
                        "world": "auction_house",
                        "conflict_type": "revenue_efficiency_tradeoff",
                        "mind": mind,
                        "institution": institution,
                        "primary_metric": "revenue_vs_second_price",
                        "primary_effect": revenue,
                        "secondary_metric": "allocative_efficiency_vs_second_price",
                        "secondary_effect": efficiency,
                        "classification": "metric_conflict",
                        "interpretation": "seller revenue rises while efficiency falls; write as auction-design tradeoff",
                    }
                )
            if revenue is not None and regret is not None and revenue > 0.0 and regret > 0.0:
                conflicts.append(
                    {
                        "world": "auction_house",
                        "conflict_type": "revenue_regret_tradeoff",
                        "mind": mind,
                        "institution": institution,
                        "primary_metric": "revenue_vs_second_price",
                        "primary_effect": revenue,
                        "secondary_metric": "regret_vs_second_price",
                        "secondary_effect": regret,
                        "classification": "metric_conflict",
                        "interpretation": "seller revenue rises while bidder regret rises; do not call it unambiguously better",
                    }
                )

    conflicts.extend(resource_activation_conflicts(outputs_dir))
    return conflicts


def resource_activation_conflicts(outputs_dir: Path) -> list[dict[str, Any]]:
    rows = _read_csv(outputs_dir / "resource_island_v1_phase3_full" / "mind_comparison.csv")
    out: list[dict[str, Any]] = []
    for row in rows:
        mind = row["mind"]
        institution = row["institution"]
        trade = _float(row.get("trade_count_mean"))
        attempts = _float(row.get("trade_attempt_count_mean"))
        blocks = _float(row.get("trade_institution_blocked_count_mean"))
        props = _float(row.get("property_opportunities_mean"))
        if institution == "trade_price_controls" and blocks > 0.0 and trade <= 0.05:
            out.append(
                {
                    "world": "resource_island",
                    "conflict_type": "binding_trade_control",
                    "mind": mind,
                    "institution": institution,
                    "primary_metric": "trade_institution_blocked_count_mean",
                    "primary_effect": blocks,
                    "secondary_metric": "trade_count_mean",
                    "secondary_effect": trade,
                    "classification": "activation_confirmed",
                    "interpretation": "price-control institution binds and suppresses successful trade under v1",
                }
            )
        if institution == "property_rights" and props > 0.0:
            out.append(
                {
                    "world": "resource_island",
                    "conflict_type": "property_pressure_confirmed",
                    "mind": mind,
                    "institution": institution,
                    "primary_metric": "property_opportunities_mean",
                    "primary_effect": props,
                    "secondary_metric": "trade_attempt_count_mean",
                    "secondary_effect": attempts,
                    "classification": "activation_confirmed",
                    "interpretation": "property-rights opportunities exist; v1 no longer silently under-tests the institution",
                }
            )
        if institution == "none" and trade <= 0.1 and attempts > 0.0:
            out.append(
                {
                    "world": "resource_island",
                    "conflict_type": "trade_attempt_success_gap",
                    "mind": mind,
                    "institution": institution,
                    "primary_metric": "trade_attempt_count_mean",
                    "primary_effect": attempts,
                    "secondary_metric": "trade_count_mean",
                    "secondary_effect": trade,
                    "classification": "activation_gap",
                    "interpretation": "agents attempt trade but do not complete it; inspect coordination/inventory constraints",
                }
            )
    return out


def trace_alignment_audits(audits: list[dict[str, Any]], trace_path: Path) -> list[dict[str, Any]]:
    if not trace_path.exists():
        return []
    lookup = _audit_lookup(audits)
    out: list[dict[str, Any]] = []
    for row in _read_csv(trace_path):
        institution = row.get("institution", "")
        component = row.get("component", "")
        if "_vs_" not in institution or not component.startswith("delta_"):
            continue
        baseline = institution.split("_vs_", 1)[1]
        current = institution.split("_vs_", 1)[0]
        metric = component.removeprefix("delta_")
        key = (row["world"], "paired_institution_effect", row["mind"], current, metric)
        full = lookup.get(key)
        if full is None:
            continue
        trace_value = _float(row["value"])
        full_mean = _float(full["mean"])
        trace_sign = _sign(trace_value)
        full_sign = _sign(full_mean)
        out.append(
            {
                "world": row["world"],
                "mind": row["mind"],
                "institution": institution,
                "component": component,
                "trace_value": trace_value,
                "full_mean": full_mean,
                "trace_sign": trace_sign,
                "full_sign": full_sign,
                "sign_agrees": int(trace_sign == full_sign),
                "interpretation": "short trace agrees with full-run sign"
                if trace_sign == full_sign
                else "short trace disagrees with full-run sign; needs budget/seed audit before mechanism prose",
            }
        )
    return out


def write_markdown(
    path: Path,
    *,
    audits: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    high_risk = [
        row
        for row in audits
        if row["classification"] in {"outlier_sensitive", "mixed_seed_sign", "ci_crosses_zero"}
    ]
    trace_mismatches = [row for row in trace_rows if int(row["sign_agrees"]) == 0]
    lines = [
        "# Cross-World Claim Audit Suite",
        "",
        "Generated by `run_claim_audit_suite.py`.",
        "",
        "This audit applies the Pricing-Arena contradiction standard across the existing full-run outputs: paired seed effects, benchmark gaps, metric-conflict detection, activation checks, and trace/full-run sign agreement where traces exist.",
        "",
        "## Summary",
        "",
        f"- Paired/benchmark audit rows: `{len(audits)}`",
        f"- Metric/activation conflict rows: `{len(conflicts)}`",
        f"- Trace/full-run alignment rows: `{len(trace_rows)}`",
        f"- Trace/full-run sign mismatches: `{len(trace_mismatches)}`",
        f"- Seed-risk audit rows: `{len(high_risk)}`",
        "",
        "## Highest-Priority Warnings",
        "",
    ]
    if not trace_mismatches and not high_risk:
        lines.append("- No high-priority warnings were generated by the automatic rules.")
    else:
        for row in trace_mismatches[:20]:
            lines.append(
                f"- Trace mismatch: `{row['world']}` `{row['mind']}` `{row['institution']}` `{row['component']}` "
                f"trace=`{_float(row['trace_value']):.4g}` full=`{_float(row['full_mean']):.4g}`."
            )
        for row in high_risk[:20]:
            lines.append(
                f"- Seed risk: `{row['world']}` `{row['mind']}` `{row['institution']}` `{row['metric']}` "
                f"classification=`{row['classification']}`, mean=`{_float(row['mean']):.4g}`, "
                f"positive_share=`{_float(row['positive_share']):.2f}`."
            )
    lines.extend(["", "## Metric Conflicts / Activation Findings", ""])
    for row in conflicts[:80]:
        lines.append(
            f"- `{row['world']}` `{row['mind']}` `{row['institution']}`: {row['interpretation']} "
            f"({row['primary_metric']}=`{_float(row['primary_effect']):.4g}`, "
            f"{row['secondary_metric']}=`{_float(row['secondary_effect']):.4g}`)."
        )
    lines.extend(
        [
            "",
            "## Safe Claim Rule",
            "",
            "A claim is thesis-ready only when the direction is robust or when the prose explicitly labels it as mixed, outlier-sensitive, accounting-only, activation-confirmed, or trace/full-run divergent.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Path]:
    start = time.time()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    audits = paired_institution_audits(args.outputs_dir)
    audits.extend(benchmark_gap_audits(args.outputs_dir))
    conflicts = conflict_audits(audits, args.outputs_dir)
    trace_rows = trace_alignment_audits(
        audits, args.outputs_dir / "mechanism_traces_local_full" / "mechanism_decomposition.csv"
    )

    audit_path = args.save_dir / "claim_audit_summary.csv"
    conflict_path = args.save_dir / "metric_conflicts.csv"
    trace_path = args.save_dir / "trace_alignment.csv"
    report_path = args.save_dir / "claim_audit_report.md"
    manifest_path = args.save_dir / "manifest.json"
    _write_csv(audit_path, audits, OUTPUT_FIELDS)
    _write_csv(conflict_path, conflicts, CONFLICT_FIELDS)
    _write_csv(trace_path, trace_rows, TRACE_FIELDS)
    write_markdown(report_path, audits=audits, conflicts=conflicts, trace_rows=trace_rows)
    manifest_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.time() - start,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "outputs_dir": str(args.outputs_dir),
                "save_dir": str(args.save_dir),
                "audit_rows": len(audits),
                "conflict_rows": len(conflicts),
                "trace_alignment_rows": len(trace_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "claim_audit_summary": audit_path,
        "metric_conflicts": conflict_path,
        "trace_alignment": trace_path,
        "report": report_path,
        "manifest": manifest_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit full-run thesis claims across all worlds and minds.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/claim_audit_suite"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    outputs = run(parse_args(argv))
    for path in outputs.values():
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
