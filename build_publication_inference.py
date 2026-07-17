from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import mpmath


WORLDS = ["pricing_arena", "resource_island", "auction_house", "public_goods", "labor_market"]
CONFIRMATORY_METRICS = {
    "pricing_arena": {"welfare", "profit_collusion_index", "exploitability"},
    "resource_island": {"welfare", "survival_rate", "trade_count"},
    "auction_house": {"revenue", "welfare", "allocative_efficiency", "ex_post_regret_mean"},
    "public_goods": {"sustainability", "welfare", "collapse_rate"},
    "labor_market": {"total_welfare", "stability", "truthful_report_rate", "manipulation_gain_mean"},
}


def student_t_two_sided_p(mean: float, std: float, n: int) -> float:
    if n < 2:
        return float("nan")
    if std == 0.0:
        return 1.0 if mean == 0.0 else 0.0
    statistic = abs(mean) / (std / math.sqrt(n))
    degrees = n - 1
    x = degrees / (degrees + statistic * statistic)
    return float(mpmath.betainc(degrees / 2.0, 0.5, 0, x, regularized=True))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    count = len(p_values)
    if count == 0:
        return []
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [1.0] * count
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, p_values[original_index] * count / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def build_rows(outputs_dir: Path) -> list[dict[str, Any]]:
    protocol_path = outputs_dir / "cross_world_synthesis" / "protocol_comparability_report.json"
    protocols = json.loads(protocol_path.read_text())
    rows: list[dict[str, Any]] = []
    for world in WORLDS:
        root = outputs_dir / f"{world}_phase3_full"
        for effect_type, filename in (
            ("institution_vs_baseline", "paired_institution_effects.csv"),
            ("mind_vs_q_learning", "paired_mind_effects.csv"),
        ):
            path = root / filename
            if not path.exists():
                continue
            for source in read_rows(path):
                metric = source["metric"]
                mean = float(source["mean"])
                std = float(source["std"])
                n = int(float(source["n"]))
                protocol_valid = not (
                    effect_type == "mind_vs_q_learning"
                    and not protocols[world]["cross_mind_capability_claims_valid"]
                )
                row = {
                    "effect_type": effect_type,
                    "world": world,
                    "claim_class": (
                        "confirmatory" if metric in CONFIRMATORY_METRICS[world] else "exploratory"
                    ),
                    "baseline": source.get("baseline_institution") or source.get("baseline_mind", ""),
                    "comparison": source.get("institution") if effect_type == "institution_vs_baseline" else source.get("mind"),
                    "mind": source.get("mind", ""),
                    "institution": source.get("institution", ""),
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "ci95_low": float(source["ci95_low"]),
                    "ci95_high": float(source["ci95_high"]),
                    "p_value": student_t_two_sided_p(mean, std, n),
                    "protocol_valid": protocol_valid,
                    "source_dir": source.get("source_dir", ""),
                }
                rows.append(row)

    families: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["protocol_valid"] and math.isfinite(row["p_value"]):
            families[(row["effect_type"], row["world"], row["claim_class"], row["metric"])].append(index)
    for indices in families.values():
        adjusted = benjamini_hochberg([rows[index]["p_value"] for index in indices])
        for index, q_value in zip(indices, adjusted):
            rows[index]["q_value"] = q_value
            rows[index]["significant_fdr_05"] = q_value <= 0.05
    for row in rows:
        row.setdefault("q_value", float("nan"))
        row.setdefault("significant_fdr_05", False)
    return rows


FIELDS = [
    "effect_type",
    "world",
    "claim_class",
    "baseline",
    "comparison",
    "mind",
    "institution",
    "metric",
    "n",
    "mean",
    "std",
    "ci95_low",
    "ci95_high",
    "p_value",
    "q_value",
    "significant_fdr_05",
    "protocol_valid",
    "source_dir",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    outputs_dir = Path("outputs")
    save_dir = outputs_dir / "publication_inference"
    rows = build_rows(outputs_dir)
    write_csv(save_dir / "publication_inference_all.csv", rows)
    confirmatory = [
        row
        for row in rows
        if row["claim_class"] == "confirmatory"
        and row["protocol_valid"]
        and row["significant_fdr_05"]
    ]
    confirmatory.sort(key=lambda row: (row["world"], row["effect_type"], row["metric"], row["q_value"]))
    write_csv(save_dir / "thesis_confirmatory_results.csv", confirmatory)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[row["world"]]["tested"] += int(row["protocol_valid"])
        counts[row["world"]]["confirmatory_fdr_05"] += int(
            row["protocol_valid"]
            and row["claim_class"] == "confirmatory"
            and row["significant_fdr_05"]
        )
        counts[row["world"]]["excluded_protocol_mismatch"] += int(not row["protocol_valid"])
    lines = [
        "# Publication Inference Summary",
        "",
        "Benjamini-Hochberg correction is applied separately within each world, effect type, claim class, and metric family.",
        "Resource Island cross-mind effects are excluded whenever the protocol-comparability audit fails; within-mind institution effects remain eligible.",
        "",
        "| World | Valid tests | Confirmatory FDR < 0.05 | Excluded protocol-mismatch rows |",
        "|---|---:|---:|---:|",
    ]
    for world in WORLDS:
        item = counts[world]
        lines.append(
            f"| {world} | {item['tested']} | {item['confirmatory_fdr_05']} | {item['excluded_protocol_mismatch']} |"
        )
    (save_dir / "publication_inference_summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote: {save_dir / 'publication_inference_all.csv'} — {len(rows)} rows")
    print(f"Wrote: {save_dir / 'thesis_confirmatory_results.csv'} — {len(confirmatory)} rows")
    print(f"Wrote: {save_dir / 'publication_inference_summary.md'}")


if __name__ == "__main__":
    main()
