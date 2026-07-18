from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from .build_gap_table import WORLD_OBLIGATIONS
from .fill_paper_cards import parse_markdown_sections


@dataclass(frozen=True)
class Requirement:
    world: str
    category: str
    obligation: str
    code_paths: tuple[str, ...] = ()
    output_paths: tuple[str, ...] = ()
    required_columns: tuple[str, ...] = ()
    required_terms: tuple[str, ...] = ()


@dataclass
class AuditRow:
    world: str
    category: str
    obligation: str
    status: str
    code_evidence: str
    output_evidence: str
    missing: str


WORLD_REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement(
        "pricing_arena",
        "benchmark",
        "Report Nash and joint-profit price benchmarks.",
        code_paths=("worlds/pricing_arena/benchmarks.py",),
        output_paths=("outputs/full_v0_multiseed/summary_aggregate.csv",),
        required_columns=("nash_price", "monopoly_price"),
        required_terms=("nash", "monopoly"),
    ),
    Requirement(
        "pricing_arena",
        "metrics",
        "Report price-normalized and profit-normalized collusion, exploitability, welfare, price, and profit.",
        code_paths=("core/metrics.py", "build_combined_table.py"),
        output_paths=("outputs/phase3_full/mind_comparison.csv",),
        required_columns=(
            "collusion_index_mean",
            "profit_collusion_index_mean",
            "exploitability_mean",
            "welfare_mean",
            "avg_price_mean",
            "profit_total_mean",
        ),
        required_terms=("profit_collusion_index", "collusion_index", "exploitability"),
    ),
    Requirement(
        "resource_island",
        "activation",
        "Report trade/property activation before interpreting Resource Island institutions.",
        code_paths=("worlds/resource_island/env.py", "worlds/resource_island/training.py"),
        output_paths=("outputs/resource_island_v1_full/summary_aggregate.csv",),
        required_columns=(
            "trade_count_mean",
            "trade_attempt_count_mean",
            "property_opportunities_mean",
            "trade_institution_blocked_count_mean",
        ),
        required_terms=(
            "property_opportunities",
            "trade_institution_blocked_count",
            "trade_food_units",
            "trade_wood_units",
        ),
    ),
    Requirement(
        "resource_island",
        "benchmark",
        "Provide oracle/greedy gather benchmarks for scale and sanity.",
        code_paths=("worlds/resource_island/benchmarks.py",),
        output_paths=("outputs/resource_island_v1_full/summary_aggregate.csv",),
        required_terms=("efficient_gather", "greedy"),
    ),
    Requirement(
        "auction_house",
        "benchmark",
        "Compare learned bidding to truthful second-price, shaded first-price, and reserve benchmarks.",
        code_paths=("worlds/auction_house/benchmarks.py",),
        output_paths=("outputs/auction_house_full/summary_aggregate.csv",),
        required_columns=(
            "truthful_bid_distance_mean_mean",
            "first_price_shading_distance_mean_mean",
            "ex_post_regret_mean_mean",
        ),
        required_terms=("truthful", "first_price", "reserve", "regret"),
    ),
    Requirement(
        "auction_house",
        "metrics",
        "Report revenue, allocative efficiency, welfare/surplus, regret, and bid-shading diagnostics.",
        code_paths=("worlds/auction_house/env.py", "worlds/auction_house/training.py"),
        output_paths=("outputs/auction_house_phase3_full/mind_comparison.csv",),
        required_columns=(
            "revenue_mean",
            "allocative_efficiency_mean",
            "welfare_mean",
            "ex_post_regret_mean_mean",
            "underbid_rate_mean",
        ),
        required_terms=("allocative_efficiency", "underbid", "overbid", "regret"),
    ),
    Requirement(
        "public_goods",
        "benchmark",
        "Compare learned commons behavior to free-rider and social-optimum brackets.",
        code_paths=("worlds/public_goods/benchmarks.py",),
        output_paths=("outputs/public_goods_full/summary_aggregate.csv",),
        required_terms=("free_rider", "social_optimum"),
    ),
    Requirement(
        "public_goods",
        "metrics",
        "Separate state-changing institutions from reward/accounting-only effects.",
        code_paths=("validate_public_goods_effects.py", "worlds/public_goods/training.py"),
        output_paths=(
            "outputs/public_goods_full/summary_aggregate.csv",
            "outputs/public_goods_full/institution_effect_validation.json",
        ),
        required_columns=(
            "sustainability_mean",
            "contribution_total_mean",
            "extraction_total_mean",
            "welfare_mean",
            "tax_revenue_mean",
        ),
        required_terms=("state_changed", "reward_changed", "sustainability"),
    ),
    Requirement(
        "labor_market",
        "benchmark",
        "Verify deferred-acceptance stability and proposing-side strategy-proofness cases.",
        code_paths=("worlds/labor_market/benchmarks.py", "run_labor_market_benchmark_cases.py"),
        output_paths=("outputs/labor_market_benchmark_cases.json",),
        required_terms=("blocking", "deferred", "strategy", "worker_report_gains"),
    ),
    Requirement(
        "labor_market",
        "metrics",
        "Report match rate, stability, truthfulness, welfare, and manipulation diagnostics.",
        code_paths=("worlds/labor_market/env.py", "worlds/labor_market/training.py"),
        output_paths=("outputs/labor_market_phase3_full/mind_comparison.csv",),
        required_columns=(
            "match_rate_mean",
            "stability_mean",
            "truthful_report_rate_mean",
            "total_welfare_mean",
        ),
        required_terms=("manipulation", "truthful", "blocking"),
    ),
)


def _read_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except UnicodeDecodeError:
        return ""


def _csv_columns(path: Path) -> set[str]:
    if not path.exists() or path.suffix.lower() != ".csv":
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            return set(next(reader))
        except StopIteration:
            return set()


def _json_terms(path: Path) -> set[str]:
    text = _read_text(path)
    if not text:
        return set()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return set()
    return set(json.dumps(data, sort_keys=True).lower().replace("_", " ").split())


def _term_present(term: str, texts: Iterable[str]) -> bool:
    normalized = term.lower().replace("_", " ")
    return any(normalized in text.lower().replace("_", " ") for text in texts)


def audit_requirement(requirement: Requirement, repo_root: Path) -> AuditRow:
    code_paths = [repo_root / path for path in requirement.code_paths]
    output_paths = [repo_root / path for path in requirement.output_paths]
    code_texts = [_read_text(path) for path in code_paths]
    output_texts = [_read_text(path) for path in output_paths]
    output_columns = set()
    for path in output_paths:
        output_columns.update(_csv_columns(path))
        if path.suffix.lower() == ".json":
            output_columns.update(_json_terms(path))

    missing = []
    existing_code = [str(path.relative_to(repo_root)) for path in code_paths if path.exists()]
    existing_outputs = [str(path.relative_to(repo_root)) for path in output_paths if path.exists()]
    if len(existing_code) < len(code_paths):
        missing.extend(
            f"missing code path: {path.relative_to(repo_root)}" for path in code_paths if not path.exists()
        )
    if len(existing_outputs) < len(output_paths):
        missing.extend(
            f"missing output path: {path.relative_to(repo_root)}"
            for path in output_paths
            if not path.exists()
        )
    for column in requirement.required_columns:
        if column not in output_columns:
            missing.append(f"missing output column: {column}")
    for term in requirement.required_terms:
        if not _term_present(term, code_texts + output_texts):
            missing.append(f"missing evidence term: {term}")

    if not missing:
        status = "pass"
    elif existing_code or existing_outputs:
        status = "partial"
    else:
        status = "missing"
    return AuditRow(
        world=requirement.world,
        category=requirement.category,
        obligation=requirement.obligation,
        status=status,
        code_evidence="; ".join(existing_code) or "none",
        output_evidence="; ".join(existing_outputs) or "none",
        missing="; ".join(missing),
    )


def card_obligation_rows(cards_dir: Path, repo_root: Path) -> list[AuditRow]:
    rows: list[AuditRow] = []
    for path in sorted(cards_dir.glob("*.md")):
        sections = parse_markdown_sections(_read_text(path))
        if not sections:
            continue
        world = sections.get("World", "").strip()
        if not world or world == "TODO":
            continue
        obligations = [
            ("paper_benchmark", sections.get("Theoretical benchmark", "")),
            ("paper_metrics", sections.get("Metrics", "")),
            ("paper_reproduce", sections.get("What we need to reproduce", "")),
        ]
        world_root = repo_root / "worlds" / world
        output_glob = list((repo_root / "outputs").glob(f"{world}*/**/*")) if (repo_root / "outputs").exists() else []
        for category, obligation in obligations:
            text = obligation.strip()
            if not text or text == "TODO" or "Not stated in supplied text" in text:
                continue
            elif world_root.exists() or output_glob:
                status = "partial"
                missing = "human review required: compare filled card obligation to exact code/results"
            else:
                status = "missing"
                missing = "no world code/output evidence found"
            rows.append(
                AuditRow(
                    world=world,
                    category=category,
                    obligation=f"{path.name}: {text}",
                    status=status,
                    code_evidence=str(world_root.relative_to(repo_root)) if world_root.exists() else "none",
                    output_evidence=f"{len(output_glob)} matching output paths" if output_glob else "none",
                    missing=missing,
                )
            )
    return rows


def audit_obligations(
    *,
    repo_root: Path,
    literature_dir: Path,
    include_card_obligations: bool = True,
) -> list[AuditRow]:
    rows = [audit_requirement(requirement, repo_root) for requirement in WORLD_REQUIREMENTS]
    if include_card_obligations:
        rows.extend(card_obligation_rows(literature_dir / "paper_cards", repo_root))
    return rows


def write_audit_csv(rows: list[AuditRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else list(AuditRow("", "", "", "", "", "", "").__dict__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_audit_markdown(rows: list[AuditRow], path: Path) -> None:
    by_world: dict[str, list[AuditRow]] = {}
    for row in rows:
        by_world.setdefault(row.world, []).append(row)
    lines = [
        "# Theory Obligation Audit",
        "",
        "This is a deterministic coverage check. `pass` means required files/columns/terms were observed. "
        "`partial` means there is implementation evidence but the obligation still needs human review. "
        "`missing` means the evidence was not found.",
        "",
    ]
    totals = {
        status: sum(1 for row in rows if row.status == status)
        for status in ("pass", "partial", "missing")
    }
    lines.append(
        f"Summary: pass={totals['pass']}, partial={totals['partial']}, missing={totals['missing']}."
    )
    lines.append("")
    for world, world_rows in sorted(by_world.items()):
        lines.extend([f"## {world}", ""])
        for row in world_rows:
            lines.extend(
                [
                    f"### {row.category}: {row.status}",
                    "",
                    f"- Obligation: {row.obligation}",
                    f"- Code evidence: {row.code_evidence}",
                    f"- Output evidence: {row.output_evidence}",
                    f"- Missing/review: {row.missing or 'none'}",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_gap_status_report(literature_dir: Path, repo_root: Path) -> Path:
    rows = []
    for world, obligation in WORLD_OBLIGATIONS.items():
        rows.append(
            {
                "world": world,
                "classical_prediction": obligation["classical_prediction"],
                "known_rl_marl_result": obligation["known_rl_marl_result"],
                "benchmark_to_reproduce": obligation["benchmark"],
                "prior_metric": obligation["prior_metric"],
                "our_metric": obligation["our_metric"],
                "remaining_gap": obligation["gap"],
            }
        )
    path = literature_dir / "theory_gap_report.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path
