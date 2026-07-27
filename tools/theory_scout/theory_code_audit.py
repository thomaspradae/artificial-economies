from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JsonExpectation:
    path: str
    dotted_key: str
    expected: Any


@dataclass(frozen=True)
class TheoryCodeGate:
    world: str
    gate: str
    theory_anchor: str
    theory_obligation: str
    metric_obligation: str
    code_paths: tuple[str, ...] = ()
    output_paths: tuple[str, ...] = ()
    required_columns: tuple[str, ...] = ()
    required_terms: tuple[str, ...] = ()
    json_expectations: tuple[JsonExpectation, ...] = ()
    needed_fix: str = ""
    rerun_required: str = "no"
    outputs_made_obsolete: str = "none"
    claim_boundary: str = ""


@dataclass(frozen=True)
class TheoryCodeAuditRow:
    world: str
    gate: str
    status: str
    decision: str
    theory_anchor: str
    theory_obligation: str
    metric_obligation: str
    code_evidence: str
    output_evidence: str
    missing_evidence: str
    needed_fix: str
    rerun_required: str
    outputs_made_obsolete: str
    claim_boundary: str


GATES: tuple[TheoryCodeGate, ...] = (
    TheoryCodeGate(
        world="pricing_arena",
        gate="pricing_benchmarks_and_collusion_metrics",
        theory_anchor="Tirole/IO theory; Calvano et al.; algorithmic pricing RL papers",
        theory_obligation=(
            "Static Nash and joint-profit references must anchor learned pricing, "
            "and collusion must be measured on both price and profit scales."
        ),
        metric_obligation=(
            "Nash/joint-profit benchmark, price-normalized collusion, "
            "profit-normalized collusion, welfare, quantity/profit, exploitability."
        ),
        code_paths=(
            "worlds/pricing_arena/benchmarks.py",
            "core/metrics.py",
            "build_combined_table.py",
        ),
        output_paths=(
            "outputs/full_v0_multiseed/summary_aggregate.csv",
            "outputs/phase3_full/mind_comparison.csv",
        ),
        required_columns=(
            "collusion_index_mean",
            "profit_collusion_index_mean",
            "exploitability_mean",
            "avg_price_mean",
            "profit_total_mean",
            "quantity_total_mean",
        ),
        required_terms=("nash", "profit_collusion_index", "exploitability"),
        needed_fix="No code rerun. Write the metric split and price-cap quantity/profit channel as the result.",
        claim_boundary="Does not prove universal collusion convergence or real-world antitrust harm.",
    ),
    TheoryCodeGate(
        world="pricing_arena",
        gate="pricing_full_capability_ladder",
        theory_anchor="DQN, PPO, independent-learning MARL, and centralized-critic method papers",
        theory_obligation=(
            "The mind ladder must be made of distinct implemented learners, not duplicate labels."
        ),
        metric_obligation="Every mind row needs n=20 multiseed and exploitability evidence under the same pricing protocol.",
        code_paths=(
            "minds/deep_rl/torch_dqn_mind.py",
            "minds/deep_rl/torch_ppo_mind.py",
            "minds/marl/independent_learners.py",
            "minds/marl/centralized_critic.py",
        ),
        output_paths=(
            "outputs/phase3_full/mind_comparison.csv",
            "outputs/phase3_full/validation_report.json",
        ),
        required_columns=("mind", "mechanism", "n_seeds_multiseed", "n_seeds_exploitability"),
        required_terms=("independent_dqn", "centralized_critic"),
        needed_fix="No rerun unless the mind implementation changes again.",
        claim_boundary="Capability findings are empirical for this protocol, not formal ordering guarantees.",
    ),
    TheoryCodeGate(
        world="resource_island",
        gate="resource_v1_activation_pressure",
        theory_anchor="Hardin/Ostrom/common-pool governance; sequential social-dilemma MARL",
        theory_obligation=(
            "Property, trade-price-control, and reputation claims are valid only if the institution "
            "has measured opportunities to bind."
        ),
        metric_obligation=(
            "Trade attempts/successes, institution blocks, property opportunities, property violations, "
            "survival, welfare, sustainability, specialization, inequality."
        ),
        code_paths=(
            "worlds/resource_island/env.py",
            "worlds/resource_island/training.py",
            "run_resource_island_smoke.py",
        ),
        output_paths=("outputs/resource_island_v1_full/summary_aggregate.csv",),
        required_columns=(
            "trade_count_mean",
            "trade_attempt_count_mean",
            "trade_institution_blocked_count_mean",
            "property_opportunities_mean",
            "property_resource_opportunities_mean",
            "specialization_index_mean",
            "resource_sustainability_mean",
        ),
        required_terms=(
            "property_opportunities",
            "trade_food_units",
            "trade_wood_units",
            "specialization",
        ),
        needed_fix="No tabular v1 rerun. Use v1 outputs for Resource Island institution claims.",
        claim_boundary="Still a simplified matching-market trade protocol, not a full field model of common-pool governance.",
    ),
    TheoryCodeGate(
        world="resource_island",
        gate="resource_v1_neural_ladder_apples_to_apples",
        theory_anchor="Sequential social-dilemma MARL; DQN/PPO/independent-DQN/centralized-critic obligations",
        theory_obligation=(
            "Capability claims must compare minds on the same v1 activation-pressure protocol. "
            "Do not compare v1 Q-learning against older random-layout neural runs."
        ),
        metric_obligation="Same v1 scenarios, steps, seeds, activation preset, trade units, and source directories for every mind.",
        code_paths=(
            "worlds/resource_island/training.py",
            "worlds/resource_island/features.py",
            "build_resource_island_mind_comparison.py",
        ),
        output_paths=(
            "outputs/resource_island_v1_dqn_full/summary_aggregate.csv",
            "outputs/resource_island_v1_ppo_full/summary_aggregate.csv",
            "outputs/resource_island_v1_independent_dqn_full/summary_aggregate.csv",
            "outputs/resource_island_v1_centralized_critic_full/summary_aggregate.csv",
            "outputs/resource_island_v1_phase3_full/mind_comparison.csv",
        ),
        required_columns=(
            "trade_count_mean",
            "trade_attempt_count_mean",
            "trade_institution_blocked_count_mean",
            "property_opportunities_mean",
        ),
        required_terms=("resource_island",),
        needed_fix=(
            "Finish or pull the active v1 neural/MARL reruns, rebuild the Resource Island mind-comparison table, "
            "then rebuild cross-world synthesis and publication-inference outputs."
        ),
        rerun_required="yes_active_or_pull",
        outputs_made_obsolete=(
            "Resource Island cross-mind rows in outputs/resource_island_phase3_full/mind_comparison.csv "
            "and any cross-world synthesis using those mismatched rows."
        ),
        claim_boundary=(
            "Until this passes, Resource Island v1 institution claims are usable for Q-learning, "
            "but Resource Island capability-ladder claims are not final."
        ),
    ),
    TheoryCodeGate(
        world="resource_island",
        gate="resource_strict_local_trade_ablation",
        theory_anchor="Spatial common-pool and market-friction theory",
        theory_obligation=(
            "If the paper claims Resource Island is a spatial trade economy, it must separate strict-local "
            "trade from whole-island matching."
        ),
        metric_obligation="Whole-island versus trade_radius=1 outcomes for trade, welfare, survival, and institution activation.",
        code_paths=("worlds/resource_island/env.py", "run_resource_island_smoke.py"),
        output_paths=("outputs/resource_island_v1_strict_radius_full/summary_aggregate.csv",),
        required_columns=("trade_count_mean", "contact_rate_mean", "welfare_mean", "survival_rate_mean"),
        required_terms=("trade_radius",),
        needed_fix="Run this only before making strong spatial-friction claims; not required for current v1 institution activation result.",
        rerun_required="optional_before_spatial_claim",
        outputs_made_obsolete="none; this is an ablation, not a replacement.",
        claim_boundary="Without it, call v1 trade a whole-island matching market layered over spatial gathering.",
    ),
    TheoryCodeGate(
        world="auction_house",
        gate="auction_benchmarks_metrics_and_ladder",
        theory_anchor="Vickrey; Myerson; RegretNet/deep auction-design papers; repeated-auction learning",
        theory_obligation=(
            "Auction results must be evaluated against truthful second-price, first-price shading, reserve/no-sale, "
            "and regret/incentive-compatibility proxies."
        ),
        metric_obligation="Revenue, bidder surplus, welfare, allocative efficiency, ex-post regret, bid curves, over/underbidding.",
        code_paths=(
            "worlds/auction_house/benchmarks.py",
            "worlds/auction_house/env.py",
            "worlds/auction_house/training.py",
            "build_world_mind_comparison.py",
        ),
        output_paths=(
            "outputs/auction_house_full/summary_aggregate.csv",
            "outputs/auction_house_full/bid_curves.csv",
            "outputs/auction_house_phase3_full/mind_comparison.csv",
        ),
        required_columns=(
            "revenue_mean",
            "allocative_efficiency_mean",
            "ex_post_regret_mean_mean",
            "truthful_bid_distance_mean_mean",
            "first_price_shading_distance_mean_mean",
            "underbid_rate_mean",
        ),
        required_terms=("truthful", "regret", "allocative_efficiency", "clock"),
        needed_fix="No structural code rerun. Interpret low second-price efficiency/regret as learned-bidder behavior, not failed auction theory.",
        claim_boundary="The repo tests fixed auction mechanisms; it does not learn new RegretNet/Myerson mechanisms.",
    ),
    TheoryCodeGate(
        world="public_goods",
        gate="public_goods_state_vs_accounting",
        theory_anchor="Samuelson/Olson/Hardin; public-goods punishment, rewards, reputation, and observability literature",
        theory_obligation=(
            "Commons institutions must be judged by contribution/extraction/sustainability, not reward accounting alone."
        ),
        metric_obligation="Free-rider/social brackets, contribution, extraction, sustainability, collapse, welfare, inequality, tax revenue.",
        code_paths=(
            "worlds/public_goods/benchmarks.py",
            "worlds/public_goods/env.py",
            "validate_public_goods_effects.py",
            "build_world_mind_comparison.py",
        ),
        output_paths=(
            "outputs/public_goods_full/summary_aggregate.csv",
            "outputs/public_goods_full/institution_effect_validation.json",
            "outputs/public_goods_phase3_full/mind_comparison.csv",
        ),
        required_columns=(
            "sustainability_mean",
            "contribution_total_mean",
            "extraction_total_mean",
            "collapse_rate_mean",
            "welfare_mean",
            "tax_revenue_mean",
        ),
        required_terms=("free_rider", "social_optimum", "state", "accounting"),
        needed_fix="No rerun unless changing institution definitions. Write state-changing versus accounting-only distinction explicitly.",
        claim_boundary="Reward gains from reputation/taxes are not automatically sustainability gains.",
    ),
    TheoryCodeGate(
        world="labor_market",
        gate="labor_market_da_stability_and_side_specific_incentives",
        theory_anchor="Gale-Shapley; Roth; DA mechanism-design literature; matching-market RL",
        theory_obligation=(
            "Deferred-acceptance claims must report stability/blocking pairs and respect proposing-side strategy-proofness."
        ),
        metric_obligation="Match rate, stability, blocking pairs, welfare, truthful-report rate, manipulation-gain diagnostics.",
        code_paths=(
            "worlds/labor_market/benchmarks.py",
            "worlds/labor_market/env.py",
            "run_labor_market_benchmark_cases.py",
            "build_world_mind_comparison.py",
        ),
        output_paths=(
            "outputs/labor_market_full/summary_aggregate.csv",
            "outputs/labor_market_benchmark_cases.json",
            "outputs/labor_market_phase3_full/mind_comparison.csv",
        ),
        required_columns=(
            "match_rate_mean",
            "stability_mean",
            "truthful_report_rate_mean",
            "total_welfare_mean",
            "manipulation_gain_mean_mean",
        ),
        required_terms=("blocking", "strategy", "truthful", "manipulation"),
        needed_fix="No rerun. Write worker-proposing DA caveat; future manipulation tests should target non-proposing side or another mechanism.",
        claim_boundary="Non-truthful reports are not automatically profitable manipulation under worker-proposing DA.",
    ),
    TheoryCodeGate(
        world="central_planner_tax",
        gate="tax_schedule_as_institution_not_world",
        theory_anchor="Public finance / redistribution as a mechanism layer",
        theory_obligation=(
            "The central planner is currently a parameterized tax institution; do not present it as an agent/world without a planner state/action model."
        ),
        metric_obligation="Tax revenue, welfare, inequality, sustainability/survival under rate sweeps.",
        code_paths=("institutions/tax_schedule.py", "run_tax_schedule_sweep.py"),
        output_paths=("outputs/tax_schedule_sweep/summary_aggregate.csv",),
        required_columns=("tax_rate", "tax_revenue_mean", "welfare_mean", "inequality_mean"),
        required_terms=("tax_rate", "redistribution"),
        needed_fix="No rerun for current claim. Add a real planner state/action model only if making central-planner-agent claims.",
        claim_boundary="This is a tax sweep, not a central planner world.",
    ),
    TheoryCodeGate(
        world="cross_world",
        gate="cross_world_protocol_comparability",
        theory_anchor="Capability-ladder synthesis obligation",
        theory_obligation=(
            "Cross-world capability claims require comparable protocols inside each world and must exclude mismatched rows."
        ),
        metric_obligation="Protocol comparability report, monotonicity report, synthesis table, publication-inference exclusions.",
        code_paths=(
            "build_cross_world_synthesis.py",
            "build_publication_inference.py",
        ),
        output_paths=(
            "outputs/cross_world_synthesis/synthesis_table.csv",
            "outputs/cross_world_synthesis/protocol_comparability_report.json",
            "outputs/publication_inference/publication_inference_summary.md",
        ),
        required_columns=("world", "mind", "key_metric_1_name", "key_metric_1"),
        required_terms=("Resource Island cross-mind effects are excluded",),
        json_expectations=(
            JsonExpectation(
                "outputs/cross_world_synthesis/protocol_comparability_report.json",
                "resource_island.cross_mind_capability_claims_valid",
                True,
            ),
        ),
        needed_fix="Rebuild this after Resource Island v1 neural/MARL outputs land; current synthesis correctly blocks Resource Island cross-mind claims.",
        rerun_required="yes_after_resource_island_v1_ladder",
        outputs_made_obsolete="outputs/cross_world_synthesis/* and outputs/publication_inference/* once Resource Island v1 ladder is rebuilt.",
        claim_boundary="Current five-world synthesis is valid only with Resource Island cross-mind exclusions.",
    ),
)


def _read_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _csv_columns(path: Path) -> set[str]:
    if not path.exists() or path.suffix.lower() != ".csv":
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return set(next(reader))
        except StopIteration:
            return set()


def _load_json(path: Path) -> Any:
    if not path.exists() or path.suffix.lower() != ".json":
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _json_lookup(data: Any, dotted_key: str) -> Any:
    current = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _contains_term(term: str, texts: list[str]) -> bool:
    needle = term.lower().replace("_", " ")
    return any(needle in text.lower().replace("_", " ") for text in texts)


def _decision(status: str, rerun_required: str) -> str:
    if status == "pass" and rerun_required == "no":
        return "write_current_result"
    if status == "pass" and rerun_required.startswith("optional"):
        return "optional_ablation_before_stronger_claim"
    if status == "pass":
        return "ready_after_requested_rerun"
    if rerun_required == "no":
        return "patch_or_interpret_before_writing"
    if rerun_required.startswith("optional"):
        return "optional_ablation_missing"
    return "finish_or_rerun_before_final_claim"


def audit_gate(gate: TheoryCodeGate, repo_root: Path) -> TheoryCodeAuditRow:
    code_paths = [repo_root / path for path in gate.code_paths]
    output_paths = [repo_root / path for path in gate.output_paths]
    code_texts = [_read_text(path) for path in code_paths]
    output_texts = [_read_text(path) for path in output_paths]

    missing: list[str] = []
    existing_code = [path for path in code_paths if path.exists()]
    existing_outputs = [path for path in output_paths if path.exists()]
    for path in code_paths:
        if not path.exists():
            missing.append(f"missing code path: {path.relative_to(repo_root)}")
    for path in output_paths:
        if not path.exists():
            missing.append(f"missing output path: {path.relative_to(repo_root)}")

    output_columns: set[str] = set()
    for path in output_paths:
        output_columns.update(_csv_columns(path))
    if existing_outputs:
        for column in gate.required_columns:
            if column not in output_columns:
                missing.append(f"missing output column: {column}")
    for term in gate.required_terms:
        if not _contains_term(term, code_texts + output_texts):
            missing.append(f"missing evidence term: {term}")
    for expectation in gate.json_expectations:
        path = repo_root / expectation.path
        observed = _json_lookup(_load_json(path), expectation.dotted_key)
        if observed != expectation.expected:
            missing.append(
                f"json mismatch: {expectation.path}:{expectation.dotted_key} "
                f"expected {expectation.expected!r}, observed {observed!r}"
            )

    if not missing:
        status = "pass"
    elif existing_code or existing_outputs:
        status = "partial"
    else:
        status = "missing"

    return TheoryCodeAuditRow(
        world=gate.world,
        gate=gate.gate,
        status=status,
        decision=_decision(status, gate.rerun_required),
        theory_anchor=gate.theory_anchor,
        theory_obligation=gate.theory_obligation,
        metric_obligation=gate.metric_obligation,
        code_evidence="; ".join(str(path.relative_to(repo_root)) for path in existing_code) or "none",
        output_evidence="; ".join(str(path.relative_to(repo_root)) for path in existing_outputs) or "none",
        missing_evidence="; ".join(missing) or "none",
        needed_fix=gate.needed_fix,
        rerun_required=gate.rerun_required,
        outputs_made_obsolete=gate.outputs_made_obsolete,
        claim_boundary=gate.claim_boundary,
    )


def build_theory_code_audit(repo_root: Path) -> list[TheoryCodeAuditRow]:
    return [audit_gate(gate, repo_root) for gate in GATES]


def write_theory_code_audit_csv(rows: list[TheoryCodeAuditRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else list(TheoryCodeAuditRow("", "", "", "", "", "", "", "", "", "", "", "", "", "").__dict__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _status_counts(rows: list[TheoryCodeAuditRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    return counts


def write_theory_code_audit_markdown(rows: list[TheoryCodeAuditRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = _status_counts(rows)
    immediate = [
        row
        for row in rows
        if row.status != "pass" and row.rerun_required not in {"no", "optional_before_spatial_claim"}
    ]
    optional = [
        row
        for row in rows
        if row.status != "pass" and row.rerun_required == "optional_before_spatial_claim"
    ]
    lines = [
        "# Theory-To-Code Audit",
        "",
        "Generated by `python -m tools.theory_scout.cli theory-code-audit`.",
        "",
        "This is the decision layer above the paper-card obligations. It answers:",
        "",
        "- Does the current code implement the theory obligation?",
        "- Do the current outputs contain the required benchmark/metric evidence?",
        "- Does a result need a rerun, a caveat, or only writing?",
        "- Which old outputs become obsolete if the fix lands?",
        "",
        "Status semantics:",
        "",
        "- `pass`: required code, outputs, columns/terms, and JSON gates were observed.",
        "- `partial`: some evidence exists, but a required output/column/term/protocol gate is missing.",
        "- `missing`: no meaningful code/output evidence was found.",
        "",
        "Summary: "
        + ", ".join(f"{status}={count}" for status, count in sorted(counts.items())),
        "",
        "## Immediate Run / Rebuild Queue",
        "",
    ]
    if immediate:
        for row in immediate:
            lines.extend(
                [
                    f"- **{row.world} / {row.gate}**: {row.needed_fix}",
                    f"  - Missing: {row.missing_evidence}",
                    f"  - Obsoletes: {row.outputs_made_obsolete}",
                ]
            )
    else:
        lines.append("- None. Current remaining gaps are writing or optional ablations.")
    lines.extend(["", "## Optional Before Stronger Claims", ""])
    if optional:
        for row in optional:
            lines.extend(
                [
                    f"- **{row.world} / {row.gate}**: {row.needed_fix}",
                    f"  - Missing: {row.missing_evidence}",
                    f"  - Boundary: {row.claim_boundary}",
                ]
            )
    else:
        lines.append("- None.")
    lines.append("")

    by_world: dict[str, list[TheoryCodeAuditRow]] = {}
    for row in rows:
        by_world.setdefault(row.world, []).append(row)
    for world, world_rows in by_world.items():
        lines.extend([f"## {world}", ""])
        for row in world_rows:
            lines.extend(
                [
                    f"### {row.gate}: {row.status}",
                    "",
                    f"- Decision: `{row.decision}`",
                    f"- Theory anchor: {row.theory_anchor}",
                    f"- Theory obligation: {row.theory_obligation}",
                    f"- Metric obligation: {row.metric_obligation}",
                    f"- Code evidence: `{row.code_evidence}`",
                    f"- Output evidence: `{row.output_evidence}`",
                    f"- Missing evidence: {row.missing_evidence}",
                    f"- Needed fix: {row.needed_fix}",
                    f"- Rerun required: `{row.rerun_required}`",
                    f"- Outputs made obsolete by fix: {row.outputs_made_obsolete}",
                    f"- Claim boundary: {row.claim_boundary}",
                    "",
                ]
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
