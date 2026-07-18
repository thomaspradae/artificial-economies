from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .http import read_jsonl
from .query_config import load_queries


WORLD_OBLIGATIONS = {
    "pricing_arena": {
        "classical": "Bertrand/Nash price benchmark and joint-profit/monopoly benchmark",
        "classical_prediction": "One-shot Bertrand competition predicts competitive/Nash pricing; repeated interaction can support supracompetitive prices under suitable dynamic incentives.",
        "known_rl_marl_result": "Q-learning and deep RL pricing agents can learn supracompetitive prices without explicit communication, but outcomes depend on algorithm, monitoring, and metric choice.",
        "benchmark": "static Nash prices, joint-profit price, profit-normalized Calvano-style collusion",
        "prior_metric": "collusion index, price/profit, welfare/profit, sometimes convergence",
        "our_metric": "price collusion, profit collusion, exploitability, welfare, price dispersion, quantity/profit",
        "ignored_failure_mode": "price-based collusion can understate profit extraction when regulation changes quantity/profit channels.",
        "code_obligation": "Reproduce Nash/joint-profit reference lines and report profit-normalized collusion next to the historical price proxy.",
        "gap": "institution robustness across learner capability and exploitability, not only collusion level",
    },
    "resource_island": {
        "classical": "commons/property/trade governance with monitoring, sanctions, and specialization",
        "classical_prediction": "Commons institutions require monitoring, credible exclusion/sanctions, and repeated interaction; trade requires observable gains from exchange and enough contact or market access.",
        "known_rl_marl_result": "Sequential social dilemmas and common-pool MARL often show cooperation failures or institution-sensitive cooperation depending on observability and incentives.",
        "benchmark": "oracle/greedy gather bounds plus institution activation thresholds",
        "prior_metric": "cooperation, sustainability, inequality, returns, sanction/compliance rates",
        "our_metric": "survival, welfare, trade, property pressure, sustainability, specialization, inequality",
        "ignored_failure_mode": "institutions can appear ineffective simply because their trigger conditions never occur.",
        "code_obligation": "Report trade attempts, successful trades, property opportunities, violations, and institution blocks before interpreting welfare differences.",
        "gap": "activation-validated institutions in a small reproducible spatial economy",
    },
    "auction_house": {
        "classical": "truthful second-price bidding, first-price bid shading, reserve revenue-efficiency tradeoff",
        "classical_prediction": "Second-price auctions make truthful bidding weakly dominant; first-price auctions induce bid shading; reserves can raise revenue while reducing allocative efficiency.",
        "known_rl_marl_result": "Learning auction papers evaluate revenue, regret/incentive compatibility, efficiency, and generalization rather than bidder reward alone.",
        "benchmark": "truthful Vickrey, shaded first-price, reserve/no-sale benchmark, ex-post regret",
        "prior_metric": "revenue, regret, incentive compatibility, efficiency, bidder utility",
        "our_metric": "revenue, efficiency, surplus, welfare, regret, over/underbidding, shading distance",
        "ignored_failure_mode": "high bidder payoff or seller revenue can hide incentive-compatibility or allocative-efficiency failures.",
        "code_obligation": "Compare learned bid curves to truthful and shaded benchmarks and report regret/exploitability-style misreport incentives.",
        "gap": "known auction theory recovered inside the same cross-world capability ladder",
    },
    "public_goods": {
        "classical": "free-rider equilibrium versus social optimum and commons regeneration threshold",
        "classical_prediction": "Private incentives underprovide contributions and overuse shared resources relative to the social optimum unless institutions alter incentives or information.",
        "known_rl_marl_result": "MARL public-goods and commons studies often find cooperation sensitive to reward shaping, punishment, reputation, and observability.",
        "benchmark": "free-rider/social-optimum policy brackets and collapse diagnostics",
        "prior_metric": "contribution, cooperation, punishment, welfare, sustainability",
        "our_metric": "contribution, extraction, sustainability, welfare, inequality, tax revenue, collapse",
        "ignored_failure_mode": "reward/accounting institutions can change measured welfare without changing the underlying pool state.",
        "code_obligation": "Separate state-changing effects from reward/accounting effects and compare learned behavior to free-rider/social-optimum brackets.",
        "gap": "separating reward-accounting institutions from state-changing institutions",
    },
    "labor_market": {
        "classical": "Gale-Shapley stability and proposing-side strategy-proofness",
        "classical_prediction": "Worker-proposing deferred acceptance produces stable matchings and is strategy-proof for the proposing side under standard assumptions.",
        "known_rl_marl_result": "Learning in matching markets is less standardized; the key obligation is to preserve mechanism-theory predictions before claiming learned manipulation.",
        "benchmark": "truthful deferred acceptance, blocking-pair checks, no profitable proposing-side report deviation",
        "prior_metric": "stability, match rate, welfare, regret/manipulation incentives",
        "our_metric": "match rate, stability, truthfulness, welfare, manipulation gain, blocking pairs",
        "ignored_failure_mode": "apparent manipulation by proposing-side agents may be a benchmark or mechanism-specification bug rather than an economic finding.",
        "code_obligation": "Verify stable/truthful benchmark cases and target manipulation tests at a side/mechanism where profitable deviations are theoretically possible.",
        "gap": "learned reporting behavior in an asymmetric-agent world using the shared mind ladder",
    },
}

WORLD_KEYWORDS = {
    "pricing_arena": ["algorithmic", "collusion", "pricing", "price", "bertrand", "calvano"],
    "resource_island": ["common-pool", "commons", "resource", "ostrom", "property", "rights"],
    "auction_house": ["auction", "auctions", "vickrey", "myerson", "bid", "bidding", "regret"],
    "public_goods": ["public", "goods", "free", "rider", "contribution", "punishment", "commons"],
    "labor_market": ["matching", "gale", "shapley", "deferred", "acceptance", "stable", "strategy"],
}

ANCHOR_PHRASES = {
    "pricing_arena": ["algorithmic pricing", "algorithmic collusion", "calvano", "bertrand"],
    "resource_island": ["common-pool resource", "governing the commons", "ostrom"],
    "auction_house": ["auction", "vickrey", "myerson", "regretnet", "optimal auctions"],
    "public_goods": ["public goods", "free rider", "common-pool resource", "tragedy of the commons"],
    "labor_market": ["gale-shapley", "deferred acceptance", "stable matching", "matching markets"],
}

PREFERRED_TITLE_CONTAINS = {
    "pricing_arena": [
        "artificial intelligence, algorithmic pricing, and collusion",
    ],
    "resource_island": [
        "a multi-agent reinforcement learning model of common-pool resource appropriation",
        "multi-agent reinforcement learning in sequential social dilemmas",
        "a review of design principles for community-based natural resource management",
    ],
    "auction_house": [
        "optimal auctions through deep learning",
        "optimal-er auctions through attention",
        "auctions and bidding",
        "optimal auction design",
    ],
    "public_goods": [
        "cooperation and punishment in public goods experiments",
        "the effect of rewards and sanctions in provision of public goods",
        "the voluntary provision of public goods",
    ],
    "labor_market": [
        "deferred acceptance algorithms",
        "college admissions and the stability of marriage",
        "the stable marriage problem",
        "what have we learned from market design",
    ],
}


def _relevance(row: dict[str, Any], world: str) -> float:
    title = str(row.get("title") or "").lower()
    abstract = str(row.get("abstract") or "").lower()
    query = str(row.get("query") or "").lower()
    haystack = f"{title} {abstract}"
    query_tokens = {token for token in query.replace("-", " ").split() if len(token) > 3}
    world_terms = set(WORLD_KEYWORDS.get(world, []))
    query_overlap = sum(1 for token in query_tokens if token in haystack)
    world_overlap = sum(1 for token in world_terms if token in haystack)
    title_bonus = 2 * sum(1 for token in query_tokens | world_terms if token in title)
    anchor_bonus = sum(1 for phrase in ANCHOR_PHRASES.get(world, []) if phrase in haystack)
    citation_score = min(int(row.get("citation_count") or 0), 500) / 10
    pdf_score = 30 if row.get("pdf_url") else 0
    return 25 * query_overlap + 35 * world_overlap + 100 * anchor_bonus + title_bonus + citation_score + pdf_score


def _best_paper_for_world(records: list[dict[str, Any]], world: str) -> dict[str, Any] | None:
    candidates = [r for r in records if r.get("world") == world]
    if not candidates:
        return None
    preferred = PREFERRED_TITLE_CONTAINS.get(world, [])
    for wanted in preferred:
        matches = [
            row
            for row in candidates
            if wanted in str(row.get("title") or "").lower()
        ]
        if matches:
            return max(matches, key=lambda row: _relevance(row, world))
    return max(candidates, key=lambda row: _relevance(row, world))


def build_gap_rows(query_path: Path, raw_path: Path) -> list[dict[str, Any]]:
    config = load_queries(query_path)
    records = read_jsonl(raw_path)
    rows: list[dict[str, Any]] = []
    for world, groups in config.get("worlds", {}).items():
        obligation = WORLD_OBLIGATIONS.get(world, {})
        paper = _best_paper_for_world(records, world) or {}
        title = paper.get("title") or "TODO: run scout search and select closest paper"
        for institution in groups.get("institutions", []) or ["TODO"]:
            for mind in groups.get("minds", []) or ["TODO"]:
                rows.append(
                    {
                        "world": world,
                        "institution": institution,
                        "mind": mind,
                        "closest_paper": title,
                        "theory_benchmark": obligation.get("benchmark", "TODO"),
                        "their_metric": obligation.get("prior_metric", "TODO"),
                        "our_metric": obligation.get("our_metric", "TODO"),
                        "gap": obligation.get("gap", "TODO"),
                    }
                )
    return rows


def write_gap_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "world",
        "institution",
        "mind",
        "closest_paper",
        "theory_benchmark",
        "their_metric",
        "our_metric",
        "gap",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_theory_obligations(rows: list[dict[str, Any]], path: Path) -> None:
    worlds = {}
    for row in rows:
        worlds.setdefault(row["world"], row)
    lines = [
        "# Theory Obligations",
        "",
        "Generated from `literature/queries.yaml` and `literature/papers_raw.jsonl`.",
        "",
    ]
    for world, row in worlds.items():
        lines.extend(
            [
                f"## {world}",
                "",
                f"- Closest paper: {row['closest_paper']}",
                f"- Classical prediction: {WORLD_OBLIGATIONS.get(world, {}).get('classical_prediction', 'TODO')}",
                f"- Known RL/MARL result: {WORLD_OBLIGATIONS.get(world, {}).get('known_rl_marl_result', 'TODO')}",
                f"- Theory benchmark: {row['theory_benchmark']}",
                f"- Prior metric: {row['their_metric']}",
                f"- Our metric: {row['our_metric']}",
                f"- Prior-work failure mode to check: {WORLD_OBLIGATIONS.get(world, {}).get('ignored_failure_mode', 'TODO')}",
                f"- Code obligation: {WORLD_OBLIGATIONS.get(world, {}).get('code_obligation', 'TODO')}",
                f"- Gap: {row['gap']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
