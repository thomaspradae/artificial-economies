from __future__ import annotations

import csv
import difflib
import unicodedata
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .fill_paper_cards import read_records, source_context_for_record
from .make_paper_cards import slugify


@dataclass(frozen=True)
class FoundationPaper:
    world: str
    priority: str
    role: str
    title: str
    authors: str
    year: str
    institution: str
    mind: str
    why_it_matters: str
    theory_obligation: str
    code_result_check: str
    aliases: tuple[str, ...] = ()
    doi: str = ""


@dataclass
class FoundationMatch:
    world: str
    priority: str
    role: str
    title: str
    authors: str
    year: str
    institution: str
    mind: str
    why_it_matters: str
    theory_obligation: str
    code_result_check: str
    cache_status: str
    match_score: str
    matched_title: str
    matched_authors: str
    matched_year: str
    matched_doi: str
    url: str
    pdf_url: str
    has_extracted_text: bool
    text_path: str
    manual_action: str
    google_scholar_url: str


FOUNDATION_PAPERS: tuple[FoundationPaper, ...] = (
    FoundationPaper(
        world="cross_world_methods",
        priority="must_read",
        role="tabular_rl",
        title="Q-learning",
        authors="Christopher J. C. H. Watkins; Peter Dayan",
        year="1992",
        institution="all",
        mind="q_learning",
        why_it_matters="Canonical tabular temporal-difference control reference.",
        theory_obligation="Report the state/action encoding and update target clearly enough that Q-learning behavior is interpretable.",
        code_result_check="minds/q_learning.py; cross-world Q-learning smoke and full runs.",
        doi="10.1007/BF00992698",
    ),
    FoundationPaper(
        world="cross_world_methods",
        priority="must_read",
        role="deep_rl",
        title="Human-level control through deep reinforcement learning",
        authors="Volodymyr Mnih et al.",
        year="2015",
        institution="all",
        mind="dqn",
        why_it_matters="Canonical DQN paper: replay, target network, TD loss, and benchmark discipline.",
        theory_obligation="Verify that DQN is a real structured-observation learner with replay and target networks, not a renamed tabular baseline.",
        code_result_check="minds/deep_rl/torch_dqn_mind.py; Phase 3 and Resource Island DQN tests.",
        doi="10.1038/nature14236",
    ),
    FoundationPaper(
        world="cross_world_methods",
        priority="must_read",
        role="policy_gradient",
        title="Proximal Policy Optimization Algorithms",
        authors="John Schulman; Filip Wolski; Prafulla Dhariwal; Alec Radford; Oleg Klimov",
        year="2017",
        institution="all",
        mind="ppo",
        why_it_matters="Canonical PPO reference for clipped policy-gradient learning.",
        theory_obligation="Report PPO as an on-policy actor-critic learner and compare it separately from value-based DQN.",
        code_result_check="minds/deep_rl/torch_ppo_mind.py; Phase 3 PPO tests and full outputs.",
        doi="10.48550/arXiv.1707.06347",
    ),
    FoundationPaper(
        world="cross_world_methods",
        priority="must_read",
        role="independent_marl",
        title="Multi-Agent Reinforcement Learning: Independent versus Cooperative Agents",
        authors="Ming Tan",
        year="1993",
        institution="all",
        mind="independent_dqn",
        why_it_matters="Canonical independent-learner baseline for MARL.",
        theory_obligation="Ensure independent learners have separate policies, replay/exploration streams, and no accidental aliasing.",
        code_result_check="minds/marl/independent_learners.py; decorrelation tests; fixed independent-DQN full outputs.",
    ),
    FoundationPaper(
        world="cross_world_methods",
        priority="must_read",
        role="centralized_critic",
        title="Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments",
        authors="Ryan Lowe; Yi Wu; Aviv Tamar; Jean Harb; Pieter Abbeel; Igor Mordatch",
        year="2017",
        institution="all",
        mind="centralized_critic",
        why_it_matters="Canonical centralized-training/decentralized-execution MARL anchor.",
        theory_obligation="Separate centralized critic evidence from independent learning evidence, especially in asymmetric worlds.",
        code_result_check="minds/marl/centralized_critic.py; cross-world P.6 smoke/full outputs.",
    ),
    FoundationPaper(
        world="pricing_arena",
        priority="must_read",
        role="classical_io",
        title="The Theory of Industrial Organization",
        authors="Jean Tirole",
        year="1988",
        institution="none",
        mind="all",
        why_it_matters="Classical industrial-organization reference for Bertrand pricing and collusion benchmarks.",
        theory_obligation="Use static Nash/Bertrand and joint-profit references when interpreting learned prices.",
        code_result_check="worlds/pricing_arena/benchmarks.py; pricing summary benchmark columns.",
    ),
    FoundationPaper(
        world="pricing_arena",
        priority="must_read",
        role="algorithmic_collusion",
        title="Artificial Intelligence, Algorithmic Pricing, and Collusion",
        authors="Emilio Calvano; Giacomo Calzolari; Vincenzo Denicolo; Sergio Pastorello",
        year="2020",
        institution="none",
        mind="q_learning",
        why_it_matters="Main algorithmic-pricing-collusion benchmark for Q-learning firms.",
        theory_obligation="Report profit-normalized collusion, not only a price proxy.",
        code_result_check="outputs/phase3_full/mind_comparison.csv includes profit_collusion_index_mean.",
        doi="10.1257/aer.20190623",
    ),
    FoundationPaper(
        world="pricing_arena",
        priority="must_read",
        role="monitoring",
        title="Algorithmic collusion with imperfect monitoring",
        authors="Emilio Calvano; Giacomo Calzolari; Vincenzo Denicolo; Sergio Pastorello",
        year="2021",
        institution="demand_shock",
        mind="q_learning",
        why_it_matters="Connects collusion to noisy monitoring and price-war punishment dynamics.",
        theory_obligation="Treat demand shocks and monitoring as institution/environment changes, not just random noise.",
        code_result_check="demand_shock mechanism; exploitability and collusion comparison under shocks.",
        doi="10.1016/j.ijindorg.2021.102712",
    ),
    FoundationPaper(
        world="pricing_arena",
        priority="core",
        role="algorithm_design",
        title="Artificial Intelligence, Algorithm Design, and Pricing",
        authors="John Asker; Chaim Fershtman; Ariel Pakes",
        year="2022",
        institution="none",
        mind="q_learning,dqn",
        why_it_matters="Shows that learning-protocol details can move pricing outcomes from competitive to monopoly-like.",
        theory_obligation="Interpret capability differences as algorithm-design effects, not only hyperparameter noise.",
        code_result_check="Phase 3 ladder: Q-learning, DQN, PPO, independent-DQN, centralized critic.",
        doi="10.1257/pandp.20221059",
    ),
    FoundationPaper(
        world="pricing_arena",
        priority="core",
        role="sequential_pricing",
        title="Autonomous algorithmic collusion: Q-learning under sequential pricing",
        authors="Timo Klein",
        year="2021",
        institution="none",
        mind="q_learning",
        why_it_matters="Shows algorithmic collusion can arise under sequential pricing protocols.",
        theory_obligation="Be explicit about simultaneous versus sequential timing in our environment.",
        code_result_check="Pricing Arena design and step semantics.",
        aliases=("Autonomous Algorithmic Collusion: Q-Learning Under Sequential Pricing",),
    ),
    FoundationPaper(
        world="pricing_arena",
        priority="supporting",
        role="deep_rl_pricing",
        title="Algorithmic Collusion in Dynamic Pricing with Deep Reinforcement Learning",
        authors="Shidi Deng; Maximilian Schiffer; Martin Bichler",
        year="2024",
        institution="none",
        mind="dqn,ppo",
        why_it_matters="Closest deep-RL pricing anchor for the neural side of the ladder.",
        theory_obligation="Compare whether neural agents reproduce or break tabular collusion patterns.",
        code_result_check="Phase 3 mind comparison table and price-cap diagnosis.",
    ),
    FoundationPaper(
        world="auction_house",
        priority="must_read",
        role="truthful_second_price",
        title="Counterspeculation, Auctions, and Competitive Sealed Tenders",
        authors="William Vickrey",
        year="1961",
        institution="second_price",
        mind="all",
        why_it_matters="Original second-price auction truthfulness anchor.",
        theory_obligation="Second-price benchmark must include truthful bidding, efficiency, and regret.",
        code_result_check="worlds/auction_house/benchmarks.py; ex_post_regret and efficiency metrics.",
        doi="10.1111/j.1540-6261.1961.tb02789.x",
    ),
    FoundationPaper(
        world="auction_house",
        priority="must_read",
        role="optimal_auction",
        title="Optimal Auction Design",
        authors="Roger B. Myerson",
        year="1981",
        institution="reserve_price",
        mind="all",
        why_it_matters="Classical revenue-optimal auction theory and reserve-price obligation.",
        theory_obligation="Reserve-price scenarios must report revenue/efficiency tradeoffs.",
        code_result_check="second_price_reserve scenario; revenue and allocative_efficiency columns.",
        doi="10.2307/1912256",
    ),
    FoundationPaper(
        world="auction_house",
        priority="must_read",
        role="learned_auction_design",
        title="Optimal auctions through deep learning",
        authors="Paul Dutting; Zhe Feng; Harikrishna Narasimhan; David C. Parkes; Sai Srivatsa Ravindranath",
        year="2019",
        institution="all",
        mind="dqn,ppo,centralized_critic",
        why_it_matters="RegretNet line: auction-learning papers care about regret/IC, revenue, efficiency, and generalization.",
        theory_obligation="Do not evaluate auctions only by bidder reward; include incentive-compatibility proxies.",
        code_result_check="Auction House regret, revenue, efficiency, over/underbidding diagnostics.",
        aliases=("Optimal Auctions through Deep Learning: Advances in Differentiable Economics",),
    ),
    FoundationPaper(
        world="auction_house",
        priority="core",
        role="ai_auction_design",
        title="Artificial Intelligence and Auction Design",
        authors="Martino Banchio; Andrzej Skrzypacz",
        year="2022",
        institution="first_price,second_price",
        mind="q_learning",
        why_it_matters="Closest Q-learning auction-design paper for first-price versus second-price behavior.",
        theory_obligation="Compare first-price bid shading/collusion-style low bids against second-price behavior.",
        code_result_check="Auction House first_price and second_price learned bid curves.",
        aliases=("Market Design for AI Algorithms",),
    ),
    FoundationPaper(
        world="auction_house",
        priority="core",
        role="repeated_auctions",
        title="Learning in repeated auctions",
        authors="Thomas Nedelec; Clement Calauzenes; Noureddine El Karoui; Vianney Perchet",
        year="2020",
        institution="all",
        mind="q_learning,dqn",
        why_it_matters="Learning dynamics in repeated auction environments.",
        theory_obligation="Treat repeated auction learning as a dynamic strategic problem, not independent one-shot auctions.",
        code_result_check="Auction House repeated draws and final-window bid curves.",
    ),
    FoundationPaper(
        world="public_goods",
        priority="must_read",
        role="public_goods_theory",
        title="The Pure Theory of Public Expenditure",
        authors="Paul A. Samuelson",
        year="1954",
        institution="none",
        mind="all",
        why_it_matters="Classical public-goods benchmark: private and social incentives diverge.",
        theory_obligation="Report free-rider versus social-optimum brackets.",
        code_result_check="worlds/public_goods/benchmarks.py; public_goods_full summaries.",
        doi="10.2307/1925895",
    ),
    FoundationPaper(
        world="public_goods",
        priority="must_read",
        role="collective_action",
        title="The Logic of Collective Action",
        authors="Mancur Olson",
        year="1965",
        institution="none",
        mind="all",
        why_it_matters="Foundational free-rider/collective-action framing.",
        theory_obligation="Explain why low contribution is expected under private incentives.",
        code_result_check="baseline extraction/contribution/sustainability metrics.",
    ),
    FoundationPaper(
        world="public_goods",
        priority="must_read",
        role="punishment",
        title="Cooperation and Punishment in Public Goods Experiments",
        authors="Ernst Fehr; Simon Gachter",
        year="2000",
        institution="penalty_schedule",
        mind="all",
        why_it_matters="Canonical evidence that sanctions can sustain contributions.",
        theory_obligation="Penalty institutions must report whether contributions/sustainability change, not only reward penalties.",
        code_result_check="validate_public_goods_effects.py; penalty_schedule state-change classification.",
        doi="10.1257/aer.90.4.980",
    ),
    FoundationPaper(
        world="public_goods",
        priority="core",
        role="conditional_cooperation",
        title="Conditional Cooperation and Voluntary Contributions to Public Goods",
        authors="Claudia Keser; Frans van Winden",
        year="2000",
        institution="reputation,information_restriction",
        mind="all",
        why_it_matters="Grounds contribution behavior in conditional cooperation rather than pure one-shot selfishness.",
        theory_obligation="Information/reputation variants should be judged by contribution response and sustainability.",
        code_result_check="Public Goods reputation and information_restriction outputs.",
    ),
    FoundationPaper(
        world="public_goods",
        priority="core",
        role="rewards_sanctions",
        title="The Effect of Rewards and Sanctions in Provision of Public Goods",
        authors="Martin Sefton; Robert Shupp; James M. Walker",
        year="2007",
        institution="penalty_schedule,contribution_matching",
        mind="all",
        why_it_matters="Directly maps to reward/sanction institution variants.",
        theory_obligation="Separate welfare-improving cooperation from costly reward accounting.",
        code_result_check="Public Goods institution-effect validator.",
    ),
    FoundationPaper(
        world="public_goods",
        priority="supporting",
        role="networked_public_goods",
        title="Cooperation and Contagion in Web-Based, Networked Public Goods Experiments",
        authors="Siddharth Suri; Duncan J. Watts",
        year="2011",
        institution="information_restriction",
        mind="all",
        why_it_matters="Supports the idea that interaction/network information changes cooperation.",
        theory_obligation="Information restrictions should be treated as strategic observability changes.",
        code_result_check="information_restriction institution and contribution/sustainability deltas.",
    ),
    FoundationPaper(
        world="resource_island",
        priority="must_read",
        role="commons",
        title="The Tragedy of the Commons",
        authors="Garrett Hardin",
        year="1968",
        institution="none",
        mind="all",
        why_it_matters="Basic common-resource overuse benchmark.",
        theory_obligation="Resource Island must report depletion/sustainability, not only agent reward.",
        code_result_check="resource_sustainability, survival, and gather/trade diagnostics.",
        doi="10.1126/science.162.3859.1243",
    ),
    FoundationPaper(
        world="resource_island",
        priority="must_read",
        role="common_pool_governance",
        title="Governing the Commons",
        authors="Elinor Ostrom",
        year="1990",
        institution="property_rights,reputation_system",
        mind="all",
        why_it_matters="Canonical common-pool institution design reference.",
        theory_obligation="Property/reputation mechanisms must have activation diagnostics before interpretation.",
        code_result_check="Resource Island v1 property opportunities, violations, and trade diagnostics.",
    ),
    FoundationPaper(
        world="resource_island",
        priority="core",
        role="design_principles",
        title="A Review of Design Principles for Community-based Natural Resource Management",
        authors="Michael Cox; Gwen Arnold; Sergio Villamayor-Tomas",
        year="2010",
        institution="property_rights,reputation_system",
        mind="all",
        why_it_matters="Operationalizes Ostrom-style design principles for natural-resource institutions.",
        theory_obligation="Check monitoring, boundaries, sanctions, and local incentive conditions.",
        code_result_check="Resource Island v1 contested layouts and activation thresholds.",
    ),
    FoundationPaper(
        world="resource_island",
        priority="core",
        role="sequential_social_dilemmas",
        title="Multi-agent Reinforcement Learning in Sequential Social Dilemmas",
        authors="Joel Z. Leibo; Vinicius Flores Zambaldi; Marc Lanctot; Janusz Marecki; Thore Graepel",
        year="2017",
        institution="all",
        mind="dqn,ppo,centralized_critic",
        why_it_matters="Closest MARL framing for spatial, temporally extended commons/public-goods dilemmas.",
        theory_obligation="Frame failed trade/institution activation as coordination/exploration evidence, not just low welfare.",
        code_result_check="Resource Island P.6 full trade-attempt and success diagnostics.",
    ),
    FoundationPaper(
        world="resource_island",
        priority="supporting",
        role="social_preferences",
        title="Inequity aversion improves cooperation in intertemporal social dilemmas",
        authors="Edward Hughes; Joel Z. Leibo; Matthew Phillips; Karl Tuyls; Edgar Duenas-Guzman; others",
        year="2018",
        institution="redistribution,reputation_system",
        mind="dqn,ppo,centralized_critic",
        why_it_matters="Shows how reward/social-preference modifications can change cooperation in sequential dilemmas.",
        theory_obligation="When institutions change rewards, check whether they alter state behavior or only accounting.",
        code_result_check="Resource Island and Public Goods welfare versus state-metric comparisons.",
    ),
    FoundationPaper(
        world="resource_island",
        priority="supporting",
        role="marl_benchmark",
        title="Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot",
        authors="Joel Z. Leibo et al.",
        year="2021",
        institution="all",
        mind="dqn,ppo,centralized_critic",
        why_it_matters="Benchmark suite for social generalization, resource sharing, and social dilemmas.",
        theory_obligation="Do not claim general social robustness from in-distribution training alone.",
        code_result_check="Cross-world synthesis and future held-out institution/world variants.",
    ),
    FoundationPaper(
        world="labor_market",
        priority="must_read",
        role="stable_matching",
        title="College Admissions and the Stability of Marriage",
        authors="David Gale; Lloyd S. Shapley",
        year="1962",
        institution="deferred_acceptance",
        mind="all",
        why_it_matters="Foundational deferred-acceptance/stable-matching paper.",
        theory_obligation="Verify matching validity and blocking-pair stability.",
        code_result_check="worlds/labor_market/benchmarks.py; labor_market_full stability metrics.",
        doi="10.1080/00029890.1962.11989827",
    ),
    FoundationPaper(
        world="labor_market",
        priority="must_read",
        role="incentives",
        title="The Economics of Matching: Stability and Incentives",
        authors="Alvin E. Roth",
        year="1982",
        institution="deferred_acceptance",
        mind="all",
        why_it_matters="Classical incentive result: stable mechanisms can be strategy-proof for one side but not both.",
        theory_obligation="Do not interpret worker-side misreport profits under worker-proposing DA as expected theory.",
        code_result_check="labor_market_benchmark_cases.json; manipulation-gain diagnostics.",
        doi="10.1287/moor.7.4.617",
    ),
    FoundationPaper(
        world="labor_market",
        priority="core",
        role="market_design_application",
        title="Changing the Boston School Choice Mechanism",
        authors="Atila Abdulkadiroglu; Parag A. Pathak; Alvin E. Roth; Tayfun Sonmez",
        year="2006",
        institution="deferred_acceptance",
        mind="q_learning",
        why_it_matters="Applied market-design case for replacing manipulable mechanisms with deferred acceptance.",
        theory_obligation="Use fixed benchmark cases to separate mechanism manipulation from noisy learned reports.",
        code_result_check="Labor Market fixed benchmark cases and full-run truthfulness/stability.",
    ),
    FoundationPaper(
        world="labor_market",
        priority="core",
        role="matching_overview",
        title="Deferred Acceptance Algorithms: History, Theory, Practice, and Open Questions",
        authors="Alvin E. Roth",
        year="2007",
        institution="deferred_acceptance",
        mind="all",
        why_it_matters="Compact theory/practice bridge for DA mechanisms.",
        theory_obligation="State which side proposes and what strategy-proofness claim is valid.",
        code_result_check="Labor Market DESIGN.md and benchmark cases.",
    ),
    FoundationPaper(
        world="labor_market",
        priority="supporting",
        role="learning_matching",
        title="Learn to Match with No Regret: Reinforcement Learning in Markov Matching Markets",
        authors="Yifei Min; Tianhao Wang; Ruitu Xu; Zhaoran Wang; Michael I. Jordan; Zhuoran Yang",
        year="2022",
        institution="deferred_acceptance",
        mind="dqn,ppo,centralized_critic",
        why_it_matters="Closest RL/matching-market anchor for future cross-mind labor-market work.",
        theory_obligation="If using deep RL in Labor Market, report regret/stability jointly, not reward alone.",
        code_result_check="Future Labor Market P.6 full ladder and synthesis.",
    ),
)


def google_scholar_url(query: str) -> str:
    return "https://scholar.google.com/scholar?" + urllib.parse.urlencode({"q": query})


def target_manual_text_path(paper: FoundationPaper, text_dir: Path) -> Path:
    return (
        text_dir
        / "manual"
        / f"{paper.world}__{paper.year}__{slugify(paper.title)}.txt"
    )


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    out = []
    for char in value.lower():
        out.append(char if char.isalnum() else " ")
    return " ".join("".join(out).split())


def _authors(record: dict[str, Any]) -> str:
    authors = record.get("authors") or []
    if isinstance(authors, str):
        return authors
    return "; ".join(str(author) for author in authors)


def _best_match(paper: FoundationPaper, records: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    candidates = [paper.title, *paper.aliases]
    if paper.doi:
        target_doi = paper.doi.lower().replace("https://doi.org/", "").strip()
        for record in records:
            doi = str(record.get("doi") or "").lower().replace("https://doi.org/", "").strip()
            if doi and doi == target_doi:
                return record, 1.0

    best: tuple[dict[str, Any] | None, float] = (None, 0.0)
    for record in records:
        title = str(record.get("title") or "")
        normalized_record_title = normalize_title(title)
        for candidate in candidates:
            normalized_candidate = normalize_title(candidate)
            if not normalized_candidate or not normalized_record_title:
                continue
            candidate_token_count = len(normalized_candidate.split())
            contains = (
                normalized_candidate in normalized_record_title
                or normalized_record_title in normalized_candidate
            )
            score = difflib.SequenceMatcher(None, normalized_candidate, normalized_record_title).ratio()
            if contains and candidate_token_count >= 4:
                score = max(score, 0.95)
            if normalized_record_title == normalize_title(paper.title):
                score += 0.08
            if str(record.get("year") or "") == paper.year:
                score += 0.04
            if score > best[1]:
                best = (record, score)
    if best[1] < 0.82:
        return None, best[1]
    return best


def build_foundation_matches(
    *,
    records_path: Path,
    text_dir: Path,
    foundations: Iterable[FoundationPaper] = FOUNDATION_PAPERS,
) -> list[FoundationMatch]:
    records = read_records(records_path) if records_path.exists() else []
    matches: list[FoundationMatch] = []
    for paper in foundations:
        record, score = _best_match(paper, records)
        source_basis = ""
        has_text = False
        manual_text_path = target_manual_text_path(paper, text_dir)
        if manual_text_path.exists() and manual_text_path.stat().st_size > 200:
            source_basis = f"text:{manual_text_path}"
            has_text = True
        if record:
            if not has_text:
                _source_text, source_basis = source_context_for_record(record, text_dir=text_dir, max_chars=200)
                has_text = source_basis.startswith("text:")
        text_path = source_basis.split(":", 1)[1] if has_text else ""
        cache_status = "found_exact_or_close" if record else "not_found_in_cache"
        if has_text:
            manual_action = "ready_for_llm"
        elif record and (record.get("pdf_url") or record.get("url") or record.get("doi")):
            manual_action = "fetch_pdf_or_paste_text"
        else:
            manual_action = "manual_search_needed"

        matches.append(
            FoundationMatch(
                world=paper.world,
                priority=paper.priority,
                role=paper.role,
                title=paper.title,
                authors=paper.authors,
                year=paper.year,
                institution=paper.institution,
                mind=paper.mind,
                why_it_matters=paper.why_it_matters,
                theory_obligation=paper.theory_obligation,
                code_result_check=paper.code_result_check,
                cache_status=cache_status,
                match_score=f"{min(score, 1.0):.3f}",
                matched_title=str(record.get("title") or "") if record else "",
                matched_authors=_authors(record) if record else "",
                matched_year=str(record.get("year") or "") if record else "",
                matched_doi=str(record.get("doi") or "") if record else paper.doi,
                url=str(record.get("url") or "") if record else "",
                pdf_url=str(record.get("pdf_url") or "") if record else "",
                has_extracted_text=has_text,
                text_path=text_path,
                manual_action=manual_action,
                google_scholar_url=google_scholar_url(paper.title),
            )
        )
    return matches


def write_foundation_csv(rows: Iterable[FoundationMatch], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [asdict(row) for row in rows]
    fields = list(materialized[0].keys()) if materialized else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _pipe(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def write_foundation_markdown(rows: list[FoundationMatch], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_world: dict[str, list[FoundationMatch]] = {}
    for row in rows:
        by_world.setdefault(row.world, []).append(row)

    lines = [
        "# Foundation Papers",
        "",
        "This is the simplified theory list: the papers that should ground the code and results.",
        "It is curated from the project theory obligations, then matched against the local scout cache.",
        "",
        "Status meanings:",
        "",
        "- `ready_for_llm`: extracted text is already available under `literature/text/`.",
        "- `fetch_pdf_or_paste_text`: metadata exists, but the paper text is not extracted yet.",
        "- `manual_search_needed`: this anchor was not found in the current metadata cache.",
        "",
    ]
    for world in sorted(by_world):
        lines.extend([f"## {world}", ""])
        lines.append(
            "| Priority | Role | Paper | Cache | Text | Manual action | Why it matters | Code/result check |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in by_world[world]:
            paper = f"{row.authors} ({row.year}). {row.title}"
            text_status = row.text_path if row.has_extracted_text else "missing"
            lines.append(
                "| "
                + " | ".join(
                    _pipe(part)
                    for part in (
                        row.priority,
                        row.role,
                        paper,
                        row.cache_status,
                        text_status,
                        row.manual_action,
                        row.why_it_matters,
                        row.code_result_check,
                    )
                )
                + " |"
            )
        lines.append("")

    missing = [row for row in rows if row.manual_action != "ready_for_llm"]
    lines.extend(
        [
            "## Manual PDF Queue",
            "",
            "Bring these PDFs or paste their full text first. They are the highest-value gaps.",
            "",
            "| World | Priority | Paper | Action | URL | PDF URL | Scholar |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    priority_order = {"must_read": 0, "core": 1, "supporting": 2}
    for row in sorted(missing, key=lambda r: (priority_order.get(r.priority, 9), r.world, r.title)):
        paper = f"{row.authors} ({row.year}). {row.title}"
        lines.append(
            "| "
            + " | ".join(
                _pipe(part)
                for part in (
                    row.world,
                    row.priority,
                    paper,
                    row.manual_action,
                    row.url,
                    row.pdf_url,
                    row.google_scholar_url,
                )
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_foundation_pdf_queue(rows: Iterable[FoundationMatch], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    queue_rows = [asdict(row) for row in rows if row.manual_action != "ready_for_llm"]
    fields = [
        "world",
        "priority",
        "role",
        "title",
        "authors",
        "year",
        "manual_action",
        "cache_status",
        "matched_title",
        "matched_doi",
        "url",
        "pdf_url",
        "google_scholar_url",
        "why_it_matters",
        "theory_obligation",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in queue_rows:
            writer.writerow({field: row.get(field, "") for field in fields})
