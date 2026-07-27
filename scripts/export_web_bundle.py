from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = REPO_ROOT / "outputs/mechanism_traces_local_full/mechanism_trace_steps.csv"
BUDGET_PATH = REPO_ROOT / "outputs/pricing_quantity_margin_audit/dqn_budget_ladder.csv"
SEED_DELTA_PATH = REPO_ROOT / "outputs/pricing_quantity_margin_audit/existing_full_seed_deltas.csv"
MIND_COMPARISON_PATH = REPO_ROOT / "outputs/phase3_full/mind_comparison.csv"
OUTPUT_PATH = REPO_ROOT / "explorer/public/data/pricing-cap-reversal/scenario.json"
ALL_WORLD_OUTPUT_PATH = REPO_ROOT / "explorer/public/data/all-world-ladder/summary.json"
WEB_FRAMES_PER_TRACE = 180
MIND_ORDER = ["q_learning", "dqn", "ppo", "independent_dqn", "centralized_critic"]

WORLD_TABLES = [
    {
        "id": "pricing_arena",
        "title": "Pricing Arena",
        "source_path": "outputs/pricing_arena_phase3_full/mind_comparison.csv",
        "institution_key": "mechanism",
        "baseline_institution": "none",
        "institution_label": "mechanism",
        "summary": "Repeated oligopoly pricing with regulatory mechanisms and exploitability audits.",
        "interpretation": (
            "The price-cap result is not a generic DQN anecdote: Q-learning, PPO, independent-DQN, "
            "and centralized-critic are all in the table, and the safe claim is budget-dependent. "
            "DQN-family learners can recover profits under the cap after enough training while exploitability falls."
        ),
        "metrics": [
            {"key": "welfare_mean", "label": "welfare", "better": "higher"},
            {"key": "profit_total_mean", "label": "firm profit", "better": "context"},
            {"key": "profit_collusion_index_mean", "label": "profit collusion", "better": "lower"},
            {"key": "exploitability_mean", "label": "exploitability", "better": "lower"},
        ],
    },
    {
        "id": "resource_island_v1",
        "title": "Resource Island v1",
        "source_path": "outputs/resource_island_v1_phase3_full/mind_comparison.csv",
        "institution_key": "institution",
        "baseline_institution": "none",
        "institution_label": "institution",
        "summary": "Spatial inventory economy with contested resources, specialization pressure, and unequal trades.",
        "interpretation": (
            "The v1 ladder reverses the earlier v0 non-trade artifact. PPO and centralized-critic activate "
            "the trade/reputation channel strongly; DQN and independent-DQN trade less but remain distinct."
        ),
        "metrics": [
            {"key": "welfare_mean", "label": "welfare", "better": "higher"},
            {"key": "survival_rate_mean", "label": "survival", "better": "higher"},
            {"key": "trade_count_mean", "label": "successful trades", "better": "context"},
            {"key": "trade_institution_blocked_count_mean", "label": "institution blocks", "better": "context"},
        ],
    },
    {
        "id": "auction_house",
        "title": "Auction House",
        "source_path": "outputs/auction_house_phase3_full/mind_comparison.csv",
        "institution_key": "scenario",
        "baseline_institution": "second_price",
        "institution_label": "auction",
        "summary": "Private-value auctions with first-price, second-price, reserve, clock, and information variants.",
        "interpretation": (
            "The ladder is explicitly not monotone. PPO has low regret in second-price/reserve cases, "
            "while centralized-critic is unstable in no-reserve second-price and first-price settings."
        ),
        "metrics": [
            {"key": "revenue_mean", "label": "seller revenue", "better": "context"},
            {"key": "welfare_mean", "label": "welfare", "better": "higher"},
            {"key": "allocative_efficiency_mean", "label": "efficiency", "better": "higher"},
            {"key": "ex_post_regret_mean_mean", "label": "ex-post regret", "better": "lower"},
        ],
    },
    {
        "id": "public_goods",
        "title": "Public Goods",
        "source_path": "outputs/public_goods_phase3_full/mind_comparison.csv",
        "institution_key": "institution",
        "baseline_institution": "none",
        "institution_label": "institution",
        "summary": "Commons economy with contribution, extraction, regeneration, collapse, and reward institutions.",
        "interpretation": (
            "PPO and centralized-critic nearly stop contributing under baseline and collapse often. "
            "DQN-family learners contribute more, and contribution matching creates the clearest discoverable incentive."
        ),
        "metrics": [
            {"key": "sustainability_mean", "label": "sustainability", "better": "higher"},
            {"key": "contribution_total_mean", "label": "contribution", "better": "higher"},
            {"key": "welfare_mean", "label": "welfare", "better": "higher"},
            {"key": "collapse_rate_mean", "label": "collapse rate", "better": "lower"},
        ],
    },
    {
        "id": "labor_market",
        "title": "Labor Market",
        "source_path": "outputs/labor_market_phase3_full/mind_comparison.csv",
        "institution_key": "institution",
        "baseline_institution": "deferred_acceptance",
        "institution_label": "mechanism",
        "summary": "Worker-report matching market with deferred acceptance, stability, truthfulness, and welfare diagnostics.",
        "interpretation": (
            "This is the strongest MARL warning. Centralized-critic has the highest welfare but much weaker "
            "stability and truthfulness, so the headline is a metric tradeoff, not simple improvement."
        ),
        "metrics": [
            {"key": "total_welfare_mean", "label": "welfare", "better": "higher"},
            {"key": "stability_mean", "label": "stability", "better": "higher"},
            {"key": "truthful_report_rate_mean", "label": "truthfulness", "better": "higher"},
            {"key": "manipulation_gain_mean_mean", "label": "manipulation gain", "better": "lower"},
        ],
    },
]

ARCHITECTURE_AXIS = [
    {
        "mind": "q_learning",
        "label": "Q-learning",
        "role": "tabular control",
        "assumption": "finite table; no function approximation",
        "whatItTests": "whether the world reproduces clean tabular learning behavior before neural approximation is introduced",
    },
    {
        "mind": "dqn",
        "label": "DQN",
        "role": "value approximation",
        "assumption": "neural Q approximation with replay and target networks",
        "whatItTests": "whether institution effects survive approximate off-policy value learning",
    },
    {
        "mind": "ppo",
        "label": "PPO",
        "role": "on-policy policy optimization",
        "assumption": "stochastic clipped policy-gradient updates instead of replayed value learning",
        "whatItTests": "whether the same economic incentives remain discoverable under on-policy optimization",
    },
    {
        "mind": "independent_dqn",
        "label": "Independent DQN",
        "role": "independent learners",
        "assumption": "each agent learns separately with decorrelated stochastic streams",
        "whatItTests": "whether DQN-like results persist once agents are independent learners rather than an alias condition",
    },
    {
        "mind": "centralized_critic",
        "label": "Centralized critic",
        "role": "centralized-training scaffold",
        "assumption": "centralized value information during learning with decentralized actions",
        "whatItTests": "whether coordination signals improve outcomes or create metric tradeoffs",
    },
]


@dataclass(frozen=True)
class TraceKey:
    mind: str
    institution: str


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: Any, digits: int = 6) -> float:
    return round(_float(value), digits)


def _read_trace_frames() -> dict[str, dict[str, list[dict[str, Any]]]]:
    wanted = {
        TraceKey("q_learning", "none"),
        TraceKey("q_learning", "price_cap"),
        TraceKey("dqn", "none"),
        TraceKey("dqn", "price_cap"),
    }
    traces: dict[str, dict[str, list[dict[str, Any]]]] = {
        key.mind: {"none": [], "price_cap": []} for key in wanted
    }
    with TRACE_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["world"] != "pricing_arena" or row["seed"] != "0" or row["n_agents"] != "2":
                continue
            key = TraceKey(row["mind"], row["institution"])
            if key not in wanted:
                continue
            payload = json.loads(row["metric_payload"])
            frame = _frame_from_payload(row, payload)
            traces[key.mind][key.institution].append(frame)

    for mind, by_institution in traces.items():
        for institution, frames in by_institution.items():
            if not frames:
                raise RuntimeError(f"missing trace frames for {mind}/{institution}")
            frames.sort(key=lambda item: item["step"])
            by_institution[institution] = _even_sample(frames, WEB_FRAMES_PER_TRACE)
    return traces


def _even_sample(frames: list[dict[str, Any]], max_frames: int) -> list[dict[str, Any]]:
    if len(frames) <= max_frames:
        return frames
    if max_frames <= 1:
        return [frames[-1]]
    indices = {
        round(index * (len(frames) - 1) / (max_frames - 1))
        for index in range(max_frames)
    }
    return [frames[index] for index in sorted(indices)]


def _frame_from_payload(row: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    agents = []
    for index in (1, 2):
        requested = _round(payload.get(f"raw_p{index}", payload.get(f"p{index}")))
        executed = _round(payload.get(f"p{index}"))
        if row["institution"] == "price_cap" and requested > executed:
            events.append(
                {
                    "type": "price_cap_bound",
                    "agentId": f"firm_{index}",
                    "requested": requested,
                    "executed": executed,
                }
            )
        agents.append(
            {
                "id": f"firm_{index}",
                "action": {"requestedPrice": requested, "price": executed},
                "reward": _round(payload.get(f"reward{index}", payload.get(f"profit{index}"))),
                "quantity": _round(payload.get(f"quantity{index}")),
                "profit": _round(payload.get(f"profit{index}")),
            }
        )
    return {
        "step": int(_float(row["step"])),
        "agents": agents,
        "events": events,
        "metrics": {
            "price": _round(payload.get("avg_price")),
            "quantity": _round(payload.get("quantity_total")),
            "profit": _round(payload.get("profit_total")),
            "consumerSurplus": _round(payload.get("consumer_surplus")),
            "welfare": _round(payload.get("welfare")),
            "collusionIndex": _round(payload.get("collusion_index")),
            "priceDispersion": _round(payload.get("price_dispersion")),
            "margin": _round(payload.get("avg_price")) - 1.0,
        },
    }


def _read_budget_ladder() -> list[dict[str, Any]]:
    rows = []
    with BUDGET_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["mind"] != "dqn":
                continue
            rows.append(
                {
                    "trainingSteps": int(row["budget_steps"]),
                    "nSeeds": int(row["n"]),
                    "profitDeltaMean": _round(row["delta_profit_total_mean"]),
                    "profitDeltaCi95Low": _round(row["delta_profit_total_ci95_low"]),
                    "profitDeltaCi95High": _round(row["delta_profit_total_ci95_high"]),
                    "quantityDeltaMean": _round(row["delta_quantity_total_mean"]),
                    "priceDeltaMean": _round(row["delta_avg_price_mean"]),
                    "welfareDeltaMean": _round(row["delta_welfare_mean"]),
                    "positiveProfitShare": _round(row["positive_profit_delta_share"]),
                    "minProfitDelta": _round(row["min_delta_profit_total"]),
                    "maxProfitDelta": _round(row["max_delta_profit_total"]),
                    "source": row["source"],
                }
            )
    rows.sort(key=lambda item: item["trainingSteps"])
    return rows


def _read_seed_deltas() -> list[dict[str, Any]]:
    rows = []
    with SEED_DELTA_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["mind"] != "dqn":
                continue
            rows.append(
                {
                    "seed": int(row["seed"]),
                    "profitDelta": _round(row["delta_profit_total"]),
                    "quantityDelta": _round(row["delta_quantity_total"]),
                    "priceDelta": _round(row["delta_avg_price"]),
                    "welfareDelta": _round(row["delta_welfare"]),
                    "priceCapProfitHigher": bool(int(_float(row["price_cap_profit_higher"]))),
                }
            )
    rows.sort(key=lambda item: item["seed"])
    return rows


def _read_summary() -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    with MIND_COMPARISON_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["mind"] not in {"q_learning", "dqn"} or row["mechanism"] not in {"none", "price_cap"}:
                continue
            summary.setdefault(row["mind"], {})[row["mechanism"]] = {
                "avgPrice": _round(row["avg_price_mean"]),
                "profit": _round(row["profit_total_mean"]),
                "quantity": _round(row["quantity_total_mean"]),
                "welfare": _round(row["welfare_mean"]),
                "consumerSurplus": _round(row["consumer_surplus_mean"]),
                "exploitability": _round(row["exploitability_mean"]),
                "collusionIndex": _round(row["collusion_index_mean"]),
                "profitCollusionIndex": _round(row["profit_collusion_index_mean"]),
            }
    return summary


def _read_world_table(config: dict[str, Any]) -> dict[str, Any]:
    path = REPO_ROOT / config["source_path"]
    rows: list[dict[str, Any]] = []
    institutions: list[str] = []
    minds_seen: list[str] = []
    metric_keys = [metric["key"] for metric in config["metrics"]]

    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            mind = raw.get("mind", "")
            if mind not in MIND_ORDER:
                continue
            institution = raw.get(config["institution_key"], raw.get("institution", ""))
            if not institution:
                continue
            if institution not in institutions:
                institutions.append(institution)
            if mind not in minds_seen:
                minds_seen.append(mind)
            metrics = {
                key: _round(raw[key])
                for key in metric_keys
                if key in raw and raw[key] not in {"", "nan", "NaN", "None"}
            }
            rows.append(
                {
                    "world": config["id"],
                    "mind": mind,
                    "mindLabel": _mind_label(mind),
                    "institution": institution,
                    "metrics": metrics,
                    "sourceDir": raw.get("source_dir", raw.get("raw_source_dir", "")),
                    "nSeeds": int(_float(raw.get("n_seeds", 20), 20)),
                }
            )

    rows.sort(key=lambda row: (_mind_sort_key(row["mind"]), row["institution"]))
    return {
        "id": config["id"],
        "title": config["title"],
        "summary": config["summary"],
        "interpretation": config["interpretation"],
        "sourcePath": config["source_path"],
        "institutionLabel": config["institution_label"],
        "baselineInstitution": config["baseline_institution"],
        "institutions": institutions,
        "minds": [mind for mind in MIND_ORDER if mind in minds_seen],
        "metrics": config["metrics"],
        "rows": rows,
    }


def _mind_label(mind: str) -> str:
    return {
        "q_learning": "Q-learning",
        "dqn": "DQN",
        "ppo": "PPO",
        "independent_dqn": "Independent DQN",
        "centralized_critic": "Centralized critic",
    }.get(mind, mind)


def _mind_sort_key(mind: str) -> int:
    try:
        return MIND_ORDER.index(mind)
    except ValueError:
        return len(MIND_ORDER)


def build_all_world_bundle() -> dict[str, Any]:
    worlds = [_read_world_table(config) for config in WORLD_TABLES]
    row_count = sum(len(world["rows"]) for world in worlds)
    return {
        "id": "all-world-ladder",
        "version": "1.0.0",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "title": "Five Worlds, Five Learner Architectures",
        "subtitle": (
            "The project is not just DQN in Pricing Arena. These are the thesis-facing full-ladder outputs "
            "for Pricing Arena, Resource Island v1, Auction House, Public Goods, and Labor Market."
        ),
        "architectureAxis": ARCHITECTURE_AXIS,
        "worlds": worlds,
        "coverage": {
            "worldCount": len(worlds),
            "mindCount": len(MIND_ORDER),
            "rowCount": row_count,
            "fullRunScale": "n=20 where applicable, 40k training steps in thesis-facing full runs",
        },
        "sourceArtifacts": [config["source_path"] for config in WORLD_TABLES],
    }


def build_bundle() -> dict[str, Any]:
    return {
        "id": "pricing-cap-reversal",
        "version": "1.0.0",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "title": "Can a price cap increase firm profits?",
        "experiment": {
            "world": "pricing_arena",
            "institutions": ["none", "price_cap"],
            "learners": ["q_learning", "dqn"],
            "seed": 0,
            "nAgents": 2,
            "evaluationFramesPerTrace": WEB_FRAMES_PER_TRACE,
            "source": "validated offline Python outputs",
        },
        "benchmarks": {
            "nashPrice": 2.5,
            "monopolyPrice": 9.0,
            "priceCap": 5.0,
            "unitCost": 1.0,
        },
        "traces": _read_trace_frames(),
        "budgetLadder": _read_budget_ladder(),
        "pairedSeedDeltas": _read_seed_deltas(),
        "summary": _read_summary(),
        "annotations": [
            {
                "id": "setup",
                "step": 500,
                "text": "Two firms repeatedly choose prices. Consumers flow toward cheaper firms, and profit is margin times quantity.",
            },
            {
                "id": "cap",
                "step": 520,
                "text": "Under the price cap, requested prices above the cap are visibly clipped before demand and profit are computed.",
            },
            {
                "id": "trace-warning",
                "step": 700,
                "text": "This replay is one evaluation trace. The claim comes from paired seed distributions and the training-budget audit.",
            },
        ],
        "sourceArtifacts": [
            "outputs/mechanism_traces_local_full/mechanism_trace_steps.csv",
            "outputs/pricing_quantity_margin_audit/dqn_budget_ladder.csv",
            "outputs/pricing_quantity_margin_audit/existing_full_seed_deltas.csv",
            "outputs/phase3_full/mind_comparison.csv",
        ],
    }


def main() -> None:
    bundle = build_bundle()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    all_world_bundle = build_all_world_bundle()
    ALL_WORLD_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALL_WORLD_OUTPUT_PATH.write_text(json.dumps(all_world_bundle, indent=2, sort_keys=True) + "\n")
    print(f"[wrote] {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"[wrote] {ALL_WORLD_OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(
        "[bundle] "
        f"{sum(len(inst) for traces in bundle['traces'].values() for inst in traces.values())} frames, "
        f"{len(bundle['budgetLadder'])} budget rows, "
        f"{len(bundle['pairedSeedDeltas'])} paired seed deltas"
    )
    print(
        "[all-world] "
        f"{all_world_bundle['coverage']['worldCount']} worlds, "
        f"{all_world_bundle['coverage']['mindCount']} learner architectures, "
        f"{all_world_bundle['coverage']['rowCount']} comparison rows"
    )


if __name__ == "__main__":
    main()
