from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from core.agent import Agent
from core.metrics import finite_mean
from core.registry import build_experiment
import institutions  # noqa: F401
import minds  # noqa: F401
from worlds.mind_ladder import (
    SUPPORTED_LADDER_MINDS,
    act_ladder_agent,
    assert_finite_feature_matrix,
    build_neural_ladder_agents,
    update_ladder_agents,
)
from worlds.labor_market.benchmarks import truthful_matching
from worlds.labor_market.env import LaborMarketConfig, LaborMarketWorld
from worlds.labor_market.features import labor_market_obs_dim, structured_observations


SUPPORTED_LABOR_MARKET_MINDS = SUPPORTED_LADDER_MINDS


SUMMARY_METRICS = (
    "match_rate",
    "worker_welfare",
    "employer_welfare",
    "total_welfare",
    "blocking_pairs",
    "stability",
    "truthful_report_rate",
    "manipulation_gain_mean",
    "matched_worker_utility_mean",
    "unmatched_count",
)


@dataclass
class LaborMarketRun:
    records: list[dict[str, Any]]
    world: LaborMarketWorld
    agents: list[Agent]
    final_epsilon: float
    mind: str = "q_learning"
    obs_dim: int | None = None


def build_labor_market_q_world(seed: int, config: LaborMarketConfig | None = None) -> LaborMarketWorld:
    cfg = config if config is not None else LaborMarketConfig()
    return build_experiment(
        {
            "world": "labor_market",
            "institution": "deferred_acceptance",
            "seed": seed,
            "world_params": {"config": cfg},
            "agents": [
                {
                    "mind": "q_learning",
                    "params": {
                        "n_prices": cfg.n_employers,
                        "seed": seed + worker,
                        "state_shape": (cfg.n_employers, cfg.n_employers),
                    },
                }
                for worker in range(cfg.n_workers)
            ],
        }
    )


def build_labor_market_world(
    seed: int,
    config: LaborMarketConfig | None = None,
    mind: str = "q_learning",
) -> LaborMarketWorld:
    if mind == "q_learning":
        return build_labor_market_q_world(seed=seed, config=config)
    if mind not in SUPPORTED_LABOR_MARKET_MINDS:
        raise ValueError(f"Unsupported Labor Market mind {mind!r}")
    cfg = config if config is not None else LaborMarketConfig()
    obs_dim = labor_market_obs_dim(cfg.n_employers)
    ladder = build_neural_ladder_agents(
        mind,
        n_agents=cfg.n_workers,
        action_dim=cfg.n_employers,
        obs_dim=obs_dim,
        seed=seed,
    )
    world = build_experiment(
        {
            "world": "labor_market",
            "institution": "deferred_acceptance",
            "seed": seed,
            "world_params": {"config": cfg},
            "n_agents": 0,
        }
    )
    world.agents = ladder.agents
    world.joint_learner = ladder.joint_learner
    return world


def _observations(world: LaborMarketWorld, mind: str) -> list[Any]:
    if mind == "q_learning":
        return world.observations()
    obs = structured_observations(world)
    assert_finite_feature_matrix(obs)
    return obs


def train_labor_market(
    steps: int = 1000,
    seed: int = 0,
    config: LaborMarketConfig | None = None,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    mind: str = "q_learning",
) -> LaborMarketRun:
    if steps < 1:
        raise ValueError("steps must be positive")
    cfg = replace(config if config is not None else LaborMarketConfig(), max_rounds=steps)
    world = build_labor_market_world(seed=seed, config=cfg, mind=mind)
    obs = _observations(world, mind)
    epsilon = float(epsilon_start)
    records: list[dict[str, Any]] = []
    for step in range(steps):
        actions = [act_ladder_agent(agent, obs[worker], epsilon) for worker, agent in enumerate(world.agents)]
        _, rewards, done, info = world.step(actions)
        next_obs = _observations(world, mind)
        update_ladder_agents(
            world.agents,
            getattr(world, "joint_learner", None),
            obs,
            actions,
            rewards,
            next_obs,
            done,
        )
        record = dict(info)
        record["train_step"] = float(step)
        record["epsilon"] = float(epsilon if mind == "q_learning" else float("nan"))
        records.append(record)
        obs = next_obs
        epsilon = max(float(epsilon_min), epsilon * float(epsilon_decay))
    return LaborMarketRun(
        records=records,
        world=world,
        agents=list(world.agents),
        final_epsilon=epsilon,
        mind=mind,
        obs_dim=labor_market_obs_dim(cfg.n_employers) if mind != "q_learning" else None,
    )


def summarize_records(records: list[dict[str, Any]], final_window: int = 100) -> dict[str, float]:
    if not records:
        raise ValueError("cannot summarize empty Labor Market records")
    window = records[-min(len(records), final_window) :]
    return {metric: finite_mean(float(record[metric]) for record in window) for metric in SUMMARY_METRICS}


def benchmark_for_config(config: LaborMarketConfig) -> dict[str, float]:
    world = LaborMarketWorld(config=config, seed=0)
    bench = truthful_matching(world.worker_values, world.employer_values)
    matches = bench["matches"]
    worker_welfare = 0.0
    employer_welfare = 0.0
    for worker, employer in enumerate(matches):
        if employer >= 0:
            worker_welfare += float(world.worker_values[worker, employer])
            employer_welfare += float(world.employer_values[employer, worker])
    return {
        "truthful_worker_welfare": worker_welfare,
        "truthful_employer_welfare": employer_welfare,
        "truthful_total_welfare": worker_welfare + employer_welfare,
        "truthful_blocking_pairs": float(len(bench["blocking_pairs"])),
        "truthful_match_rate": float(sum(1 for employer in matches if employer >= 0) / len(matches)),
    }
