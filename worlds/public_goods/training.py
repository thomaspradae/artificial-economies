from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

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
from worlds.public_goods.benchmarks import free_rider_benchmark, social_optimum_benchmark
from worlds.public_goods.env import N_ACTIONS, PublicGoodsConfig, PublicGoodsWorld
from worlds.public_goods.features import public_goods_obs_dim, structured_observations


SUPPORTED_PUBLIC_GOODS_MINDS = SUPPORTED_LADDER_MINDS


SUMMARY_METRICS = (
    "pool_stock",
    "sustainability",
    "contribution_total",
    "extraction_total",
    "contribution_rate",
    "extraction_rate",
    "collapse_rate",
    "reward_total",
    "welfare",
    "inequality",
    "matched_contribution",
    "penalty_total",
    "reputation_bonus_total",
    "tax_revenue",
    "contributor_count",
    "extractor_count",
)


@dataclass
class PublicGoodsRun:
    records: list[dict[str, Any]]
    world: PublicGoodsWorld
    agents: list[Agent]
    final_epsilon: float
    mind: str = "q_learning"
    obs_dim: int | None = None


def build_public_goods_q_world(
    seed: int,
    institution: str = "none",
    config: PublicGoodsConfig | None = None,
    institution_params: dict[str, Any] | None = None,
) -> PublicGoodsWorld:
    cfg = config if config is not None else PublicGoodsConfig()
    return build_experiment(
        {
            "world": "public_goods",
            "institution": institution,
            "institution_params": institution_params or {},
            "seed": seed,
            "world_params": {"config": cfg},
            "agents": [
                {
                    "mind": "q_learning",
                    "params": {
                        "n_prices": N_ACTIONS,
                        "seed": seed + agent_id,
                        "state_shape": (N_ACTIONS, N_ACTIONS),
                    },
                }
                for agent_id in range(cfg.n_agents)
            ],
        }
    )


def build_public_goods_world(
    seed: int,
    institution: str = "none",
    config: PublicGoodsConfig | None = None,
    institution_params: dict[str, Any] | None = None,
    mind: str = "q_learning",
) -> PublicGoodsWorld:
    if mind == "q_learning":
        return build_public_goods_q_world(
            seed=seed,
            institution=institution,
            config=config,
            institution_params=institution_params,
        )
    if mind not in SUPPORTED_PUBLIC_GOODS_MINDS:
        raise ValueError(f"Unsupported Public Goods mind {mind!r}")
    cfg = config if config is not None else PublicGoodsConfig()
    ladder = build_neural_ladder_agents(
        mind,
        n_agents=cfg.n_agents,
        action_dim=N_ACTIONS,
        obs_dim=public_goods_obs_dim(),
        seed=seed,
    )
    world = build_experiment(
        {
            "world": "public_goods",
            "institution": institution,
            "institution_params": institution_params or {},
            "seed": seed,
            "world_params": {"config": cfg},
            "n_agents": 0,
        }
    )
    world.agents = ladder.agents
    world.joint_learner = ladder.joint_learner
    return world


def _observations(world: PublicGoodsWorld, mind: str) -> list[Any]:
    if mind == "q_learning":
        return world.observations()
    obs = structured_observations(world)
    assert_finite_feature_matrix(obs)
    return obs


def train_public_goods(
    steps: int = 1000,
    seed: int = 0,
    institution: str = "none",
    institution_params: dict[str, Any] | None = None,
    config: PublicGoodsConfig | None = None,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    mind: str = "q_learning",
) -> PublicGoodsRun:
    if steps < 1:
        raise ValueError("steps must be positive")
    cfg = replace(config if config is not None else PublicGoodsConfig(), max_rounds=steps)
    world = build_public_goods_world(
        seed=seed,
        institution=institution,
        config=cfg,
        institution_params=institution_params,
        mind=mind,
    )
    obs = _observations(world, mind)
    epsilon = float(epsilon_start)
    records: list[dict[str, Any]] = []
    for step in range(steps):
        actions = [
            act_ladder_agent(agent, obs[agent_id], epsilon)
            for agent_id, agent in enumerate(world.agents)
        ]
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
        record["epsilon"] = float(epsilon if mind == "q_learning" else np.nan)
        records.append(record)
        obs = next_obs
        epsilon = max(float(epsilon_min), epsilon * float(epsilon_decay))
    return PublicGoodsRun(
        records=records,
        world=world,
        agents=list(world.agents),
        final_epsilon=epsilon,
        mind=mind,
        obs_dim=public_goods_obs_dim() if mind != "q_learning" else None,
    )


def summarize_records(records: list[dict[str, Any]], final_window: int = 100) -> dict[str, float]:
    if not records:
        raise ValueError("cannot summarize empty Public Goods records")
    window = records[-min(len(records), final_window) :]
    return {metric: finite_mean(float(record[metric]) for record in window) for metric in SUMMARY_METRICS}


def benchmark_for_config(config: PublicGoodsConfig, steps: int = 100) -> dict[str, float]:
    free = free_rider_benchmark(config, steps=steps)
    social = social_optimum_benchmark(config, steps=steps)
    return {
        "free_rider_welfare": free["welfare"],
        "free_rider_sustainability": free["sustainability"],
        "free_rider_collapse_rate": free["collapse_rate"],
        "social_welfare": social["welfare"],
        "social_sustainability": social["sustainability"],
        "social_collapse_rate": social["collapse_rate"],
    }
