from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from core.agent import Agent
from core.metrics import finite_mean
from core.registry import build_experiment
import institutions  # noqa: F401 - registers institutions
import minds  # noqa: F401 - registers minds
from minds.q_learning import QLearningMind
from worlds.mind_ladder import (
    SUPPORTED_LADDER_MINDS,
    act_ladder_agent,
    assert_finite_feature_matrix,
    build_neural_ladder_agents,
    update_ladder_agents,
)
from worlds.auction_house.benchmarks import (
    expected_outcome_over_grid,
    first_price_equilibrium_bid,
    truthful_bid,
)
from worlds.auction_house.env import AuctionHouseConfig, AuctionHouseWorld
from worlds.auction_house.features import auction_house_obs_dim, structured_observations


SUPPORTED_AUCTION_MINDS = SUPPORTED_LADDER_MINDS

SUMMARY_METRICS = (
    "revenue",
    "bidder_surplus",
    "welfare",
    "total_welfare",
    "max_possible_welfare",
    "allocative_efficiency",
    "welfare_efficiency",
    "allocation_error",
    "mean_value",
    "mean_bid",
    "mean_bid_to_value",
    "truthful_bid_distance_mean",
    "first_price_shading_distance_mean",
    "ex_post_regret_mean",
    "ex_post_regret_max",
    "overbid_rate",
    "underbid_rate",
    "no_sale",
)


@dataclass
class AuctionHouseRun:
    records: list[dict[str, Any]]
    world: AuctionHouseWorld
    agents: list[Agent]
    final_epsilon: float
    mind: str = "q_learning"
    obs_dim: int | None = None


def build_auction_house_q_world(
    seed: int,
    config: AuctionHouseConfig | None = None,
    institution: str = "none",
    institution_params: dict[str, Any] | None = None,
) -> AuctionHouseWorld:
    """Build Auction House with tabular Q-learning over private value bins."""
    cfg = config if config is not None else AuctionHouseConfig()
    return build_experiment(
        {
            "world": "auction_house",
            "institution": institution,
            "institution_params": institution_params or {},
            "seed": seed,
            "world_params": {"config": cfg},
            "agents": [
                {
                    "mind": "q_learning",
                    "params": {
                        "n_prices": len(cfg.bid_grid),
                        "seed": seed + bidder_id,
                        "state_shape": (len(cfg.valuation_grid),),
                    },
                }
                for bidder_id in range(cfg.n_bidders)
            ],
        }
    )


def build_auction_house_world(
    seed: int,
    config: AuctionHouseConfig | None = None,
    institution: str = "none",
    institution_params: dict[str, Any] | None = None,
    mind: str = "q_learning",
) -> AuctionHouseWorld:
    """Build Auction House with tabular Q or structured neural/MARL minds."""
    if mind == "q_learning":
        return build_auction_house_q_world(
            seed=seed,
            config=config,
            institution=institution,
            institution_params=institution_params,
        )
    if mind not in SUPPORTED_AUCTION_MINDS:
        raise ValueError(f"Unsupported Auction House mind {mind!r}")
    cfg = config if config is not None else AuctionHouseConfig()
    ladder = build_neural_ladder_agents(
        mind,
        n_agents=cfg.n_bidders,
        action_dim=len(cfg.bid_grid),
        obs_dim=auction_house_obs_dim(),
        seed=seed,
    )
    world = build_experiment(
        {
            "world": "auction_house",
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


def benchmark_for_config(config: AuctionHouseConfig) -> dict[str, float]:
    """Economically relevant reference outcome for the configured mechanism."""
    if config.auction_format in ("second_price", "clock"):
        return expected_outcome_over_grid(
            config.valuation_grid,
            n_bidders=config.n_bidders,
            bid_strategy=lambda value: truthful_bid(value, config.bid_grid),
            auction_format="second_price",
            reserve_price=config.reserve_price,
        )
    return expected_outcome_over_grid(
        config.valuation_grid,
        n_bidders=config.n_bidders,
        bid_strategy=lambda value: first_price_equilibrium_bid(
            value,
            n_bidders=config.n_bidders,
            valuation_low=config.valuation_low,
            bid_grid=config.bid_grid,
        ),
        auction_format="first_price",
        reserve_price=config.reserve_price,
    )


def _observations(world: AuctionHouseWorld, mind: str) -> list[Any]:
    if mind == "q_learning":
        return world.observations()
    obs = structured_observations(world)
    assert_finite_feature_matrix(obs)
    return obs


def train_auction_house(
    steps: int = 1000,
    seed: int = 0,
    config: AuctionHouseConfig | None = None,
    institution: str = "none",
    institution_params: dict[str, Any] | None = None,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.999,
    mind: str = "q_learning",
) -> AuctionHouseRun:
    """Train a tabular bidder population in repeated sealed-bid auctions."""
    if mind not in SUPPORTED_AUCTION_MINDS:
        raise ValueError(f"Unsupported Auction House mind {mind!r}")
    if steps < 1:
        raise ValueError("steps must be positive")

    cfg = config if config is not None else AuctionHouseConfig()
    cfg = replace(cfg, max_rounds=steps)
    world = build_auction_house_world(
        seed=seed,
        config=cfg,
        institution=institution,
        institution_params=institution_params,
        mind=mind,
    )
    obs = _observations(world, mind)
    epsilon = float(epsilon_start)
    records: list[dict[str, Any]] = []

    for step in range(steps):
        actions = [
            act_ladder_agent(agent, obs[bidder_id], epsilon)
            for bidder_id, agent in enumerate(world.agents)
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

    return AuctionHouseRun(
        records=records,
        world=world,
        agents=list(world.agents),
        final_epsilon=epsilon,
        mind=mind,
        obs_dim=auction_house_obs_dim() if mind != "q_learning" else None,
    )


def summarize_records(records: list[dict[str, Any]], final_window: int = 100) -> dict[str, float]:
    """Summarize final-window Auction House metrics."""
    if not records:
        raise ValueError("cannot summarize empty Auction House records")
    window = records[-min(len(records), final_window) :]
    return {metric: finite_mean(float(record[metric]) for record in window) for metric in SUMMARY_METRICS}


def learned_bid_curve(agents: list[Agent], valuation_grid: tuple[float, ...]) -> list[dict[str, float]]:
    """Extract greedy tabular bid curves for reporting and sanity checks."""
    curves: list[dict[str, float]] = []
    for bidder_id, agent in enumerate(agents):
        if not isinstance(agent, QLearningMind):
            continue
        for value_index, value in enumerate(valuation_grid):
            action = agent.greedy_action((value_index,))
            curves.append(
                {
                    "bidder_id": float(bidder_id),
                    "valuation_bin": float(value_index),
                    "valuation": float(value),
                    "greedy_action": float(action),
                }
            )
    return curves
