from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.agent import Agent
from core.registry import build_experiment
import institutions  # noqa: F401 - registers institution classes
import minds  # noqa: F401 - registers mind classes
from minds.marl.centralized_critic import CentralizedCriticLearners
from minds.marl.independent_learners import IndependentDQNLearners
from minds.q_learning import QLearningMind
from worlds.resource_island.env import N_ACTIONS, ResourceIslandConfig, ResourceIslandWorld
from worlds.resource_island.features import (
    assert_finite_observations,
    resource_island_obs_dim,
    structured_observations,
)


SUPPORTED_RESOURCE_ISLAND_MINDS = (
    "q_learning",
    "dqn",
    "ppo",
    "independent_dqn",
    "centralized_critic",
)


@dataclass
class ResourceIslandRun:
    """Training result for a Resource Island smoke or replicated run."""

    records: list[dict[str, Any]]
    world: ResourceIslandWorld
    agents: list[Agent]
    final_epsilon: float
    mind: str = "q_learning"
    obs_dim: int | None = None


SUMMARY_METRICS = (
    "survival_rate",
    "reward_total",
    "welfare",
    "alive_count",
    "mean_energy",
    "food_inventory",
    "wood_inventory",
    "mean_pairwise_distance",
    "contact_rate",
    "gathered_food",
    "gathered_wood",
    "trade_count",
    "trade_attempt_count",
    "trade_blocked_count",
    "trade_inventory_blocked_count",
    "trade_institution_blocked_count",
    "property_claims",
    "property_violations",
    "property_opportunities",
    "property_resource_opportunities",
    "property_gather_opportunities",
    "specialization_index",
    "inequality_over_time",
    "resource_sustainability",
    "tax_revenue",
    "tax_revenue_cumulative",
)


def build_resource_island_q_world(
    seed: int,
    institution: str = "none",
    config: ResourceIslandConfig | None = None,
    institution_params: dict[str, Any] | None = None,
) -> ResourceIslandWorld:
    """Build Resource Island with QLearningMind over the world's tabular observation."""
    cfg = config if config is not None else ResourceIslandConfig()
    return build_experiment(
        {
            "world": "resource_island",
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
                        "state_shape": (N_ACTIONS, N_ACTIONS, N_ACTIONS),
                    },
                }
                for agent_id in range(cfg.n_agents)
            ],
        }
    )


def _mind_params(mind: str, seed: int, obs_dim: int) -> dict[str, Any]:
    if mind == "q_learning":
        return {
            "n_prices": N_ACTIONS,
            "seed": seed,
            "state_shape": (N_ACTIONS, N_ACTIONS, N_ACTIONS),
        }
    if mind == "dqn":
        return {
            "action_dim": N_ACTIONS,
            "obs_dim": obs_dim,
            "hidden_dim": 32,
            "batch_size": 16,
            "min_replay_size": 16,
            "target_update_interval": 50,
            "seed": seed,
        }
    if mind == "ppo":
        return {
            "action_dim": N_ACTIONS,
            "obs_dim": obs_dim,
            "hidden_dim": 32,
            "rollout_steps": 16,
            "batch_size": 16,
            "seed": seed,
        }
    raise ValueError(f"Unsupported Resource Island mind {mind!r}")


def build_resource_island_world(
    seed: int,
    institution: str = "none",
    config: ResourceIslandConfig | None = None,
    mind: str = "q_learning",
    obs_radius: int = 1,
    institution_params: dict[str, Any] | None = None,
) -> ResourceIslandWorld:
    """Build Resource Island with tabular or structured neural minds."""
    if mind == "q_learning":
        return build_resource_island_q_world(
            seed=seed,
            institution=institution,
            config=config,
            institution_params=institution_params,
        )
    if mind not in SUPPORTED_RESOURCE_ISLAND_MINDS:
        raise ValueError(f"Unsupported Resource Island mind {mind!r}")

    cfg = config if config is not None else ResourceIslandConfig()
    obs_dim = resource_island_obs_dim(obs_radius)
    if mind == "independent_dqn":
        joint_learner = IndependentDQNLearners(
            n_agents=cfg.n_agents,
            action_dim=N_ACTIONS,
            obs_dim=obs_dim,
            base_seed=seed,
            hidden_dim=32,
            batch_size=16,
            min_replay_size=16,
            target_update_interval=50,
        )
        world = build_experiment(
            {
                "world": "resource_island",
                "institution": institution,
                "institution_params": institution_params or {},
                "seed": seed,
                "world_params": {"config": cfg},
                "n_agents": 0,
            }
        )
        world.agents = joint_learner.agents
        world.joint_learner = joint_learner
        return world

    if mind == "centralized_critic":
        joint_learner = CentralizedCriticLearners(
            n_agents=cfg.n_agents,
            action_dim=N_ACTIONS,
            obs_dim=obs_dim,
            seed=seed,
            hidden_dim=32,
            rollout_steps=16,
            batch_size=16,
        )
        world = build_experiment(
            {
                "world": "resource_island",
                "institution": institution,
                "institution_params": institution_params or {},
                "seed": seed,
                "world_params": {"config": cfg},
                "n_agents": 0,
            }
        )
        world.agents = joint_learner.agents
        world.joint_learner = joint_learner
        return world

    return build_experiment(
        {
            "world": "resource_island",
            "institution": institution,
            "institution_params": institution_params or {},
            "seed": seed,
            "world_params": {"config": cfg},
            "agents": [
                {"mind": mind, "params": _mind_params(mind, seed + agent_id, obs_dim)}
                for agent_id in range(cfg.n_agents)
            ],
        }
    )


def _act(agent: Agent, obs: Any, epsilon: float) -> int:
    if isinstance(agent, QLearningMind):
        return int(agent.act(obs, epsilon=epsilon))
    return int(agent.act(obs))


def _observations(world: ResourceIslandWorld, mind: str, obs_radius: int) -> list[Any]:
    if mind == "q_learning":
        return world.observations()
    obs = structured_observations(world, radius=obs_radius)
    assert_finite_observations(obs)
    return obs


def _update_agents(
    world: ResourceIslandWorld,
    obs: list[Any],
    actions: list[int],
    rewards: np.ndarray,
    next_obs: list[Any],
    done: bool,
) -> None:
    joint_learner = getattr(world, "joint_learner", None)
    if joint_learner is not None:
        joint_learner.update(
            obs,
            actions,
            [float(reward) for reward in rewards],
            next_obs,
            [bool(done) for _ in world.agents],
        )
        return

    for agent_id, agent in enumerate(world.agents):
        agent.update(obs[agent_id], actions[agent_id], float(rewards[agent_id]), next_obs[agent_id], done)


def train_resource_island(
    steps: int = 200,
    seed: int = 0,
    institution: str = "none",
    institution_params: dict[str, Any] | None = None,
    config: ResourceIslandConfig | None = None,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    mind: str = "q_learning",
    obs_radius: int = 1,
) -> ResourceIslandRun:
    """Train tabular or structured neural minds in Resource Island."""
    if steps < 1:
        raise ValueError("steps must be positive")

    world = build_resource_island_world(
        seed=seed,
        institution=institution,
        institution_params=institution_params,
        config=config,
        mind=mind,
        obs_radius=obs_radius,
    )
    obs = _observations(world, mind, obs_radius)
    epsilon = float(epsilon_start)
    records: list[dict[str, Any]] = []

    for step in range(steps):
        actions = [_act(agent, obs[agent_id], epsilon) for agent_id, agent in enumerate(world.agents)]
        _, rewards, done, info = world.step(actions)
        next_obs = _observations(world, mind, obs_radius)
        _update_agents(world, obs, actions, rewards, next_obs, done)
        record = dict(info)
        record["train_step"] = float(step)
        record["epsilon"] = float(epsilon if mind == "q_learning" else np.nan)
        records.append(record)
        obs = next_obs
        epsilon = max(float(epsilon_min), epsilon * float(epsilon_decay))
        if done and step < steps - 1:
            world.reset()
            obs = _observations(world, mind, obs_radius)

    obs_dim = resource_island_obs_dim(obs_radius) if mind != "q_learning" else None
    return ResourceIslandRun(
        records=records,
        world=world,
        agents=list(world.agents),
        final_epsilon=epsilon,
        mind=mind,
        obs_dim=obs_dim,
    )


def finite_mean(values: list[float]) -> float:
    """Mean over finite values only."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan")
    return float(np.mean(array))


def summarize_records(records: list[dict[str, Any]], final_window: int = 50) -> dict[str, float]:
    """Summarize final-window Resource Island metrics."""
    if not records:
        raise ValueError("cannot summarize empty Resource Island records")
    window = records[-min(len(records), final_window) :]
    return {metric: finite_mean([float(record[metric]) for record in window]) for metric in SUMMARY_METRICS}
