from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from arena_v0 import MECHANISMS, MarketConfig
from core.registry import build_experiment
import institutions  # noqa: F401 - registers institutions
import minds  # noqa: F401 - registers minds
from minds.marl.centralized_critic import CentralizedCriticLearners
from minds.marl.independent_learners import IndependentDQNLearners
from minds.q_learning import QLearningMind
from worlds.pricing_arena.benchmarks import compute_static_benchmarks
from worlds.pricing_arena.env import PricingArenaWorld


SUPPORTED_MINDS = (
    "q_learning",
    "random",
    "heuristic_pricing",
    "dqn",
    "ppo",
    "independent_dqn",
    "centralized_critic",
    "simple_dqn",
    "simple_ppo",
)


@dataclass
class TrainingRun:
    data: dict[str, np.ndarray]
    benchmarks: dict[str, object]
    agent1: Any
    agent2: Any
    config: MarketConfig
    final_state: tuple[int, int]
    final_epsilon: float


def _institution_params(config: MarketConfig) -> dict[str, float]:
    if config.mechanism == "price_cap":
        return {"price_cap": config.price_cap}
    if config.mechanism == "tax_high_price":
        return {"tax_threshold": config.tax_threshold, "tax_rate": config.tax_rate}
    if config.mechanism == "random_audit":
        return {
            "audit_probability": config.audit_probability,
            "audit_threshold": config.audit_threshold,
            "audit_penalty": config.audit_penalty,
        }
    if config.mechanism == "anti_collusion":
        return {
            "collusion_threshold": config.collusion_threshold,
            "collusion_window": config.collusion_window,
            "collusion_penalty": config.collusion_penalty,
        }
    if config.mechanism == "demand_shock":
        return {"shock_probability": config.shock_probability, "shock_scale": config.shock_scale}
    return {}


def _build_pricing_world(
    mechanism: str,
    seed: int,
    agents: list[Any] | None = None,
    mind: str = "q_learning",
) -> PricingArenaWorld:
    config = MarketConfig(mechanism=mechanism)
    if agents is None:
        if mind == "independent_dqn":
            joint_learner = IndependentDQNLearners(
                n_agents=2,
                action_dim=19,
                obs_dim=38,
                base_seed=seed,
                hidden_dim=32,
                batch_size=16,
                min_replay_size=16,
                target_update_interval=50,
            )
            world = build_experiment(
                {
                    "world": "pricing_arena",
                    "institution": mechanism,
                    "institution_params": _institution_params(config),
                    "seed": seed,
                    "world_params": {"config": config},
                    "n_agents": 0,
                }
            )
            world.agents = joint_learner.agents
            world.joint_learner = joint_learner
            return world

        if mind == "centralized_critic":
            joint_learner = CentralizedCriticLearners(
                n_agents=2,
                action_dim=19,
                obs_dim=38,
                seed=seed,
                hidden_dim=32,
                rollout_steps=16,
                batch_size=16,
            )
            world = build_experiment(
                {
                    "world": "pricing_arena",
                    "institution": mechanism,
                    "institution_params": _institution_params(config),
                    "seed": seed,
                    "world_params": {"config": config},
                    "n_agents": 0,
                }
            )
            world.agents = joint_learner.agents
            world.joint_learner = joint_learner
            return world

        agents_config = [
            {"mind": mind, "params": _mind_params(mind, seed)},
            {"mind": mind, "params": _mind_params(mind, seed + 1)},
        ]
        return build_experiment(
            {
                "world": "pricing_arena",
                "institution": mechanism,
                "institution_params": _institution_params(config),
                "agents": agents_config,
                "seed": seed,
                "world_params": {"config": config},
            }
        )

    institution_world = build_experiment(
        {
            "world": "pricing_arena",
            "institution": mechanism,
            "institution_params": _institution_params(config),
            "seed": seed,
            "world_params": {"config": config},
            "n_agents": 0,
        }
    )
    institution_world.agents = agents
    return institution_world


def _mind_params(mind: str, seed: int) -> dict[str, int]:
    if mind == "q_learning":
        return {"n_prices": 19, "seed": seed}
    if mind in {"random", "heuristic_pricing"}:
        return {"n_actions": 19, "seed": seed} if mind == "random" else {"n_actions": 19}
    if mind in {"dqn", "simple_dqn"}:
        return {
            "action_dim": 19,
            "obs_dim": 38,
            "hidden_dim": 32,
            "batch_size": 16,
            "min_replay_size": 16,
            "target_update_interval": 50,
            "seed": seed,
        }
    if mind in {"ppo", "simple_ppo"}:
        return {
            "action_dim": 19,
            "obs_dim": 38,
            "hidden_dim": 32,
            "rollout_steps": 16,
            "batch_size": 16,
            "seed": seed,
        }
    raise ValueError(f"Unsupported pricing mind {mind!r}")


def _policy_action(
    agent: Any,
    state: tuple[int, int],
    epsilon: float | None = None,
    greedy: bool = False,
) -> int:
    if epsilon is not None and isinstance(agent, QLearningMind):
        return int(agent.act(state, epsilon=epsilon))
    if greedy and hasattr(agent, "greedy_action"):
        return int(agent.greedy_action(state))
    return int(agent.act(state))


def _maybe_update(agent: Any, state: tuple[int, int], action: int, reward: float, next_state: tuple[int, int]) -> None:
    if isinstance(agent, QLearningMind):
        agent.update(state, action, reward, next_state, done=False)
    else:
        agent.update(state, action, reward, next_state, False)


def _update_world_agents(
    world: PricingArenaWorld,
    state: tuple[int, int],
    actions: list[int],
    rewards: np.ndarray,
    next_state: tuple[int, int],
) -> None:
    joint_learner = getattr(world, "joint_learner", None)
    if joint_learner is not None:
        joint_learner.update(
            [state for _ in world.agents],
            actions,
            [float(reward) for reward in rewards],
            [next_state for _ in world.agents],
            [False for _ in world.agents],
        )
        return

    for agent, action, reward in zip(world.agents, actions, rewards):
        _maybe_update(agent, state, int(action), float(reward), next_state)


def _add_learning_metrics(info: dict[str, float], step: int, epsilon: float) -> None:
    info["step"] = float(step)
    info["epsilon"] = float(epsilon)


def records_to_arrays(records: list[dict[str, float]]) -> dict[str, np.ndarray]:
    if not records:
        raise ValueError("cannot convert empty records to arrays")
    return {key: np.array([record[key] for record in records]) for key in records[0].keys()}


def train_market_with_agents(
    mechanism: str = "none",
    steps: int = 40_000,
    seed: int = 7,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.03,
    epsilon_decay: float = 0.99985,
    mind: str = "q_learning",
) -> TrainingRun:
    if steps < 1:
        raise ValueError("steps must be positive")

    world = _build_pricing_world(mechanism=mechanism, seed=seed, mind=mind)
    config = world.config
    benchmarks = compute_static_benchmarks(config.price_grid)
    state = world.state
    epsilon = epsilon_start
    records: list[dict[str, float]] = []

    for t in range(steps):
        a1 = _policy_action(world.agents[0], state, epsilon if mind == "q_learning" else None)
        a2 = _policy_action(world.agents[1], state, epsilon if mind == "q_learning" else None)
        next_state, rewards, _, info = world.step([a1, a2])

        _update_world_agents(world, state, [a1, a2], rewards, next_state)

        state = next_state
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        _add_learning_metrics(info, t, epsilon)
        records.append(info)

    return TrainingRun(
        data=records_to_arrays(records),
        benchmarks=benchmarks,
        agent1=world.agents[0],
        agent2=world.agents[1],
        config=config,
        final_state=state,
        final_epsilon=epsilon,
    )


def train_market(
    mechanism: str = "none",
    steps: int = 40_000,
    seed: int = 7,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.03,
    epsilon_decay: float = 0.99985,
    mind: str = "q_learning",
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    run = train_market_with_agents(
        mechanism=mechanism,
        steps=steps,
        seed=seed,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        mind=mind,
    )
    return run.data, run.benchmarks


def evaluate_policy_pair(
    agent1: Any,
    agent2: Any,
    mechanism: str = "none",
    steps: int = 5_000,
    seed: int = 10_000,
    initial_state: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    if steps < 1:
        raise ValueError("steps must be positive")

    world = _build_pricing_world(mechanism=mechanism, seed=seed, agents=[agent1, agent2])
    if initial_state is not None:
        world.state = initial_state
    agent1.reset()
    agent2.reset()
    state = world.state
    records: list[dict[str, float]] = []

    for t in range(steps):
        a1 = _policy_action(agent1, state, greedy=True)
        a2 = _policy_action(agent2, state, greedy=True)
        next_state, _, _, info = world.step([a1, a2])
        state = next_state
        _add_learning_metrics(info, t, 0.0)
        records.append(info)

    return records_to_arrays(records)


def train_adversary_against_frozen_firm1(
    frozen_agent1: Any,
    mechanism: str = "none",
    steps: int = 20_000,
    seed: int = 20_000,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.03,
    epsilon_decay: float = 0.99985,
    initial_state: tuple[int, int] | None = None,
) -> TrainingRun:
    if steps < 1:
        raise ValueError("steps must be positive")

    adversary = QLearningMind(n_prices=19, seed=seed)
    world = _build_pricing_world(mechanism=mechanism, seed=seed, agents=[frozen_agent1, adversary])
    if initial_state is not None:
        world.state = initial_state
    frozen_agent1.reset()
    state = world.state
    epsilon = epsilon_start
    records: list[dict[str, float]] = []

    for t in range(steps):
        a1 = _policy_action(frozen_agent1, state, greedy=True)
        a2 = adversary.act(state, epsilon=epsilon)
        next_state, rewards, _, info = world.step([a1, a2])
        adversary.update(state, a2, float(rewards[1]), next_state, done=False)

        state = next_state
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        _add_learning_metrics(info, t, epsilon)
        records.append(info)

    return TrainingRun(
        data=records_to_arrays(records),
        benchmarks=world.benchmarks,
        agent1=frozen_agent1,
        agent2=adversary,
        config=world.config,
        final_state=state,
        final_epsilon=epsilon,
    )
