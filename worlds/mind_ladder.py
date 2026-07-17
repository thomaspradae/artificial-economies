from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.agent import Agent
from minds.deep_rl.torch_dqn_mind import TorchDQNMind
from minds.deep_rl.torch_ppo_mind import TorchPPOMind
from minds.marl.centralized_critic import CentralizedCriticLearners
from minds.marl.independent_learners import IndependentDQNLearners
from minds.q_learning import QLearningMind


NEURAL_LADDER_MINDS = ("dqn", "ppo", "independent_dqn", "centralized_critic")
SUPPORTED_LADDER_MINDS = ("q_learning", *NEURAL_LADDER_MINDS)


@dataclass
class LadderAgents:
    agents: list[Agent]
    joint_learner: Any | None = None
    obs_dim: int | None = None


def build_neural_ladder_agents(
    mind: str,
    *,
    n_agents: int,
    action_dim: int,
    obs_dim: int,
    seed: int,
    hidden_dim: int = 32,
    batch_size: int = 16,
    min_replay_size: int = 16,
    rollout_steps: int = 16,
    target_update_interval: int = 50,
) -> LadderAgents:
    """Build neural/MARL agents for any fixed-width discrete-action world."""
    if n_agents < 1:
        raise ValueError("n_agents must be positive")
    if mind == "dqn":
        return LadderAgents(
            agents=[
                TorchDQNMind(
                    action_dim=action_dim,
                    obs_dim=obs_dim,
                    hidden_dim=hidden_dim,
                    batch_size=batch_size,
                    min_replay_size=min_replay_size,
                    target_update_interval=target_update_interval,
                    seed=seed + agent_id,
                )
                for agent_id in range(n_agents)
            ],
            obs_dim=obs_dim,
        )
    if mind == "ppo":
        return LadderAgents(
            agents=[
                TorchPPOMind(
                    action_dim=action_dim,
                    obs_dim=obs_dim,
                    hidden_dim=hidden_dim,
                    rollout_steps=rollout_steps,
                    batch_size=batch_size,
                    seed=seed + agent_id,
                )
                for agent_id in range(n_agents)
            ],
            obs_dim=obs_dim,
        )
    if mind == "independent_dqn":
        learners = IndependentDQNLearners(
            n_agents=n_agents,
            action_dim=action_dim,
            obs_dim=obs_dim,
            base_seed=seed,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
            target_update_interval=target_update_interval,
        )
        return LadderAgents(agents=learners.agents, joint_learner=learners, obs_dim=obs_dim)
    if mind == "centralized_critic":
        learners = CentralizedCriticLearners(
            n_agents=n_agents,
            action_dim=action_dim,
            obs_dim=obs_dim,
            seed=seed,
            hidden_dim=hidden_dim,
            rollout_steps=rollout_steps,
            batch_size=batch_size,
        )
        return LadderAgents(agents=learners.agents, joint_learner=learners, obs_dim=obs_dim)
    raise ValueError(f"Unsupported neural ladder mind {mind!r}")


def act_ladder_agent(agent: Agent, obs: Any, epsilon: float) -> int:
    """Act with tabular epsilon when available; neural minds own exploration."""
    if isinstance(agent, QLearningMind):
        return int(agent.act(obs, epsilon=epsilon))
    return int(agent.act(obs))


def update_ladder_agents(
    agents: list[Agent],
    joint_learner: Any | None,
    observations: list[Any],
    actions: list[int],
    rewards: np.ndarray | list[float],
    next_observations: list[Any],
    done: bool,
) -> None:
    """Update either independent agents or a joint learner coordinator."""
    reward_list = [float(reward) for reward in rewards]
    if joint_learner is not None:
        joint_learner.update(
            observations,
            [int(action) for action in actions],
            reward_list,
            next_observations,
            [bool(done) for _ in agents],
        )
        return
    for agent, obs, action, reward, next_obs in zip(agents, observations, actions, reward_list, next_observations):
        agent.update(obs, int(action), float(reward), next_obs, bool(done))


def assert_finite_feature_matrix(observations: list[Any]) -> None:
    """Fail early if a world feature encoder emits non-finite values."""
    for index, obs in enumerate(observations):
        array = np.asarray(obs, dtype=float)
        if array.ndim != 1:
            raise ValueError(f"observation {index} is not a 1D feature vector")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"observation {index} contains non-finite values")
