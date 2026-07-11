from __future__ import annotations

from typing import Any

from core.registry import register_mind
from minds.deep_rl.dqn_mind import DQNMind
from minds.deep_rl.ppo_mind import PPOMind


@register_mind("independent_dqn")
class IndependentDQNMind(DQNMind):
    """Explicit MARL baseline: each agent is its own DQN learner."""


class IndependentLearners:
    """Coordinator for N independent DQN/PPO learners with no shared state."""

    def __init__(self, n_agents: int, mind: str = "dqn", **mind_params: Any):
        if n_agents < 1:
            raise ValueError("n_agents must be positive")
        cls = DQNMind if mind == "dqn" else PPOMind if mind == "ppo" else None
        if cls is None:
            raise ValueError("mind must be 'dqn' or 'ppo'")
        seed = int(mind_params.pop("seed", 0))
        self.agents = [cls(seed=seed + index, **mind_params) for index in range(n_agents)]

    def act(self, observations: list[Any]) -> list[int]:
        if len(observations) != len(self.agents):
            raise ValueError("one observation per agent is required")
        return [int(agent.act(obs)) for agent, obs in zip(self.agents, observations)]

    def update(
        self,
        observations: list[Any],
        actions: list[int],
        rewards: list[float],
        next_observations: list[Any],
        dones: list[bool],
    ) -> None:
        for agent, obs, action, reward, next_obs, done in zip(
            self.agents,
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ):
            agent.update(obs, action, reward, next_obs, done)
