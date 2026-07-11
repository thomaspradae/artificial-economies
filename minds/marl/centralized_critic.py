from __future__ import annotations

from typing import Any

import numpy as np

from core.agent import Agent
from core.registry import register_mind
from minds.deep_rl.features import encode_observation
from minds.deep_rl.ppo_mind import PPOMind


class LinearCentralCritic:
    """Central value baseline over joint observations."""

    def __init__(self, input_dim: int, lr: float, gamma: float):
        self.input_dim = input_dim
        self.lr = lr
        self.gamma = gamma
        self.weights = np.zeros(input_dim, dtype=float)
        self.bias = 0.0

    def value(self, features: np.ndarray) -> float:
        return float(features @ self.weights + self.bias)

    def update(self, features: np.ndarray, reward: float, next_features: np.ndarray, done: bool) -> float:
        target = float(reward) + (0.0 if done else self.gamma * self.value(next_features))
        prediction = self.value(features)
        td_error = target - prediction
        self.weights += self.lr * td_error * features
        self.bias += self.lr * td_error
        return float(td_error)


@register_mind("centralized_critic_agent")
class CentralizedCriticMind(Agent):
    """Decentralized actor trained from a centralized critic signal."""

    def __init__(self, action_dim: int = 19, seed: int = 0, **actor_params: Any):
        self.actor = PPOMind(action_dim=action_dim, seed=seed, **actor_params)

    def act(self, obs: Any) -> int:
        return self.actor.act(obs)

    def greedy_action(self, obs: Any) -> int:
        return self.actor.greedy_action(obs)

    def update(self, obs: Any, action: Any, reward: float, next_obs: Any, done: bool) -> None:
        self.actor.update(obs, action, reward, next_obs, done)

    def update_with_advantage(self, obs: Any, action: Any, advantage: float, value_target: float | None = None) -> None:
        self.actor.update_with_advantage(obs, action, advantage, value_target)

    def reset(self) -> None:
        self.actor.reset()


class CentralizedCriticLearners:
    """MADDPG-style scaffold: decentralized actors, centralized value critic.

    The critic observes the concatenated joint observation during training.
    Inference remains decentralized because each `CentralizedCriticMind`
    acts only from its own observation.
    """

    def __init__(
        self,
        n_agents: int,
        action_dim: int = 19,
        obs_dim: int | None = None,
        critic_lr: float = 5e-4,
        gamma: float = 0.96,
        seed: int = 0,
        **actor_params: Any,
    ):
        if n_agents < 1:
            raise ValueError("n_agents must be positive")
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.agents = [
            CentralizedCriticMind(action_dim=action_dim, obs_dim=obs_dim, seed=seed + index, **actor_params)
            for index in range(n_agents)
        ]
        self.critic_lr = critic_lr
        self.critic: LinearCentralCritic | None = None
        self.last_td_error = float("nan")

    def _feature(self, observations: list[Any]) -> np.ndarray:
        parts = [
            encode_observation(obs, obs_dim=self.obs_dim, action_dim=self.action_dim)
            for obs in observations
        ]
        return np.concatenate(parts)

    def _ensure_critic(self, observations: list[Any]) -> None:
        if self.critic is not None:
            return
        features = self._feature(observations)
        self.critic = LinearCentralCritic(len(features), self.critic_lr, self.gamma)

    def update(
        self,
        observations: list[Any],
        actions: list[int],
        rewards: list[float],
        next_observations: list[Any],
        dones: list[bool],
    ) -> None:
        if len(observations) != self.n_agents:
            raise ValueError("one observation per centralized actor is required")
        self._ensure_critic(observations)
        assert self.critic is not None
        features = self._feature(observations)
        next_features = self._feature(next_observations)
        reward = float(np.mean(rewards))
        done = bool(all(dones))
        td_error = self.critic.update(features, reward, next_features, done)
        target_value = self.critic.value(features) + td_error
        self.last_td_error = td_error
        for agent, obs, action in zip(self.agents, observations, actions):
            agent.update_with_advantage(obs, action, td_error, target_value)
