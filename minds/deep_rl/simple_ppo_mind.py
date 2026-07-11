from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.agent import Agent
from core.registry import register_mind
from minds.deep_rl.features import encode_observation
from minds.deep_rl.numpy_nn import ActorCriticNetwork


@dataclass
class SimpleRolloutItem:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    old_prob: float
    value: float
    next_value: float


@register_mind("simple_ppo")
class SimplePPOMind(Agent):
    """Lightweight NumPy PPO baseline kept for fast smoke tests only."""

    def __init__(
        self,
        action_dim: int = 19,
        obs_dim: int | None = None,
        hidden_dim: int = 64,
        gamma: float = 0.96,
        gae_lambda: float = 0.95,
        rollout_steps: int = 32,
        epochs: int = 3,
        batch_size: int = 32,
        policy_lr: float = 3e-4,
        value_lr: float = 1e-3,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        grad_clip: float = 5.0,
        seed: int = 0,
    ):
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rollout_steps = rollout_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.net: ActorCriticNetwork | None = None
        self.rollout: list[SimpleRolloutItem] = []
        self._last_action: int | None = None
        self._last_prob: float | None = None
        self._last_value: float | None = None
        self.last_update: dict[str, float] = {}

    def _features(self, obs: Any) -> np.ndarray:
        return encode_observation(obs, obs_dim=self.obs_dim, action_dim=self.action_dim)

    def _ensure_network(self, obs: Any) -> None:
        if self.net is not None:
            return
        features = self._features(obs)
        self.obs_dim = int(len(features))
        self.net = ActorCriticNetwork(self.obs_dim, self.action_dim, self.hidden_dim, self.rng)

    def act(self, obs: Any) -> int:
        self._ensure_network(obs)
        assert self.net is not None
        features = self._features(obs)
        probs, value = self.net.policy_value(features)
        action = int(self.rng.choice(self.action_dim, p=probs))
        self._last_action = action
        self._last_prob = float(probs[action])
        self._last_value = value
        return action

    def greedy_action(self, obs: Any) -> int:
        self._ensure_network(obs)
        assert self.net is not None
        probs, _ = self.net.policy_value(self._features(obs))
        return int(np.argmax(probs))

    def update(self, obs: Any, action: Any, reward: float, next_obs: Any, done: bool) -> None:
        self._ensure_network(obs)
        self._ensure_network(next_obs)
        assert self.net is not None

        features = self._features(obs)
        next_features = self._features(next_obs)
        probs, value = self.net.policy_value(features)
        _, next_value = self.net.policy_value(next_features)
        old_prob = self._last_prob if self._last_action == int(action) and self._last_prob is not None else float(probs[int(action)])
        old_value = self._last_value if self._last_action == int(action) and self._last_value is not None else value

        self.rollout.append(
            SimpleRolloutItem(features, int(action), float(reward), next_features, bool(done), old_prob, old_value, next_value)
        )
        if len(self.rollout) >= self.rollout_steps or done:
            self._train_rollout()

    def update_with_advantage(self, obs: Any, action: Any, advantage: float, value_target: float | None = None) -> None:
        self._ensure_network(obs)
        assert self.net is not None
        features = self._features(obs)
        probs, value = self.net.policy_value(features)
        target = value + float(advantage) if value_target is None else float(value_target)
        self.last_update = self.net.train_actor_critic(
            states=features[None, :],
            actions=np.array([int(action)], dtype=int),
            advantages=np.array([float(advantage)], dtype=float),
            returns=np.array([target], dtype=float),
            old_action_probs=np.array([float(probs[int(action)])], dtype=float),
            clip_ratio=self.clip_ratio,
            policy_lr=self.policy_lr,
            value_lr=self.value_lr,
            entropy_coef=self.entropy_coef,
            grad_clip=self.grad_clip,
        )

    def _train_rollout(self) -> None:
        if not self.rollout:
            return
        rewards = np.array([item.reward for item in self.rollout], dtype=float)
        values = np.array([item.value for item in self.rollout], dtype=float)
        next_values = np.array([item.next_value for item in self.rollout], dtype=float)
        dones = np.array([item.done for item in self.rollout], dtype=float)
        deltas = rewards + (1.0 - dones) * self.gamma * next_values - values

        advantages = np.zeros_like(deltas)
        gae = 0.0
        for index in range(len(deltas) - 1, -1, -1):
            gae = deltas[index] + (1.0 - dones[index]) * self.gamma * self.gae_lambda * gae
            advantages[index] = gae
        returns = advantages + values
        if len(advantages) > 1 and np.std(advantages) > 1e-8:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        states = np.vstack([item.obs for item in self.rollout])
        actions = np.array([item.action for item in self.rollout], dtype=int)
        old_probs = np.array([item.old_prob for item in self.rollout], dtype=float)

        n = len(states)
        effective_batch = min(self.batch_size, n)
        for _ in range(self.epochs):
            indices = self.rng.permutation(n)
            for start in range(0, n, effective_batch):
                batch = indices[start : start + effective_batch]
                self.last_update = self.net.train_actor_critic(
                    states=states[batch],
                    actions=actions[batch],
                    advantages=advantages[batch],
                    returns=returns[batch],
                    old_action_probs=old_probs[batch],
                    clip_ratio=self.clip_ratio,
                    policy_lr=self.policy_lr,
                    value_lr=self.value_lr,
                    entropy_coef=self.entropy_coef,
                    grad_clip=self.grad_clip,
                )
        self.rollout = []

    def reset(self) -> None:
        self._last_action = None
        self._last_prob = None
        self._last_value = None
