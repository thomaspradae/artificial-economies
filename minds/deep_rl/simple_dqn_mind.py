from __future__ import annotations

from typing import Any

import numpy as np

from core.agent import Agent
from core.registry import register_mind
from minds.deep_rl.features import encode_observation
from minds.deep_rl.numpy_nn import MLPQNetwork, ReplayBuffer, Transition


@register_mind("simple_dqn")
class SimpleDQNMind(Agent):
    """Lightweight NumPy DQN baseline kept for fast smoke tests only."""

    def __init__(
        self,
        action_dim: int = 19,
        obs_dim: int | None = None,
        hidden_dim: int = 64,
        lr: float = 5e-4,
        gamma: float = 0.96,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.03,
        epsilon_decay: float = 0.9995,
        replay_size: int = 10_000,
        batch_size: int = 32,
        min_replay_size: int = 32,
        target_update_interval: int = 100,
        grad_clip: float = 5.0,
        seed: int = 0,
    ):
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.target_update_interval = target_update_interval
        self.grad_clip = grad_clip
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.replay = ReplayBuffer(replay_size, self.rng)
        self.q_net: MLPQNetwork | None = None
        self.target_net: MLPQNetwork | None = None
        self.update_count = 0
        self.last_loss = float("nan")

    def _features(self, obs: Any) -> np.ndarray:
        return encode_observation(obs, obs_dim=self.obs_dim, action_dim=self.action_dim)

    def _ensure_network(self, obs: Any) -> None:
        if self.q_net is not None:
            return
        features = self._features(obs)
        self.obs_dim = int(len(features))
        self.q_net = MLPQNetwork(self.obs_dim, self.action_dim, self.hidden_dim, self.rng)
        self.target_net = self.q_net.copy()

    def act(self, obs: Any, epsilon: float | None = None) -> int:
        self._ensure_network(obs)
        exploration = self.epsilon if epsilon is None else epsilon
        if self.rng.random() < exploration:
            return int(self.rng.integers(self.action_dim))
        return self.greedy_action(obs)

    def greedy_action(self, obs: Any) -> int:
        self._ensure_network(obs)
        assert self.q_net is not None
        return int(np.argmax(self.q_net.predict(self._features(obs))))

    def update(self, obs: Any, action: Any, reward: float, next_obs: Any, done: bool) -> None:
        self._ensure_network(obs)
        self._ensure_network(next_obs)
        assert self.q_net is not None and self.target_net is not None

        self.replay.append(
            Transition(
                obs=self._features(obs),
                action=int(action),
                reward=float(reward),
                next_obs=self._features(next_obs),
                done=bool(done),
            )
        )
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if len(self.replay) < max(self.min_replay_size, self.batch_size):
            return

        batch = self.replay.sample(self.batch_size)
        states = np.vstack([item.obs for item in batch])
        actions = np.array([item.action for item in batch], dtype=int)
        rewards = np.array([item.reward for item in batch], dtype=float)
        next_states = np.vstack([item.next_obs for item in batch])
        dones = np.array([item.done for item in batch], dtype=float)

        next_q, _ = self.target_net.forward(next_states)
        targets = rewards + (1.0 - dones) * self.gamma * np.max(next_q, axis=1)
        self.last_loss = self.q_net.train_q(states, actions, targets, self.lr, self.grad_clip)
        self.update_count += 1

        if self.update_count % self.target_update_interval == 0:
            self.target_net.load_from(self.q_net)

    def reset(self) -> None:
        """No episodic hidden state; learned parameters are intentionally kept."""
