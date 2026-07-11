from __future__ import annotations

from typing import Any

import numpy as np

from arena_v0 import QAgent
from core.agent import Agent
from core.registry import register_mind


@register_mind("q_learning")
class QLearningMind(QAgent, Agent):
    """Tabular Q-learning mind preserving the existing `QAgent` update rule."""

    def __init__(
        self,
        n_prices: int = 19,
        lr: float = 0.08,
        gamma: float = 0.96,
        seed: int = 0,
        state_shape: tuple[int, ...] | None = None,
    ):
        super().__init__(n_prices=n_prices, lr=lr, gamma=gamma, seed=seed)
        self.state_shape = tuple(int(value) for value in state_shape) if state_shape is not None else (n_prices, n_prices)
        if len(self.state_shape) < 1:
            raise ValueError("state_shape must contain at least one discrete dimension")
        if any(value <= 0 for value in self.state_shape):
            raise ValueError("state_shape dimensions must be positive")
        if self.state_shape != (n_prices, n_prices):
            self.q_values = np.zeros((*self.state_shape, n_prices))
        self.default_epsilon = 0.0

    def _state_index(self, obs: tuple[int, ...]) -> tuple[int, ...]:
        state = tuple(int(value) for value in obs)
        if len(state) != len(self.state_shape):
            raise ValueError(f"expected observation with {len(self.state_shape)} dimensions, got {len(state)}")
        for value, upper in zip(state, self.state_shape):
            if value < 0 or value >= upper:
                raise ValueError(f"observation index {value} outside [0, {upper})")
        return state

    def greedy_action(self, state: tuple[int, ...]) -> int:
        return int(np.argmax(self.q_values[self._state_index(state)]))

    def act(self, obs: tuple[int, ...], epsilon: float | None = None) -> int:
        exploration = self.default_epsilon if epsilon is None else epsilon
        if self.rng.random() < exploration:
            return int(self.rng.integers(self.n_prices))
        return self.greedy_action(obs)

    def update(
        self,
        obs: tuple[int, ...],
        action: int,
        reward: float,
        next_obs: tuple[int, ...],
        done: bool = False,
    ) -> None:
        del done
        state = self._state_index(obs)
        next_state = self._state_index(next_obs)
        old_value = self.q_values[state + (int(action),)]
        future_value = float(np.max(self.q_values[next_state]))
        target = reward + self.gamma * future_value
        self.q_values[state + (int(action),)] += self.lr * (target - old_value)
