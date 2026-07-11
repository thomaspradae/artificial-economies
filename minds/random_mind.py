from __future__ import annotations

import numpy as np

from core.agent import Agent
from core.registry import register_mind


@register_mind("random")
class RandomMind(Agent):
    """Uniform random discrete-action baseline with no learning."""

    def __init__(self, n_actions: int = 19, seed: int = 0):
        if n_actions < 1:
            raise ValueError("n_actions must be positive")
        self.n_actions = n_actions
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def act(self, obs: object) -> int:
        del obs
        return int(self.rng.integers(self.n_actions))

    def update(self, obs: object, action: object, reward: float, next_obs: object, done: bool) -> None:
        del obs, action, reward, next_obs, done

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

