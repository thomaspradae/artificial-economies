from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    """Base class for any decision-making entity in a World."""

    @abstractmethod
    def act(self, obs: Any) -> object:
        """Given an observation, return an action."""

    @abstractmethod
    def update(self, obs: Any, action: Any, reward: float, next_obs: Any, done: bool) -> None:
        """Learning update after a step. No-op for non-learning agents."""

    def reset(self) -> None:
        """Reset internal state between episodes. Default no-op."""

