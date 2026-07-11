from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.agent import Agent
from core.institution import Institution


class World(ABC):
    """Base class for an environment where Agents interact under an Institution."""

    def __init__(self, agents: list[Agent] | None = None, institution: Institution | None = None, seed: int | None = None):
        self.agents = agents or []
        self.institution = institution
        self.seed = seed

    @abstractmethod
    def reset(self) -> object:
        """Reset world state and return initial observations."""

    @abstractmethod
    def step(self, actions: list[Any]) -> tuple[Any, Any, bool, dict[str, Any]]:
        """Advance one timestep and return next_obs, rewards, done, info."""

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Return current cumulative metrics for this run."""

    def render_state(self) -> object:
        """Return a serializable visualization snapshot when implemented."""
        raise NotImplementedError

