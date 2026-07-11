from __future__ import annotations

from core.agent import Agent
from core.registry import register_mind


@register_mind("heuristic_pricing")
class HeuristicPricingMind(Agent):
    """Simple pricing baseline: undercut last observed opponent price when possible."""

    def __init__(self, n_actions: int = 19, undercut: int = 1, fallback_action: int | None = None):
        if n_actions < 1:
            raise ValueError("n_actions must be positive")
        self.n_actions = n_actions
        self.undercut = max(0, undercut)
        self.fallback_action = n_actions // 2 if fallback_action is None else fallback_action

    def act(self, obs: tuple[int, int] | object) -> int:
        if not isinstance(obs, tuple) or len(obs) != 2:
            return int(min(max(self.fallback_action, 0), self.n_actions - 1))
        opponent_action = int(obs[1])
        return int(min(max(opponent_action - self.undercut, 0), self.n_actions - 1))

    def update(self, obs: object, action: object, reward: float, next_obs: object, done: bool) -> None:
        del obs, action, reward, next_obs, done

