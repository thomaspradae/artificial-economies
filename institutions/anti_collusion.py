from __future__ import annotations

from typing import Any

import numpy as np

from core.institution import Institution
from core.registry import register_institution


@register_institution("anti_collusion")
class AntiCollusion(Institution):
    """Apply a fixed penalty when both prices are high and close together."""

    name = "anti_collusion"

    def __init__(self, collusion_threshold: float = 5.5, collusion_window: float = 0.75, collusion_penalty: float = 20.0):
        self.collusion_threshold = collusion_threshold
        self.collusion_window = collusion_window
        self.collusion_penalty = collusion_penalty

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        if out.get("phase") not in (None, "post_profit") or "rewards" not in out:
            return out
        prices = np.asarray(out["prices"], dtype=float)
        penalties = np.asarray(out.get("penalties", np.zeros_like(prices)), dtype=float).copy()
        rewards = np.asarray(out["rewards"], dtype=float).copy()
        both_high = float(np.mean(prices)) >= self.collusion_threshold
        close_prices = abs(float(prices[0] - prices[1])) <= self.collusion_window
        if both_high and close_prices:
            penalties[:] += self.collusion_penalty
            rewards[:] -= self.collusion_penalty
        out["penalties"] = penalties
        out["rewards"] = rewards
        return out
