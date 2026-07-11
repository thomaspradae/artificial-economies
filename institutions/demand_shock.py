from __future__ import annotations

from typing import Any

import numpy as np

from core.institution import Institution
from core.registry import register_institution


@register_institution("demand_shock")
class DemandShock(Institution):
    """Randomly scale market size by a lognormal demand multiplier."""

    name = "demand_shock"

    def __init__(self, shock_probability: float = 0.03, shock_scale: float = 0.30):
        self.shock_probability = shock_probability
        self.shock_scale = shock_scale

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        if out.get("phase") not in (None, "pre_demand"):
            return out
        rng = out.get("rng")
        if rng is None:
            rng = np.random.default_rng(out.get("seed", 0))
        market_size = float(out["market_size"])
        if rng.random() < self.shock_probability:
            market_size *= float(rng.lognormal(mean=0.0, sigma=self.shock_scale))
        out["market_size"] = market_size
        return out
