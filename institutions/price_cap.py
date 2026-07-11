from __future__ import annotations

from typing import Any

import numpy as np

from core.institution import Institution
from core.registry import register_institution


@register_institution("price_cap")
class PriceCap(Institution):
    """Clamp realized prices to a maximum cap before demand is evaluated."""

    name = "price_cap"

    def __init__(self, price_cap: float = 5.5):
        self.price_cap = price_cap

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        if out.get("phase") not in (None, "pre_demand"):
            return out
        out["prices"] = np.minimum(np.asarray(out["prices"], dtype=float), self.price_cap)
        return out
