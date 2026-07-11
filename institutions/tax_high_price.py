from __future__ import annotations

from typing import Any

import numpy as np

from core.institution import Institution
from core.registry import register_institution


@register_institution("tax_high_price")
class TaxHighPrice(Institution):
    """Subtract a quantity-weighted tax from rewards above a price threshold."""

    name = "tax_high_price"

    def __init__(self, tax_threshold: float = 5.5, tax_rate: float = 0.30):
        self.tax_threshold = tax_threshold
        self.tax_rate = tax_rate

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        if out.get("phase") not in (None, "post_profit") or "quantities" not in out:
            return out
        prices = np.asarray(out["prices"], dtype=float)
        quantities = np.asarray(out["quantities"], dtype=float)
        penalties = np.asarray(out.get("penalties", np.zeros_like(prices)), dtype=float).copy()
        rewards = np.asarray(out["rewards"], dtype=float).copy()
        excess = np.maximum(prices - self.tax_threshold, 0.0)
        tax = self.tax_rate * excess * quantities
        penalties += tax
        rewards -= tax
        out["penalties"] = penalties
        out["rewards"] = rewards
        return out
