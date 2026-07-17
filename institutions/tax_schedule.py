from __future__ import annotations

from typing import Any

import numpy as np

from core.institution import Institution
from core.registry import register_institution


@register_institution("tax_schedule")
class TaxSchedule(Institution):
    """Generic progressive tax-and-redistribution institution.

    Worlds can call this at phase ``post_rewards`` or any phase that supplies a
    ``rewards`` vector. Taxes are computed only on positive rewards and are
    redistributed equally over eligible agents.
    """

    name = "tax_schedule"

    def __init__(
        self,
        brackets: tuple[tuple[float, float], ...] = ((0.0, 0.1), (1.0, 0.25), (3.0, 0.4)),
        redistribute: bool = True,
    ):
        if not brackets:
            raise ValueError("brackets must not be empty")
        normalized = tuple((float(threshold), float(rate)) for threshold, rate in brackets)
        if any(rate < 0.0 or rate > 1.0 for _, rate in normalized):
            raise ValueError("tax rates must be in [0, 1]")
        self.brackets = tuple(sorted(normalized, key=lambda item: item[0]))
        self.redistribute = bool(redistribute)
        self.total_revenue = 0.0

    def reset(self) -> None:
        self.total_revenue = 0.0

    def _tax_one(self, value: float) -> float:
        income = max(float(value), 0.0)
        tax = 0.0
        for index, (threshold, rate) in enumerate(self.brackets):
            next_threshold = self.brackets[index + 1][0] if index + 1 < len(self.brackets) else None
            if income <= threshold:
                continue
            upper = income if next_threshold is None else min(income, next_threshold)
            if upper > threshold:
                tax += (upper - threshold) * rate
        return float(min(tax, income))

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        if "rewards" not in out or out.get("phase") not in {"post_rewards", "public_goods_post_rewards"}:
            return out
        rewards = np.asarray(out["rewards"], dtype=float).copy()
        taxes = np.asarray([self._tax_one(value) for value in rewards], dtype=float)
        rewards -= taxes
        revenue = float(np.sum(taxes))
        self.total_revenue += revenue
        if self.redistribute and len(rewards) > 0:
            alive = np.asarray(out.get("alive", np.ones(len(rewards), dtype=bool)), dtype=bool)
            count = int(np.sum(alive))
            if count > 0:
                rewards[alive] += revenue / count
        out["rewards"] = rewards
        out["tax_revenue"] = revenue
        out["tax_revenue_cumulative"] = self.total_revenue
        return out
