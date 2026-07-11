from __future__ import annotations

from typing import Any

import numpy as np

from core.institution import Institution
from core.registry import register_institution


@register_institution("random_audit")
class RandomAudit(Institution):
    """Apply a fixed penalty with some probability when average price is high."""

    name = "random_audit"

    def __init__(self, audit_probability: float = 0.08, audit_threshold: float = 5.5, audit_penalty: float = 35.0):
        self.audit_probability = audit_probability
        self.audit_threshold = audit_threshold
        self.audit_penalty = audit_penalty

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        if out.get("phase") not in (None, "post_profit") or "rewards" not in out:
            return out
        prices = np.asarray(out["prices"], dtype=float)
        rng = out.get("rng")
        if rng is None:
            rng = np.random.default_rng(out.get("seed", 0))
        audit_hit = 0.0
        penalties = np.asarray(out.get("penalties", np.zeros_like(prices)), dtype=float).copy()
        rewards = np.asarray(out["rewards"], dtype=float).copy()
        if float(np.mean(prices)) >= self.audit_threshold and rng.random() < self.audit_probability:
            penalties[:] += self.audit_penalty
            rewards[:] -= self.audit_penalty
            audit_hit = 1.0
        out["penalties"] = penalties
        out["rewards"] = rewards
        out["audit_hit"] = audit_hit
        return out
