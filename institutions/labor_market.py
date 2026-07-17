from __future__ import annotations

from typing import Any

import numpy as np

from core.institution import Institution
from core.registry import register_institution
from worlds.labor_market.benchmarks import deferred_acceptance


@register_institution("deferred_acceptance")
class DeferredAcceptanceInstitution(Institution):
    """Worker-proposing deferred-acceptance matching institution."""

    name = "deferred_acceptance"

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        if out.get("phase") != "labor_market_match":
            return out
        worker_preferences = np.asarray(out["reported_worker_preferences"], dtype=int)
        employer_preferences = np.asarray(out["employer_preferences"], dtype=int)
        matches = deferred_acceptance(worker_preferences, employer_preferences)
        out["matches"] = matches
        return out
