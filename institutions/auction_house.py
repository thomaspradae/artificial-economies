from __future__ import annotations

from typing import Any

import numpy as np

from core.institution import Institution
from core.registry import register_institution


@register_institution("auction_information_policy")
class AuctionInformationPolicy(Institution):
    """Modify auction bidders' private value observations.

    The policy preserves Auction House's one-dimensional tabular observation
    shape. A public signal mixes the bidder's own value with the average rival
    value before discretization; optional bin noise then perturbs the observed
    value bin. This is an information/noise variant, not a payment rule.
    """

    name = "auction_information_policy"

    def __init__(
        self,
        public_signal_weight: float = 0.0,
        noise_bins: int = 0,
        seed: int = 0,
    ):
        if public_signal_weight < 0.0:
            raise ValueError("public_signal_weight must be non-negative")
        if noise_bins < 0:
            raise ValueError("noise_bins must be non-negative")
        self.public_signal_weight = float(public_signal_weight)
        self.noise_bins = int(noise_bins)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        if out.get("phase") != "auction_observation":
            return out

        grid = np.asarray(out["valuation_grid"], dtype=float)
        observed_bin = int(out["valuation_bin"])

        if self.public_signal_weight > 0.0:
            valuations = np.asarray(out["valuations"], dtype=float)
            agent_id = int(out["agent_id"])
            rivals = np.delete(valuations, agent_id)
            rival_signal = float(np.mean(rivals)) if len(rivals) else float(valuations[agent_id])
            mixed_value = (
                float(valuations[agent_id]) + self.public_signal_weight * rival_signal
            ) / (1.0 + self.public_signal_weight)
            observed_bin = int(np.argmin(np.abs(grid - mixed_value)))
            out["public_signal_value"] = mixed_value
            out["public_signal_applied"] = 1.0
        else:
            out["public_signal_applied"] = 0.0

        if self.noise_bins > 0:
            shift = int(self.rng.integers(-self.noise_bins, self.noise_bins + 1))
            observed_bin += shift
            out["information_noise_bins"] = float(abs(shift))
        else:
            out["information_noise_bins"] = 0.0

        out["valuation_bin"] = int(np.clip(observed_bin, 0, len(grid) - 1))
        return out
