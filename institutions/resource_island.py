from __future__ import annotations

from typing import Any

import numpy as np

from core.institution import Institution
from core.registry import register_institution


@register_institution("property_rights")
class PropertyRights(Institution):
    """Claim-based gathering rights over Resource Island cells."""

    name = "property_rights"

    def __init__(self, violation_penalty: float = 1.0):
        self.violation_penalty = float(violation_penalty)
        self.claims: dict[tuple[int, int], int] = {}

    def reset(self) -> None:
        self.claims = {}

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        phase = out.get("phase")
        if phase == "pre_gather":
            position = tuple(out["position"])
            owner = self.claims.get(position)
            if owner is not None and owner != int(out["agent_id"]):
                out["allowed"] = False
                out["penalty"] = float(out.get("penalty", 0.0)) + self.violation_penalty
                out["property_violation"] = 1
        elif phase == "post_gather" and int(out.get("gathered_amount", 0)) > 0:
            position = tuple(out["position"])
            if position not in self.claims:
                self.claims[position] = int(out["agent_id"])
                out["property_claim_created"] = 1
        return out


@register_institution("redistribution")
class Redistribution(Institution):
    """Tax positive step rewards and redistribute the pool over alive agents."""

    name = "redistribution"

    def __init__(self, tax_rate: float = 0.2):
        if not 0.0 <= tax_rate <= 1.0:
            raise ValueError("tax_rate must be in [0, 1]")
        self.tax_rate = float(tax_rate)

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        if out.get("phase") != "post_rewards":
            return out
        rewards = np.asarray(out["rewards"], dtype=float).copy()
        alive = np.asarray(out["alive"], dtype=bool)
        positive = np.maximum(rewards, 0.0)
        taxes = self.tax_rate * positive
        pool = float(np.sum(taxes))
        rewards -= taxes
        alive_count = int(np.sum(alive))
        if alive_count > 0:
            rewards[alive] += pool / alive_count
        out["rewards"] = rewards
        out["redistribution_pool"] = pool
        return out


@register_institution("trade_price_controls")
class TradePriceControls(Institution):
    """Block Resource Island trades above a maximum exchange ratio."""

    name = "trade_price_controls"

    def __init__(self, max_exchange_ratio: float = 1.0):
        if max_exchange_ratio <= 0.0:
            raise ValueError("max_exchange_ratio must be positive")
        self.max_exchange_ratio = float(max_exchange_ratio)

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        if out.get("phase") != "pre_trade":
            return out
        food_units = float(out.get("food_units", 0.0))
        wood_units = float(out.get("wood_units", 0.0))
        if food_units <= 0.0 or wood_units <= 0.0:
            out["allowed"] = False
            return out
        ratio = max(food_units / wood_units, wood_units / food_units)
        if ratio > self.max_exchange_ratio:
            out["allowed"] = False
        return out


@register_institution("reputation_system")
class ReputationSystem(Institution):
    """Reward Resource Island agents for successful trading reputation."""

    name = "reputation_system"

    def __init__(self, trade_reputation_gain: float = 1.0, reward_bonus: float = 0.02):
        self.trade_reputation_gain = float(trade_reputation_gain)
        self.reward_bonus = float(reward_bonus)
        self.reputation: dict[int, float] = {}

    def reset(self) -> None:
        self.reputation = {}

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        phase = out.get("phase")
        if phase == "post_trade":
            for agent_id in out.get("participants", ()):
                agent = int(agent_id)
                self.reputation[agent] = self.reputation.get(agent, 0.0) + self.trade_reputation_gain
        elif phase == "post_rewards":
            rewards = np.asarray(out["rewards"], dtype=float).copy()
            for agent_id, score in self.reputation.items():
                if 0 <= agent_id < len(rewards):
                    rewards[agent_id] += self.reward_bonus * score
            out["rewards"] = rewards
            out["reputation"] = dict(self.reputation)
        return out
