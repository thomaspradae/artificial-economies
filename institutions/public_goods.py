from __future__ import annotations

from typing import Any

import numpy as np

from core.institution import Institution
from core.registry import register_institution


@register_institution("public_goods_penalty")
class PublicGoodsPenalty(Institution):
    """Penalize extraction above a sustainable per-agent allowance."""

    name = "public_goods_penalty"

    def __init__(self, sustainable_extraction: float = 1.0, penalty_rate: float = 0.5):
        self.sustainable_extraction = float(sustainable_extraction)
        self.penalty_rate = float(penalty_rate)

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        if out.get("phase") != "public_goods_post_rewards":
            return out
        realized = np.asarray(out["realized_extractions"], dtype=float)
        rewards = np.asarray(out["rewards"], dtype=float).copy()
        excess = np.maximum(realized - self.sustainable_extraction, 0.0)
        penalties = self.penalty_rate * excess
        rewards -= penalties
        out["rewards"] = rewards
        out["penalty_total"] = float(np.sum(penalties))
        return out


@register_institution("contribution_matching")
class ContributionMatching(Institution):
    """Match public-good contributions with extra pool stock."""

    name = "contribution_matching"

    def __init__(self, match_rate: float = 0.5):
        if match_rate < 0.0:
            raise ValueError("match_rate must be non-negative")
        self.match_rate = float(match_rate)

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        if out.get("phase") != "public_goods_pre_transition":
            return out
        contributions = np.asarray(out["contributions"], dtype=float)
        out["matched_contribution"] = float(self.match_rate * np.sum(contributions))
        return out


@register_institution("public_goods_reputation")
class PublicGoodsReputation(Institution):
    """Reward repeated contributors with a small reputation bonus."""

    name = "public_goods_reputation"

    def __init__(self, contribution_reputation_gain: float = 1.0, reward_bonus: float = 0.002):
        self.contribution_reputation_gain = float(contribution_reputation_gain)
        self.reward_bonus = float(reward_bonus)
        self.reputation: dict[int, float] = {}

    def reset(self) -> None:
        self.reputation = {}

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        if out.get("phase") != "public_goods_post_rewards":
            return out
        contributions = np.asarray(out["contributions"], dtype=float)
        rewards = np.asarray(out["rewards"], dtype=float).copy()
        for agent_id, contribution in enumerate(contributions):
            if contribution > 0.0:
                self.reputation[agent_id] = self.reputation.get(agent_id, 0.0) + self.contribution_reputation_gain
        bonus_total = 0.0
        for agent_id, score in self.reputation.items():
            if 0 <= agent_id < len(rewards):
                bonus = self.reward_bonus * score
                rewards[agent_id] += bonus
                bonus_total += bonus
        out["rewards"] = rewards
        out["reputation_bonus_total"] = float(bonus_total)
        return out


@register_institution("information_restriction")
class InformationRestriction(Institution):
    """Hide public-pool stock from tabular observations."""

    name = "information_restriction"

    def __init__(self, hidden_pool_bin: int = 2):
        self.hidden_pool_bin = int(hidden_pool_bin)

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        if out.get("phase") == "public_goods_observation":
            out["pool_bin"] = self.hidden_pool_bin
            out["information_hidden"] = 1.0
        return out
