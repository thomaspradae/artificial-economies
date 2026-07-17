from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.agent import Agent
from core.institution import Institution
from core.metrics import finite_mean, gini
from core.registry import register_world
from core.world import World
from institutions.none import NoInstitution
from worlds.public_goods.benchmarks import (
    contribution_amount,
    extraction_request,
    next_pool_stock,
    ration_extractions,
)


NOOP = 0
CONTRIBUTE_LOW = 1
CONTRIBUTE_HIGH = 2
EXTRACT_LOW = 3
EXTRACT_HIGH = 4
N_ACTIONS = 5


@dataclass
class PublicGoodsConfig:
    n_agents: int = 4
    max_rounds: int = 1
    pool_capacity: float = 20.0
    initial_pool: float = 10.0
    collapse_threshold: float = 2.0
    contribution_unit: float = 1.0
    extraction_unit: float = 1.0
    contribution_cost: float = 0.4
    extraction_reward: float = 1.0
    public_multiplier: float = 1.4
    regeneration_rate: float = 0.08
    pool_value_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.n_agents < 1:
            raise ValueError("n_agents must be positive")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if self.pool_capacity <= 0.0:
            raise ValueError("pool_capacity must be positive")
        if not 0.0 <= self.initial_pool <= self.pool_capacity:
            raise ValueError("initial_pool must be in [0, pool_capacity]")
        if not 0.0 <= self.regeneration_rate <= 1.0:
            raise ValueError("regeneration_rate must be in [0, 1]")


@register_world("public_goods")
class PublicGoodsWorld(World):
    """Repeated public-goods / common-pool resource world."""

    def __init__(
        self,
        agents: list[Agent] | None = None,
        institution: Institution | None = None,
        seed: int | None = None,
        config: PublicGoodsConfig | None = None,
    ):
        super().__init__(agents=agents, institution=institution, seed=seed)
        self.config = config if config is not None else PublicGoodsConfig()
        self.institution = institution if institution is not None else NoInstitution()
        self.rng = np.random.default_rng(0 if seed is None else seed)
        self.round_idx = 0
        self.pool_stock = float(self.config.initial_pool)
        self.cumulative_rewards = np.zeros(self.config.n_agents, dtype=float)
        self.history: list[dict[str, Any]] = []
        self.reset()

    def reset(self) -> list[tuple[int, int]]:
        self.round_idx = 0
        self.pool_stock = float(self.config.initial_pool)
        self.cumulative_rewards = np.zeros(self.config.n_agents, dtype=float)
        self.history = []
        self.institution.reset()
        for agent in self.agents:
            agent.reset()
        return self.observations()

    def observations(self) -> list[tuple[int, int]]:
        obs = []
        for agent_id in range(self.config.n_agents):
            raw = {
                "phase": "public_goods_observation",
                "agent_id": agent_id,
                "pool_bin": self.pool_bin(self.pool_stock),
                "agent_bin": agent_id % N_ACTIONS,
                "pool_stock": self.pool_stock,
            }
            state = self.institution.apply(raw)
            obs.append((int(state.get("pool_bin", raw["pool_bin"])), int(state.get("agent_bin", raw["agent_bin"]))))
        return obs

    def step(self, actions: list[Any]) -> tuple[list[tuple[int, int]], np.ndarray, bool, dict[str, Any]]:
        if len(actions) != self.config.n_agents:
            raise ValueError(f"PublicGoodsWorld expects {self.config.n_agents} actions")
        action_array = np.asarray([self._validate_action(action) for action in actions], dtype=int)
        contributions = np.asarray(
            [contribution_amount(action, self.config.contribution_unit) for action in action_array],
            dtype=float,
        )
        requests = np.asarray(
            [extraction_request(action, self.config.extraction_unit) for action in action_array],
            dtype=float,
        )
        realized = ration_extractions(requests, self.pool_stock)
        rewards = self.config.extraction_reward * realized - self.config.contribution_cost * contributions

        pre_state = self.institution.apply(
            {
                "phase": "public_goods_pre_transition",
                "actions": action_array.copy(),
                "contributions": contributions.copy(),
                "extraction_requests": requests.copy(),
                "realized_extractions": realized.copy(),
                "pool_stock": float(self.pool_stock),
                "matched_contribution": 0.0,
            }
        )
        matched_contribution = float(pre_state.get("matched_contribution", 0.0))
        self.pool_stock = next_pool_stock(
            self.pool_stock,
            contributions,
            realized,
            pool_capacity=self.config.pool_capacity,
            public_multiplier=self.config.public_multiplier,
            regeneration_rate=self.config.regeneration_rate,
            matched_contribution=matched_contribution,
        )
        post_state = self.institution.apply(
            {
                "phase": "public_goods_post_rewards",
                "actions": action_array.copy(),
                "contributions": contributions.copy(),
                "extraction_requests": requests.copy(),
                "realized_extractions": realized.copy(),
                "pool_stock": float(self.pool_stock),
                "rewards": rewards.copy(),
            }
        )
        rewards = np.asarray(post_state.get("rewards", rewards), dtype=float)
        self.cumulative_rewards += rewards
        info = self._info(
            action_array,
            contributions,
            requests,
            realized,
            rewards,
            matched_contribution=matched_contribution,
            post_state=post_state,
        )
        self.history.append(info)
        self.round_idx += 1
        done = self.round_idx >= self.config.max_rounds
        return self.observations(), rewards, done, info

    def _validate_action(self, action: Any) -> int:
        value = int(action)
        if value < 0 or value >= N_ACTIONS:
            raise ValueError(f"Public Goods action {value} outside [0, {N_ACTIONS})")
        return value

    def pool_bin(self, pool_stock: float) -> int:
        if self.config.pool_capacity <= 0.0:
            return 0
        scaled = float(pool_stock) / float(self.config.pool_capacity)
        return int(np.clip(np.floor(scaled * N_ACTIONS), 0, N_ACTIONS - 1))

    def _info(
        self,
        actions: np.ndarray,
        contributions: np.ndarray,
        requests: np.ndarray,
        realized: np.ndarray,
        rewards: np.ndarray,
        matched_contribution: float,
        post_state: dict[str, Any],
    ) -> dict[str, Any]:
        contribution_capacity = self.config.n_agents * 2.0 * self.config.contribution_unit
        extraction_capacity = self.config.n_agents * 2.0 * self.config.extraction_unit
        collapse = float(self.pool_stock <= self.config.collapse_threshold)
        return {
            "round": float(self.round_idx),
            "pool_stock": float(self.pool_stock),
            "sustainability": float(self.pool_stock / self.config.pool_capacity),
            "contribution_total": float(np.sum(contributions)),
            "extraction_request_total": float(np.sum(requests)),
            "extraction_total": float(np.sum(realized)),
            "contribution_rate": float(np.sum(contributions) / contribution_capacity) if contribution_capacity else 0.0,
            "extraction_rate": float(np.sum(realized) / extraction_capacity) if extraction_capacity else 0.0,
            "collapse": collapse,
            "collapse_rate": collapse,
            "reward_total": float(np.sum(rewards)),
            "welfare": float(np.sum(rewards) + self.config.pool_value_weight * self.pool_stock),
            "inequality": gini(np.maximum(self.cumulative_rewards, 0.0)),
            "mean_reward": float(np.mean(rewards)),
            "matched_contribution": float(matched_contribution),
            "penalty_total": float(post_state.get("penalty_total", 0.0)),
            "reputation_bonus_total": float(post_state.get("reputation_bonus_total", 0.0)),
            "tax_revenue": float(post_state.get("tax_revenue", 0.0)),
            "information_hidden": float(post_state.get("information_hidden", 0.0)),
            "contributor_count": float(np.sum(contributions > 0.0)),
            "extractor_count": float(np.sum(realized > 0.0)),
            "noop_count": float(np.sum(actions == NOOP)),
        }

    def get_metrics(self) -> dict[str, Any]:
        if not self.history:
            return {}
        numeric_keys = [
            key
            for key, value in self.history[0].items()
            if isinstance(value, (float, int, np.floating, np.integer, bool))
        ]
        return {key: finite_mean(record[key] for record in self.history) for key in numeric_keys}

    def render_state(self) -> dict[str, Any]:
        return {
            "round_idx": self.round_idx,
            "pool_stock": self.pool_stock,
            "pool_bin": self.pool_bin(self.pool_stock),
            "cumulative_rewards": self.cumulative_rewards.tolist(),
            "last_info": self.history[-1] if self.history else None,
        }
