from __future__ import annotations

from typing import Any

import numpy as np

from arena_v0 import MarketConfig, collusion_index, logsumexp
from core.agent import Agent
from core.institution import Institution
from core.registry import register_world
from core.world import World
from institutions.none import NoInstitution
from worlds.pricing_arena.benchmarks import compute_static_benchmarks


def _institution_name(institution: Institution | None) -> str:
    if institution is None:
        return "none"
    return str(getattr(institution, "name", institution.__class__.__name__))


@register_world("pricing_arena")
class PricingArenaWorld(World):
    """World wrapper for the existing repeated duopoly pricing arena."""

    def __init__(
        self,
        agents: list[Agent] | None = None,
        institution: Institution | None = None,
        seed: int | None = None,
        config: MarketConfig | None = None,
    ):
        super().__init__(agents=agents, institution=institution, seed=seed)
        if config is None:
            config = MarketConfig(mechanism=_institution_name(institution))
        self.config = config
        self.institution = institution if institution is not None else NoInstitution()
        self.rng = np.random.default_rng(0 if seed is None else seed)
        self.n_firms = int(len(np.asarray(self.config.quality, dtype=float)))
        if self.n_firms < 2:
            raise ValueError("PricingArenaWorld requires at least two firms")
        self.benchmarks = compute_static_benchmarks(
            self.config.price_grid,
            n_firms=self.n_firms,
            cost=self.config.cost,
            market_size=self.config.market_size,
            alpha=self.config.alpha,
            tau=self.config.tau,
            quality=np.asarray(self.config.quality, dtype=float),
        )
        self.n_prices = len(self.config.price_grid)
        self.state = tuple(self.n_prices // 2 for _ in range(self.n_firms))
        self.step_idx = 0
        self.history: list[dict[str, float]] = []

    def reset(self) -> tuple[int, ...]:
        self.rng = np.random.default_rng(0 if self.seed is None else self.seed)
        self.state = tuple(self.n_prices // 2 for _ in range(self.n_firms))
        self.step_idx = 0
        self.history = []
        if self.institution is not None:
            self.institution.reset()
        for agent in self.agents:
            agent.reset()
        return self.state

    def demand(self, prices: np.ndarray, market_size: float | None = None) -> np.ndarray:
        """Logit demand over outside option plus both firms."""
        size = self.config.market_size if market_size is None else market_size
        utilities = self.config.quality - self.config.alpha * prices
        logits = np.concatenate(([0.0], utilities)) / self.config.tau
        probabilities = np.exp(logits - logsumexp(logits))
        return size * probabilities[1:]

    def consumer_surplus_proxy(self, prices: np.ndarray, market_size: float | None = None) -> float:
        """Inclusive-value surplus proxy used by the original pricing arena."""
        size = self.config.market_size if market_size is None else market_size
        utilities = self.config.quality - self.config.alpha * prices
        logits = np.concatenate(([0.0], utilities)) / self.config.tau
        return float(size * (self.config.tau / self.config.alpha) * logsumexp(logits))

    def step(self, actions: list[Any]) -> tuple[tuple[int, ...], np.ndarray, bool, dict[str, Any]]:
        if len(actions) != self.n_firms:
            raise ValueError(f"PricingArenaWorld expects {self.n_firms} firm actions")
        action_pair = tuple(int(action) for action in actions)
        raw_prices = self.config.price_grid[np.array(action_pair, dtype=int)]
        prices = raw_prices.copy()
        market_size = self.config.market_size

        pre_state = {
            "phase": "pre_demand",
            "actions": action_pair,
            "prices": prices,
            "raw_prices": raw_prices,
            "market_size": market_size,
            "rng": self.rng,
            "step_idx": self.step_idx,
        }
        pre_state = self.institution.apply(pre_state) if self.institution is not None else pre_state
        prices = np.asarray(pre_state["prices"], dtype=float)
        market_size = float(pre_state["market_size"])

        quantities = self.demand(prices, market_size=market_size)
        profits = (prices - self.config.cost) * quantities
        penalties = np.zeros_like(prices, dtype=float)
        rewards = profits.copy()

        post_state = {
            "phase": "post_profit",
            "actions": action_pair,
            "prices": prices,
            "raw_prices": raw_prices,
            "market_size": market_size,
            "quantities": quantities,
            "profits": profits,
            "rewards": rewards,
            "penalties": penalties,
            "audit_hit": 0.0,
            "rng": self.rng,
            "step_idx": self.step_idx,
        }
        post_state = self.institution.apply(post_state) if self.institution is not None else post_state
        rewards = np.asarray(post_state["rewards"], dtype=float)
        penalties = np.asarray(post_state["penalties"], dtype=float)
        audit_hit = float(post_state.get("audit_hit", 0.0))

        consumer_surplus = self.consumer_surplus_proxy(prices, market_size=market_size)
        welfare = float(np.sum(profits) + consumer_surplus)
        next_state = tuple(int(i) for i in action_pair)
        info = {
            "avg_price": float(np.mean(prices)),
            "price_dispersion": float(abs(prices[0] - prices[1])) if self.n_firms == 2 else float(np.std(prices)),
            "profit_total": float(np.sum(profits)),
            "reward_total": float(np.sum(rewards)),
            "quantity_total": float(np.sum(quantities)),
            "penalty_total": float(np.sum(penalties)),
            "consumer_surplus": consumer_surplus,
            "welfare": welfare,
            "market_size": float(market_size),
            "audit_hit": audit_hit,
            "n_firms": float(self.n_firms),
        }
        for index in range(self.n_firms):
            suffix = index + 1
            info[f"p{suffix}"] = float(prices[index])
            info[f"raw_p{suffix}"] = float(raw_prices[index])
            info[f"quantity{suffix}"] = float(quantities[index])
            info[f"profit{suffix}"] = float(profits[index])
            info[f"reward{suffix}"] = float(rewards[index])
            info[f"penalty{suffix}"] = float(penalties[index])
        info["step"] = float(self.step_idx)
        info["collusion_index"] = collusion_index(
            info["avg_price"],
            self.benchmarks["nash_price"],
            float(self.benchmarks["monopoly_price"]),
        )
        self.state = next_state
        self.history.append(info)
        self.step_idx += 1
        return next_state, rewards, False, info

    def get_metrics(self) -> dict[str, Any]:
        if not self.history:
            return {}
        keys = self.history[0].keys()
        return {
            key: float(np.nanmean([record[key] for record in self.history]))
            for key in keys
            if isinstance(self.history[0][key], (float, int, np.floating, np.integer))
        }

    def render_state(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "step_idx": self.step_idx,
            "last_info": self.history[-1] if self.history else None,
        }
