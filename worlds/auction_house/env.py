from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from core.agent import Agent
from core.institution import Institution
from core.metrics import finite_mean
from core.registry import register_world
from core.world import World
from institutions.none import NoInstitution
from worlds.auction_house.benchmarks import (
    allocate_winner,
    auction_outcome,
    ex_post_bidder_regret,
    first_price_bid_benchmark,
    payment_for_winner,
    truthful_bid_benchmark,
)


AuctionFormat = Literal["first_price", "second_price", "clock"]


@dataclass
class AuctionHouseConfig:
    """Configuration for a single-item sealed-bid auction world."""

    n_bidders: int = 2
    auction_format: AuctionFormat = "second_price"
    max_rounds: int = 1
    bid_grid: tuple[float, ...] = tuple(float(value) for value in range(11))
    valuation_grid: tuple[float, ...] | None = None
    valuation_low: float = 0.0
    valuation_high: float = 10.0
    reserve_price: float = 0.0
    fixed_valuations: tuple[float, ...] | None = field(default=None)

    def __post_init__(self) -> None:
        if self.valuation_grid is None:
            self.valuation_grid = tuple(float(value) for value in self.bid_grid)
        if self.n_bidders < 2:
            raise ValueError("n_bidders must be at least 2")
        if self.auction_format not in ("first_price", "second_price", "clock"):
            raise ValueError("auction_format must be first_price, second_price, or clock")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if len(self.bid_grid) < 2:
            raise ValueError("bid_grid must contain at least two bids")
        if any(self.bid_grid[index] > self.bid_grid[index + 1] for index in range(len(self.bid_grid) - 1)):
            raise ValueError("bid_grid must be sorted ascending")
        if len(self.valuation_grid) < 2:
            raise ValueError("valuation_grid must contain at least two values")
        if any(
            self.valuation_grid[index] > self.valuation_grid[index + 1]
            for index in range(len(self.valuation_grid) - 1)
        ):
            raise ValueError("valuation_grid must be sorted ascending")
        if self.reserve_price < 0.0:
            raise ValueError("reserve_price must be non-negative")
        if self.fixed_valuations is not None and len(self.fixed_valuations) != self.n_bidders:
            raise ValueError("fixed_valuations must have one value per bidder")


@register_world("auction_house")
class AuctionHouseWorld(World):
    """One-item sealed-bid auction with deterministic allocation and payments."""

    def __init__(
        self,
        agents: list[Agent] | None = None,
        institution: Institution | None = None,
        seed: int | None = None,
        config: AuctionHouseConfig | None = None,
    ):
        super().__init__(agents=agents, institution=institution, seed=seed)
        self.config = config if config is not None else AuctionHouseConfig()
        self.institution = institution if institution is not None else NoInstitution()
        self.rng = np.random.default_rng(0 if seed is None else seed)
        self.round_idx = 0
        self.reset_count = 0
        self.valuations = np.zeros(self.config.n_bidders, dtype=float)
        self.history: list[dict[str, Any]] = []
        self.reset()

    def reset(self) -> list[tuple[int]]:
        self.round_idx = 0
        self.history = []
        self.institution.reset()
        for agent in self.agents:
            agent.reset()
        self.valuations = self._draw_valuations()
        self.reset_count += 1
        return self.observations()

    def observations(self) -> list[tuple[int]]:
        """Private tabular observations: each bidder sees only its value bin."""
        obs: list[tuple[int]] = []
        for bidder_id, value in enumerate(self.valuations):
            raw = {
                "phase": "auction_observation",
                "agent_id": bidder_id,
                "valuation": float(value),
                "valuations": self.valuations.copy(),
                "valuation_grid": tuple(float(item) for item in self.config.valuation_grid),
                "valuation_bin": self.valuation_bin(float(value)),
                "n_bins": len(self.config.valuation_grid),
            }
            state = self.institution.apply(raw)
            observed_bin = int(np.clip(int(state.get("valuation_bin", raw["valuation_bin"])), 0, len(self.config.valuation_grid) - 1))
            obs.append((observed_bin,))
        return obs

    def step(self, actions: list[Any]) -> tuple[list[float], np.ndarray, bool, dict[str, Any]]:
        if len(actions) != self.config.n_bidders:
            raise ValueError(f"AuctionHouseWorld expects {self.config.n_bidders} bids")

        bids = self._bids_from_actions(actions)
        winner = self.allocate(bids)
        payment = self.payment_for_winner(bids, winner)
        rewards = np.zeros(self.config.n_bidders, dtype=float)
        if winner is not None:
            rewards[winner] = self.valuations[winner] - payment

        state = {
            "phase": "post_auction",
            "bids": bids.copy(),
            "valuations": self.valuations.copy(),
            "winner": winner,
            "payment": payment,
            "rewards": rewards.copy(),
        }
        state = self.institution.apply(state)
        rewards = np.asarray(state.get("rewards", rewards), dtype=float)
        payment = float(state.get("payment", payment))

        info = self._info(bids, winner, payment, rewards)
        self.history.append(info)
        self.round_idx += 1
        done = self.round_idx >= self.config.max_rounds
        if not done:
            self.valuations = self._draw_valuations()
        return self.observations(), rewards, done, info

    def _draw_valuations(self) -> np.ndarray:
        if self.config.fixed_valuations is not None:
            return np.asarray(self.config.fixed_valuations, dtype=float)
        if self.config.valuation_grid is not None:
            return self.rng.choice(
                np.asarray(self.config.valuation_grid, dtype=float),
                size=self.config.n_bidders,
                replace=True,
            ).astype(float)
        return self.rng.uniform(
            self.config.valuation_low,
            self.config.valuation_high,
            size=self.config.n_bidders,
        )

    def _bids_from_actions(self, actions: list[Any]) -> np.ndarray:
        bids = []
        grid = self.config.bid_grid
        for action in actions:
            if isinstance(action, (int, np.integer)):
                action_int = int(action)
                if action_int < 0 or action_int >= len(grid):
                    raise ValueError(f"bid action index {action_int} outside bid grid")
                bids.append(float(grid[action_int]))
            else:
                bids.append(float(action))
        return np.asarray(bids, dtype=float)

    def valuation_bin(self, value: float) -> int:
        grid = np.asarray(self.config.valuation_grid, dtype=float)
        return int(np.argmin(np.abs(grid - float(value))))

    def allocate(self, bids: np.ndarray) -> int | None:
        """Return the lowest bidder id among tied highest bids."""
        return allocate_winner(bids, reserve_price=self.config.reserve_price)

    def payment_for_winner(self, bids: np.ndarray, winner: int | None) -> float:
        return payment_for_winner(
            bids,
            winner,
            auction_format=self.config.auction_format,
            reserve_price=self.config.reserve_price,
        )

    def _info(self, bids: np.ndarray, winner: int | None, payment: float, rewards: np.ndarray) -> dict[str, Any]:
        outcome = auction_outcome(
            self.valuations,
            bids,
            auction_format=self.config.auction_format,
            reserve_price=self.config.reserve_price,
        )
        efficient_winner = outcome["efficient_winner"]
        truthful_bids = truthful_bid_benchmark(self.valuations, self.config.bid_grid)
        first_price_bids = first_price_bid_benchmark(
            self.valuations,
            n_bidders=self.config.n_bidders,
            valuation_low=self.config.valuation_low,
            bid_grid=self.config.bid_grid,
        )
        regrets = [
            ex_post_bidder_regret(
                self.valuations,
                bids,
                bidder=bidder,
                bid_grid=self.config.bid_grid,
                auction_format=self.config.auction_format,
                reserve_price=self.config.reserve_price,
            )
            for bidder in range(self.config.n_bidders)
        ]
        truthful_distances = np.abs(bids - np.asarray(truthful_bids, dtype=float))
        first_price_distances = np.abs(bids - np.asarray(first_price_bids, dtype=float))
        overbids = bids > self.valuations + 1e-12
        underbids = bids < self.valuations - 1e-12
        positive_values = self.valuations > 1e-12
        mean_bid_to_value = (
            float(np.mean(bids[positive_values] / self.valuations[positive_values]))
            if np.any(positive_values)
            else 0.0
        )
        winner_value = 0.0 if winner is None else float(self.valuations[winner])
        winner_bid = 0.0 if winner is None else float(bids[winner])
        return {
            "round": float(self.round_idx),
            "auction_format": self.config.auction_format,
            "bids": tuple(float(value) for value in bids),
            "valuations": tuple(float(value) for value in self.valuations),
            "winner": -1.0 if winner is None else float(winner),
            "efficient_winner": -1.0 if efficient_winner is None else float(efficient_winner),
            "payment": float(payment),
            "revenue": float(payment),
            "bidder_surplus": float(np.sum(rewards)),
            "reward_total": float(np.sum(rewards)),
            "welfare": float(outcome["welfare"]),
            "total_welfare": float(outcome["welfare"]),
            "max_possible_welfare": float(outcome["max_possible_welfare"]),
            "allocative_efficiency": float(outcome["allocative_efficiency"]),
            "welfare_efficiency": float(outcome["welfare_efficiency"]),
            "allocation_error": float(1.0 - float(outcome["allocative_efficiency"])),
            "winner_value": winner_value,
            "winner_bid": winner_bid,
            "mean_value": float(np.mean(self.valuations)),
            "mean_bid": float(np.mean(bids)),
            "mean_bid_to_value": mean_bid_to_value,
            "truthful_bid_distance_mean": float(np.mean(truthful_distances)),
            "first_price_shading_distance_mean": float(np.mean(first_price_distances)),
            "ex_post_regret_mean": float(np.mean(regrets)),
            "ex_post_regret_max": float(np.max(regrets)),
            "overbid_rate": float(np.mean(overbids)),
            "underbid_rate": float(np.mean(underbids)),
            "no_sale": float(winner is None),
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
            "valuations": self.valuations.tolist(),
            "valuation_bins": [self.valuation_bin(float(value)) for value in self.valuations],
            "last_info": self.history[-1] if self.history else None,
        }
