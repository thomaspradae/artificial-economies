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


AuctionFormat = Literal["first_price", "second_price"]


@dataclass
class AuctionHouseConfig:
    """Configuration for a single-item sealed-bid auction world."""

    n_bidders: int = 2
    auction_format: AuctionFormat = "second_price"
    max_rounds: int = 1
    bid_grid: tuple[float, ...] = tuple(float(value) for value in range(11))
    valuation_low: float = 0.0
    valuation_high: float = 10.0
    fixed_valuations: tuple[float, ...] | None = field(default=None)

    def __post_init__(self) -> None:
        if self.n_bidders < 2:
            raise ValueError("n_bidders must be at least 2")
        if self.auction_format not in ("first_price", "second_price"):
            raise ValueError("auction_format must be first_price or second_price")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if len(self.bid_grid) < 2:
            raise ValueError("bid_grid must contain at least two bids")
        if any(self.bid_grid[index] > self.bid_grid[index + 1] for index in range(len(self.bid_grid) - 1)):
            raise ValueError("bid_grid must be sorted ascending")
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
        self.valuations = np.zeros(self.config.n_bidders, dtype=float)
        self.history: list[dict[str, Any]] = []
        self.reset()

    def reset(self) -> list[float]:
        self.rng = np.random.default_rng(0 if self.seed is None else self.seed)
        self.round_idx = 0
        self.history = []
        self.institution.reset()
        for agent in self.agents:
            agent.reset()
        self.valuations = self._draw_valuations()
        return self.observations()

    def observations(self) -> list[float]:
        return [float(value) for value in self.valuations]

    def step(self, actions: list[Any]) -> tuple[list[float], np.ndarray, bool, dict[str, Any]]:
        if len(actions) != self.config.n_bidders:
            raise ValueError(f"AuctionHouseWorld expects {self.config.n_bidders} bids")

        bids = self._bids_from_actions(actions)
        winner = self.allocate(bids)
        payment = self.payment_for_winner(bids, winner)
        rewards = np.zeros(self.config.n_bidders, dtype=float)
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

    @staticmethod
    def allocate(bids: np.ndarray) -> int:
        """Return the lowest bidder id among tied highest bids."""
        return int(np.flatnonzero(bids == np.max(bids))[0])

    def payment_for_winner(self, bids: np.ndarray, winner: int) -> float:
        if self.config.auction_format == "first_price":
            return float(bids[winner])
        sorted_bids = np.sort(bids)
        return float(sorted_bids[-2])

    def _info(self, bids: np.ndarray, winner: int, payment: float, rewards: np.ndarray) -> dict[str, Any]:
        efficient_winner = int(np.flatnonzero(self.valuations == np.max(self.valuations))[0])
        max_value = float(np.max(self.valuations))
        efficiency = 1.0 if winner == efficient_winner else 0.0
        return {
            "round": float(self.round_idx),
            "auction_format": self.config.auction_format,
            "bids": tuple(float(value) for value in bids),
            "valuations": tuple(float(value) for value in self.valuations),
            "winner": float(winner),
            "efficient_winner": float(efficient_winner),
            "payment": float(payment),
            "revenue": float(payment),
            "reward_total": float(np.sum(rewards)),
            "welfare": float(self.valuations[winner]),
            "max_possible_welfare": max_value,
            "allocative_efficiency": efficiency,
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
            "last_info": self.history[-1] if self.history else None,
        }
