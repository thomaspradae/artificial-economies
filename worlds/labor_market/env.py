from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.agent import Agent
from core.institution import Institution
from core.metrics import finite_mean
from core.registry import register_world
from core.world import World
from institutions.none import NoInstitution
from worlds.labor_market.benchmarks import (
    blocking_pairs,
    deferred_acceptance,
    preference_order,
    reported_preferences_from_top,
)


@dataclass
class LaborMarketConfig:
    n_workers: int = 3
    n_employers: int = 3
    max_rounds: int = 1
    worker_values: tuple[tuple[float, ...], ...] | None = None
    employer_values: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.n_workers < 1 or self.n_employers < 1:
            raise ValueError("n_workers and n_employers must be positive")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if self.worker_values is not None and len(self.worker_values) != self.n_workers:
            raise ValueError("worker_values must have one row per worker")
        if self.employer_values is not None and len(self.employer_values) != self.n_employers:
            raise ValueError("employer_values must have one row per employer")


@register_world("labor_market")
class LaborMarketWorld(World):
    """Asymmetric two-sided matching world with learning workers."""

    def __init__(
        self,
        agents: list[Agent] | None = None,
        institution: Institution | None = None,
        seed: int | None = None,
        config: LaborMarketConfig | None = None,
    ):
        super().__init__(agents=agents, institution=institution, seed=seed)
        self.config = config if config is not None else LaborMarketConfig()
        self.institution = institution if institution is not None else NoInstitution()
        self.rng = np.random.default_rng(0 if seed is None else seed)
        self.round_idx = 0
        self.worker_values = np.zeros((self.config.n_workers, self.config.n_employers), dtype=float)
        self.employer_values = np.zeros((self.config.n_employers, self.config.n_workers), dtype=float)
        self.worker_preferences = np.zeros((self.config.n_workers, self.config.n_employers), dtype=int)
        self.employer_preferences = np.zeros((self.config.n_employers, self.config.n_workers), dtype=int)
        self.history: list[dict[str, Any]] = []
        self.reset()

    def reset(self) -> list[tuple[int, int]]:
        self.round_idx = 0
        self.history = []
        self.institution.reset()
        for agent in self.agents:
            agent.reset()
        self.worker_values = self._worker_values()
        self.employer_values = self._employer_values()
        self.worker_preferences = preference_order(self.worker_values)
        self.employer_preferences = preference_order(self.employer_values)
        return self.observations()

    def observations(self) -> list[tuple[int, int]]:
        return [
            (worker % self.config.n_employers, int(self.worker_preferences[worker, 0]))
            for worker in range(self.config.n_workers)
        ]

    def step(self, actions: list[Any]) -> tuple[list[tuple[int, int]], np.ndarray, bool, dict[str, Any]]:
        if len(actions) != self.config.n_workers:
            raise ValueError(f"LaborMarketWorld expects {self.config.n_workers} worker reports")
        reported_tops = np.asarray([self._validate_action(action) for action in actions], dtype=int)
        reported_preferences = reported_preferences_from_top(self.worker_preferences, reported_tops)
        state = self.institution.apply(
            {
                "phase": "labor_market_match",
                "reported_worker_preferences": reported_preferences.copy(),
                "employer_preferences": self.employer_preferences.copy(),
            }
        )
        matches = np.asarray(
            state.get("matches", deferred_acceptance(reported_preferences, self.employer_preferences)),
            dtype=int,
        )
        rewards = np.zeros(self.config.n_workers, dtype=float)
        employer_payoffs = np.zeros(self.config.n_employers, dtype=float)
        for worker, employer in enumerate(matches):
            if employer >= 0:
                rewards[worker] = self.worker_values[worker, employer]
                employer_payoffs[employer] = self.employer_values[employer, worker]
        info = self._info(reported_tops, reported_preferences, matches, rewards, employer_payoffs)
        self.history.append(info)
        self.round_idx += 1
        done = self.round_idx >= self.config.max_rounds
        return self.observations(), rewards, done, info

    def _validate_action(self, action: Any) -> int:
        value = int(action)
        if value < 0 or value >= self.config.n_employers:
            raise ValueError(f"reported employer {value} outside [0, {self.config.n_employers})")
        return value

    def _worker_values(self) -> np.ndarray:
        if self.config.worker_values is not None:
            return np.asarray(self.config.worker_values, dtype=float)
        return self.rng.uniform(0.0, 1.0, size=(self.config.n_workers, self.config.n_employers))

    def _employer_values(self) -> np.ndarray:
        if self.config.employer_values is not None:
            return np.asarray(self.config.employer_values, dtype=float)
        return self.rng.uniform(0.0, 1.0, size=(self.config.n_employers, self.config.n_workers))

    def _info(
        self,
        reported_tops: np.ndarray,
        reported_preferences: np.ndarray,
        matches: np.ndarray,
        rewards: np.ndarray,
        employer_payoffs: np.ndarray,
    ) -> dict[str, Any]:
        blocks = blocking_pairs(matches, self.worker_preferences, self.employer_preferences)
        truthful_tops = self.worker_preferences[:, 0]
        truthful_rate = float(np.mean(reported_tops == truthful_tops))
        manipulation_gains = [
            self._best_report_gain(worker, int(reported_tops[worker]), matches, rewards)
            for worker in range(self.config.n_workers)
        ]
        match_rate = float(np.mean(matches >= 0))
        return {
            "round": float(self.round_idx),
            "matches": tuple(int(value) for value in matches),
            "reported_tops": tuple(int(value) for value in reported_tops),
            "match_rate": match_rate,
            "worker_welfare": float(np.sum(rewards)),
            "employer_welfare": float(np.sum(employer_payoffs)),
            "total_welfare": float(np.sum(rewards) + np.sum(employer_payoffs)),
            "blocking_pairs": float(len(blocks)),
            "stability": float(len(blocks) == 0),
            "truthful_report_rate": truthful_rate,
            "manipulation_gain_mean": finite_mean(manipulation_gains),
            "matched_worker_utility_mean": finite_mean(rewards),
            "unmatched_count": float(np.sum(matches < 0)),
        }

    def _best_report_gain(self, worker: int, chosen_report: int, matches: np.ndarray, rewards: np.ndarray) -> float:
        current = float(rewards[worker])
        gains = []
        for report in range(self.config.n_employers):
            trial_tops = self.worker_preferences[:, 0].copy()
            trial_tops[worker] = report
            trial_preferences = reported_preferences_from_top(self.worker_preferences, trial_tops)
            trial_matches = deferred_acceptance(trial_preferences, self.employer_preferences)
            employer = int(trial_matches[worker])
            payoff = 0.0 if employer < 0 else float(self.worker_values[worker, employer])
            gains.append(payoff - current)
        del chosen_report, matches
        return float(max(gains)) if gains else 0.0

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
            "worker_values": self.worker_values.tolist(),
            "employer_values": self.employer_values.tolist(),
            "worker_preferences": self.worker_preferences.tolist(),
            "employer_preferences": self.employer_preferences.tolist(),
            "last_info": self.history[-1] if self.history else None,
        }
