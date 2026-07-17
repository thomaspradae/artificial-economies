from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np


def contribution_amount(action: int, contribution_unit: float = 1.0) -> float:
    """Contribution implied by a Public Goods action."""
    if int(action) == 1:
        return float(contribution_unit)
    if int(action) == 2:
        return float(2.0 * contribution_unit)
    return 0.0


def extraction_request(action: int, extraction_unit: float = 1.0) -> float:
    """Extraction requested by a Public Goods action."""
    if int(action) == 3:
        return float(extraction_unit)
    if int(action) == 4:
        return float(2.0 * extraction_unit)
    return 0.0


def ration_extractions(requests: Iterable[float], available_pool: float) -> np.ndarray:
    """Proportionally ration extraction requests against available pool stock."""
    req = np.maximum(np.asarray(list(requests), dtype=float), 0.0)
    total = float(np.sum(req))
    if total <= 0.0 or available_pool <= 0.0:
        return np.zeros_like(req)
    if total <= available_pool:
        return req
    return req * (float(available_pool) / total)


def next_pool_stock(
    pool_stock: float,
    contributions: Iterable[float],
    realized_extractions: Iterable[float],
    pool_capacity: float,
    public_multiplier: float,
    regeneration_rate: float,
    matched_contribution: float = 0.0,
) -> float:
    """Deterministic pool transition used by the world and benchmarks."""
    pool = float(pool_stock)
    pool -= float(np.sum(list(realized_extractions)))
    pool += public_multiplier * float(np.sum(list(contributions)))
    pool += float(matched_contribution)
    pool = min(float(pool_capacity), max(0.0, pool))
    pool += regeneration_rate * (float(pool_capacity) - pool)
    return float(min(float(pool_capacity), max(0.0, pool)))


def simulate_fixed_policy(config: object, actions: list[int], steps: int = 50) -> dict[str, float]:
    """Roll out a fixed action profile using config fields shared with the world."""
    if len(actions) != int(config.n_agents):
        raise ValueError("actions must contain one action per agent")
    pool = float(config.initial_pool)
    welfare: list[float] = []
    sustainability: list[float] = []
    collapse: list[float] = []
    contributions_seen: list[float] = []
    extractions_seen: list[float] = []
    for _ in range(int(steps)):
        contributions = np.asarray(
            [contribution_amount(action, config.contribution_unit) for action in actions],
            dtype=float,
        )
        requests = np.asarray(
            [extraction_request(action, config.extraction_unit) for action in actions],
            dtype=float,
        )
        realized = ration_extractions(requests, pool)
        rewards = (
            float(config.extraction_reward) * realized
            - float(config.contribution_cost) * contributions
        )
        pool = next_pool_stock(
            pool,
            contributions,
            realized,
            pool_capacity=config.pool_capacity,
            public_multiplier=config.public_multiplier,
            regeneration_rate=config.regeneration_rate,
        )
        welfare.append(float(np.sum(rewards) + float(config.pool_value_weight) * pool))
        sustainability.append(float(pool / float(config.pool_capacity)))
        collapse.append(float(pool <= float(config.collapse_threshold)))
        contributions_seen.append(float(np.sum(contributions)))
        extractions_seen.append(float(np.sum(realized)))
    return {
        "welfare": float(np.mean(welfare)),
        "sustainability": float(np.mean(sustainability)),
        "collapse_rate": float(np.mean(collapse)),
        "contribution_total": float(np.mean(contributions_seen)),
        "extraction_total": float(np.mean(extractions_seen)),
    }


def free_rider_benchmark(config: object, steps: int = 50) -> dict[str, float]:
    """All agents extract high and never contribute."""
    return simulate_fixed_policy(config, [4 for _ in range(int(config.n_agents))], steps=steps)


def social_optimum_benchmark(config: object, steps: int = 50) -> dict[str, float]:
    """A conservative cooperative bracket: high contribution and low extraction."""
    cooperative = replace(config, regeneration_rate=float(config.regeneration_rate))
    return simulate_fixed_policy(cooperative, [2 for _ in range(int(config.n_agents))], steps=steps)
