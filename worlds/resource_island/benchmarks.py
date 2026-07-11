from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GreedyPlanResult:
    """Full-information greedy planner summary for Resource Island."""

    estimated_gathered: int
    assignments: tuple[tuple[int, tuple[int, int], int], ...]


def efficient_gather_upper_bound(
    resources: np.ndarray,
    n_agents: int,
    steps: int,
    gather_amount: int = 1,
) -> int:
    """Maximum resource units gatherable over a horizon, ignoring travel and conflict."""
    if n_agents < 0 or steps < 0 or gather_amount < 1:
        raise ValueError("n_agents and steps must be nonnegative and gather_amount must be positive")
    total_resources = int(np.sum(np.asarray(resources, dtype=int)))
    max_agent_capacity = int(n_agents * steps * gather_amount)
    return min(total_resources, max_agent_capacity)


def resource_locations(resources: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return nonempty resource cells as (row, col, resource_type, amount)."""
    array = np.asarray(resources, dtype=int)
    if array.ndim != 3:
        raise ValueError("resources must have shape rows x cols x resource_types")
    locations: list[tuple[int, int, int, int]] = []
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            for resource_type in range(array.shape[2]):
                amount = int(array[row, col, resource_type])
                if amount > 0:
                    locations.append((row, col, resource_type, amount))
    return locations


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Manhattan distance between two grid cells."""
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def greedy_full_information_gather_plan(
    positions: Iterable[tuple[int, int]],
    resources: np.ndarray,
    steps: int,
) -> GreedyPlanResult:
    """Greedy full-information gather estimate with nearest-resource assignment."""
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    resource_cells = resource_locations(resources)
    remaining = {(row, col, resource_type): amount for row, col, resource_type, amount in resource_cells}
    assignments: list[tuple[int, tuple[int, int], int]] = []
    gathered = 0

    for agent_id, position in enumerate(positions):
        candidates = []
        for row, col, resource_type, amount in resource_cells:
            if remaining.get((row, col, resource_type), 0) <= 0:
                continue
            distance = manhattan(position, (row, col))
            turns_needed = distance + 1
            if turns_needed <= steps:
                candidates.append((distance, row, col, resource_type, amount))
        if not candidates:
            continue
        _, row, col, resource_type, _ = min(candidates)
        key = (row, col, resource_type)
        remaining[key] -= 1
        gathered += 1
        assignments.append((agent_id, (row, col), resource_type))

    return GreedyPlanResult(estimated_gathered=gathered, assignments=tuple(assignments))
