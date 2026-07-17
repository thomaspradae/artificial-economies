from __future__ import annotations

import numpy as np


def _add_unit_to_cell(resources: np.ndarray, row: int, col: int, resource_type: int, resource_capacity: int) -> bool:
    if resources[row, col, resource_type] >= resource_capacity:
        return False
    resources[row, col, resource_type] += 1
    return True


def layout_resource_map(
    grid_size: int,
    n_resource_types: int,
    resource_capacity: int,
    initial_resource_units: int,
    resource_layout: str,
) -> np.ndarray:
    """Create deterministic resource layouts for economics-focused ablations."""
    resources = np.zeros((grid_size, grid_size, n_resource_types), dtype=int)
    if initial_resource_units <= 0:
        return resources

    if resource_layout == "contested":
        center = grid_size // 2
        cells = [
            (center, center),
            (max(0, center - 1), center),
            (min(grid_size - 1, center + 1), center),
            (center, max(0, center - 1)),
            (center, min(grid_size - 1, center + 1)),
        ]
    elif resource_layout == "split":
        cells = [(0, 0), (grid_size - 1, grid_size - 1)]
    else:
        raise ValueError("resource_layout must be random, contested, or split")

    placed = 0
    cursor = 0
    while placed < initial_resource_units:
        if resource_layout == "split":
            resource_type = cursor % n_resource_types
            row, col = cells[resource_type % len(cells)]
        else:
            row, col = cells[cursor % len(cells)]
            resource_type = (cursor // len(cells)) % n_resource_types
        if _add_unit_to_cell(resources, row, col, resource_type, resource_capacity):
            placed += 1
        cursor += 1
        if cursor > initial_resource_units * len(cells) * max(1, n_resource_types) * resource_capacity:
            break
    return resources


def initial_resource_map(
    rng: np.random.Generator,
    grid_size: int,
    n_resource_types: int,
    resource_capacity: int,
    initial_resource_units: int,
    initial_resources: np.ndarray | None = None,
    resource_layout: str = "random",
) -> np.ndarray:
    """Create a bounded integer resource map for Resource Island."""
    expected = (grid_size, grid_size, n_resource_types)
    if initial_resources is not None:
        resources = np.asarray(initial_resources, dtype=int).copy()
        if resources.shape != expected:
            raise ValueError(f"initial_resources must have shape {expected}")
        return np.clip(resources, 0, resource_capacity)

    if resource_layout != "random":
        return layout_resource_map(
            grid_size=grid_size,
            n_resource_types=n_resource_types,
            resource_capacity=resource_capacity,
            initial_resource_units=initial_resource_units,
            resource_layout=resource_layout,
        )

    resources = np.zeros(expected, dtype=int)
    for _ in range(initial_resource_units):
        row = int(rng.integers(grid_size))
        col = int(rng.integers(grid_size))
        resource_type = int(rng.integers(n_resource_types))
        resources[row, col, resource_type] = min(
            resource_capacity,
            resources[row, col, resource_type] + 1,
        )
    return resources


def local_resource_count(resources: np.ndarray, position: tuple[int, int], radius: int) -> int:
    """Count resources inside a square local-visibility window."""
    row, col = (int(value) for value in position)
    grid_size = int(resources.shape[0])
    r0 = max(0, row - radius)
    r1 = min(grid_size, row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(grid_size, col + radius + 1)
    return int(np.sum(resources[r0:r1, c0:c1, :]))


def regenerate_resource(
    resources: np.ndarray,
    rng: np.random.Generator,
    spawn_probability: float,
    resource_capacity: int,
) -> bool:
    """Stochastically add one resource unit, respecting per-cell capacity."""
    if spawn_probability <= 0.0:
        return False
    if rng.random() >= spawn_probability:
        return False
    row = int(rng.integers(resources.shape[0]))
    col = int(rng.integers(resources.shape[1]))
    resource_type = int(rng.integers(resources.shape[2]))
    resources[row, col, resource_type] = min(
        resource_capacity,
        resources[row, col, resource_type] + 1,
    )
    return True
