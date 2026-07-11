from __future__ import annotations

from typing import Any

import numpy as np

from worlds.resource_island.env import FOOD, WOOD, ResourceIslandWorld
from worlds.resource_island.trading import manhattan_distance


def resource_island_obs_dim(radius: int = 1) -> int:
    """Fixed structured-observation width for Resource Island neural minds."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    window_cells = (2 * radius + 1) ** 2
    return 1 + 2 + 2 + (2 * window_cells) + 1 + 1 + 1


def encode_resource_island_observation(
    world: ResourceIslandWorld,
    agent_id: int,
    radius: int = 1,
) -> np.ndarray:
    """Encode one agent's Resource Island view as a fixed-length float vector.

    Feature order:
    energy, own food/wood inventory, own row/col position, local food plane,
    local wood plane, nearby-agent count, nearby inventory imbalance mean,
    alive flag.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if not 0 <= agent_id < world.config.n_agents:
        raise IndexError("agent_id out of range")

    cfg = world.config
    grid_size = cfg.grid_size
    inventory_scale = max(1.0, float(cfg.resource_capacity * grid_size * grid_size))
    position_scale = max(1.0, float(grid_size - 1))
    resource_scale = max(1.0, float(cfg.resource_capacity))

    row, col = (int(value) for value in world.positions[agent_id])
    features: list[float] = [
        float(np.clip(world.energy[agent_id] / cfg.max_energy, 0.0, 1.0)),
        float(np.clip(world.inventory[agent_id, FOOD] / inventory_scale, 0.0, 1.0)),
        float(np.clip(world.inventory[agent_id, WOOD] / inventory_scale, 0.0, 1.0)),
        float(row / position_scale),
        float(col / position_scale),
    ]

    food_plane: list[float] = []
    wood_plane: list[float] = []
    for d_row in range(-radius, radius + 1):
        for d_col in range(-radius, radius + 1):
            rr = row + d_row
            cc = col + d_col
            if 0 <= rr < grid_size and 0 <= cc < grid_size:
                food_plane.append(float(world.resources[rr, cc, FOOD] / resource_scale))
                wood_plane.append(float(world.resources[rr, cc, WOOD] / resource_scale))
            else:
                food_plane.append(0.0)
                wood_plane.append(0.0)
    features.extend(food_plane)
    features.extend(wood_plane)

    nearby_ids: list[int] = []
    for other_id in range(cfg.n_agents):
        if other_id == agent_id or not bool(world.alive[other_id]):
            continue
        distance = manhattan_distance(tuple(world.positions[agent_id]), tuple(world.positions[other_id]))
        if distance <= radius:
            nearby_ids.append(other_id)

    max_other_agents = max(1, cfg.n_agents - 1)
    features.append(float(len(nearby_ids) / max_other_agents))
    if nearby_ids:
        imbalances = []
        for other_id in nearby_ids:
            food = float(world.inventory[other_id, FOOD])
            wood = float(world.inventory[other_id, WOOD])
            denominator = max(1.0, food + wood)
            imbalances.append((food - wood) / denominator)
        features.append(float(np.clip(np.mean(imbalances), -1.0, 1.0)))
    else:
        features.append(0.0)

    features.append(1.0 if bool(world.alive[agent_id]) else 0.0)
    array = np.asarray(features, dtype=np.float32)
    expected = resource_island_obs_dim(radius)
    if len(array) != expected:
        raise RuntimeError(f"Resource Island feature width {len(array)} != expected {expected}")
    return array


def structured_observations(world: ResourceIslandWorld, radius: int = 1) -> list[np.ndarray]:
    """Return one fixed-width structured observation per Resource Island agent."""
    return [encode_resource_island_observation(world, agent_id, radius) for agent_id in range(world.config.n_agents)]


def assert_finite_observations(observations: list[Any]) -> None:
    """Fail fast when a structured observation contains NaN or infinity."""
    for obs in observations:
        array = np.asarray(obs, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError("Resource Island structured observation contains non-finite values")
