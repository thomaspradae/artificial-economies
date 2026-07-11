from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.agent import Agent
from core.institution import Institution
from core.metrics import finite_mean, gini, resource_sustainability, specialization_index, survival_rate
from core.registry import register_world
from core.world import World
from institutions.none import NoInstitution
from worlds.resource_island.resources import (
    initial_resource_map,
    local_resource_count as count_local_resources,
    regenerate_resource,
)
from worlds.resource_island.trading import manhattan_distance, within_trade_radius


RESOURCE_TYPES = ("food", "wood")
FOOD = 0
WOOD = 1

STAY = 0
MOVE_UP = 1
MOVE_DOWN = 2
MOVE_LEFT = 3
MOVE_RIGHT = 4
GATHER = 5
OFFER_FOOD_FOR_WOOD = 6
OFFER_WOOD_FOR_FOOD = 7

ACTION_NAMES = {
    STAY: "stay",
    MOVE_UP: "move_up",
    MOVE_DOWN: "move_down",
    MOVE_LEFT: "move_left",
    MOVE_RIGHT: "move_right",
    GATHER: "gather",
    OFFER_FOOD_FOR_WOOD: "offer_food_for_wood",
    OFFER_WOOD_FOR_FOOD: "offer_wood_for_food",
}
N_ACTIONS = len(ACTION_NAMES)


@dataclass
class ResourceIslandConfig:
    """Configuration for the Resource Island spatial economy."""

    grid_size: int = 5
    n_agents: int = 2
    max_steps: int = 100
    initial_energy: float = 8.0
    max_energy: float = 16.0
    energy_per_food: float = 4.0
    stay_cost: float = 0.2
    move_cost: float = 1.0
    gather_cost: float = 0.7
    trade_cost: float = 0.5
    survival_reward: float = 1.0
    gather_reward: float = 0.7
    trade_reward: float = 0.3
    death_penalty: float = 5.0
    resource_capacity: int = 3
    initial_resource_units: int = 12
    resource_spawn_probability: float = 0.08
    vision_radius: int = 1
    trade_radius: int | None = None
    tabular_bins: int = N_ACTIONS
    start_positions: tuple[tuple[int, int], ...] | None = None
    initial_resources: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if self.n_agents < 1:
            raise ValueError("n_agents must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.tabular_bins != N_ACTIONS:
            raise ValueError("tabular_bins must equal the Resource Island action count for QLearningMind")
        if self.trade_radius is None:
            self.trade_radius = 2 * (self.grid_size - 1)
        if self.trade_radius < 1:
            raise ValueError("trade_radius must be positive")


def action_cost(config: ResourceIslandConfig, action: int) -> float:
    """Energy cost for one action."""
    if action in (MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT):
        return config.move_cost
    if action == GATHER:
        return config.gather_cost
    if action in (OFFER_FOOD_FOR_WOOD, OFFER_WOOD_FOR_FOOD):
        return config.trade_cost
    return config.stay_cost


@register_world("resource_island")
class ResourceIslandWorld(World):
    """Spatial gather/trade world with tabular observations for Q-learning."""

    def __init__(
        self,
        agents: list[Agent] | None = None,
        institution: Institution | None = None,
        seed: int | None = None,
        config: ResourceIslandConfig | None = None,
    ):
        super().__init__(agents=agents, institution=institution, seed=seed)
        self.config = config if config is not None else ResourceIslandConfig()
        self.institution = institution if institution is not None else NoInstitution()
        self.rng = np.random.default_rng(0 if seed is None else seed)
        self.step_idx = 0
        self.positions = np.zeros((self.config.n_agents, 2), dtype=int)
        self.energy = np.full(self.config.n_agents, self.config.initial_energy, dtype=float)
        self.inventory = np.zeros((self.config.n_agents, len(RESOURCE_TYPES)), dtype=int)
        self.gathered_totals = np.zeros((self.config.n_agents, len(RESOURCE_TYPES)), dtype=int)
        self.trade_counts = np.zeros(self.config.n_agents, dtype=int)
        self.trade_attempt_count = 0
        self.trade_blocked_count = 0
        self.trade_inventory_blocked_count = 0
        self.trade_institution_blocked_count = 0
        self.property_claim_count = 0
        self.property_violation_count = 0
        self.alive = np.ones(self.config.n_agents, dtype=bool)
        self.resources = np.zeros((self.config.grid_size, self.config.grid_size, len(RESOURCE_TYPES)), dtype=int)
        self.initial_resource_total = 0
        self.cumulative_resource_spawned = 0
        self.history: list[dict[str, Any]] = []
        self.reset()

    def reset(self) -> list[tuple[int, int, int]]:
        self.rng = np.random.default_rng(0 if self.seed is None else self.seed)
        self.step_idx = 0
        self.positions = self._initial_positions()
        self.energy = np.full(self.config.n_agents, self.config.initial_energy, dtype=float)
        self.inventory = np.zeros((self.config.n_agents, len(RESOURCE_TYPES)), dtype=int)
        self.gathered_totals = np.zeros((self.config.n_agents, len(RESOURCE_TYPES)), dtype=int)
        self.trade_counts = np.zeros(self.config.n_agents, dtype=int)
        self.trade_attempt_count = 0
        self.trade_blocked_count = 0
        self.trade_inventory_blocked_count = 0
        self.trade_institution_blocked_count = 0
        self.property_claim_count = 0
        self.property_violation_count = 0
        self.alive = np.ones(self.config.n_agents, dtype=bool)
        self.resources = self._initial_resources()
        self.initial_resource_total = int(np.sum(self.resources))
        self.cumulative_resource_spawned = 0
        self.history = []
        self.institution.reset()
        for agent in self.agents:
            agent.reset()
        return self.observations()

    def _initial_positions(self) -> np.ndarray:
        if self.config.start_positions is not None:
            positions = np.asarray(self.config.start_positions, dtype=int)
            if positions.shape != (self.config.n_agents, 2):
                raise ValueError("start_positions must have shape n_agents x 2")
            return np.clip(positions, 0, self.config.grid_size - 1)

        corners = [
            (0, 0),
            (self.config.grid_size - 1, self.config.grid_size - 1),
            (0, self.config.grid_size - 1),
            (self.config.grid_size - 1, 0),
        ]
        positions = list(corners[: min(len(corners), self.config.n_agents)])
        while len(positions) < self.config.n_agents:
            candidate = (
                int(self.rng.integers(self.config.grid_size)),
                int(self.rng.integers(self.config.grid_size)),
            )
            if candidate not in positions:
                positions.append(candidate)
        return np.asarray(positions, dtype=int)

    def _initial_resources(self) -> np.ndarray:
        return initial_resource_map(
            self.rng,
            grid_size=self.config.grid_size,
            n_resource_types=len(RESOURCE_TYPES),
            resource_capacity=self.config.resource_capacity,
            initial_resource_units=self.config.initial_resource_units,
            initial_resources=self.config.initial_resources,
        )

    def observations(self) -> list[tuple[int, int, int]]:
        return [self.discretize_obs(agent_id) for agent_id in range(self.config.n_agents)]

    def discretize_obs(self, agent_id: int) -> tuple[int, int, int]:
        """Return energy, local-resource, and inventory-imbalance bins for tabular Q-learning."""
        if not self.alive[agent_id]:
            return (0, 0, 0)
        bins = self.config.tabular_bins
        energy_share = np.clip(self.energy[agent_id] / self.config.max_energy, 0.0, 1.0)
        energy_bin = int(round(energy_share * (bins - 1)))
        local_resource_count = self.local_resource_count(agent_id)
        local_resource_bin = int(min(bins - 1, local_resource_count))
        imbalance_bin = self.inventory_imbalance_bin(agent_id)
        return (energy_bin, local_resource_bin, imbalance_bin)

    def inventory_imbalance_bin(self, agent_id: int) -> int:
        """Encode food-surplus, balanced, or wood-surplus inventory into Q-table coordinates."""
        food = int(self.inventory[agent_id, FOOD])
        wood = int(self.inventory[agent_id, WOOD])
        midpoint = self.config.tabular_bins // 2
        if food > wood:
            return 0
        if wood > food:
            return self.config.tabular_bins - 1
        return midpoint

    def local_resource_count(self, agent_id: int) -> int:
        return count_local_resources(
            self.resources,
            tuple(int(value) for value in self.positions[agent_id]),
            self.config.vision_radius,
        )

    def step(self, actions: list[Any]) -> tuple[list[tuple[int, int, int]], np.ndarray, bool, dict[str, Any]]:
        if len(actions) != self.config.n_agents:
            raise ValueError(f"ResourceIslandWorld expects {self.config.n_agents} actions")

        clean_actions = [self._clean_action(action, agent_id) for agent_id, action in enumerate(actions)]
        rewards = np.zeros(self.config.n_agents, dtype=float)
        gathered_this_step = np.zeros((self.config.n_agents, len(RESOURCE_TYPES)), dtype=int)
        trades_this_step = np.zeros(self.config.n_agents, dtype=int)
        diagnostic_counts = {
            "trade_attempts_step": 0,
            "trade_blocked_step": 0,
            "trade_inventory_blocked_step": 0,
            "trade_institution_blocked_step": 0,
            "property_claims_step": 0,
            "property_violations_step": 0,
        }

        self._apply_movement(clean_actions)

        for agent_id, action in enumerate(clean_actions):
            if not self.alive[agent_id]:
                continue
            cost = action_cost(self.config, action)
            self.energy[agent_id] -= cost
            rewards[agent_id] += self.config.survival_reward - cost

        self._apply_gathering(clean_actions, rewards, gathered_this_step, diagnostic_counts)
        self._apply_trading(clean_actions, rewards, trades_this_step, diagnostic_counts)
        self._apply_energy_and_death(rewards)

        reward_state = {
            "phase": "post_rewards",
            "rewards": rewards,
            "alive": self.alive.copy(),
            "inventory": self.inventory.copy(),
            "gathered_totals": self.gathered_totals.copy(),
            "trade_counts": self.trade_counts.copy(),
            "step_idx": self.step_idx,
        }
        reward_state = self.institution.apply(reward_state)
        rewards = np.asarray(reward_state.get("rewards", rewards), dtype=float)

        self._regenerate_resources()
        self.step_idx += 1
        done = bool(self.step_idx >= self.config.max_steps or not np.any(self.alive))
        info = self._info(clean_actions, rewards, gathered_this_step, trades_this_step, diagnostic_counts, done)
        self.history.append(info)
        return self.observations(), rewards, done, info

    def _clean_action(self, action: Any, agent_id: int) -> int:
        if not self.alive[agent_id]:
            return STAY
        action_int = int(action)
        if action_int not in ACTION_NAMES:
            raise ValueError(f"invalid Resource Island action {action_int}")
        return action_int

    def _apply_movement(self, actions: list[int]) -> None:
        proposals = self.positions.copy()
        for agent_id, action in enumerate(actions):
            if not self.alive[agent_id]:
                continue
            row, col = (int(value) for value in self.positions[agent_id])
            if action == MOVE_UP:
                row -= 1
            elif action == MOVE_DOWN:
                row += 1
            elif action == MOVE_LEFT:
                col -= 1
            elif action == MOVE_RIGHT:
                col += 1
            proposals[agent_id] = np.array(
                [
                    int(np.clip(row, 0, self.config.grid_size - 1)),
                    int(np.clip(col, 0, self.config.grid_size - 1)),
                ],
                dtype=int,
            )

        destinations: dict[tuple[int, int], list[int]] = {}
        for agent_id, proposal in enumerate(proposals):
            if not self.alive[agent_id]:
                continue
            destinations.setdefault((int(proposal[0]), int(proposal[1])), []).append(agent_id)

        for destination, agent_ids in destinations.items():
            if len(agent_ids) == 1:
                self.positions[agent_ids[0]] = np.asarray(destination, dtype=int)

    def _apply_gathering(
        self,
        actions: list[int],
        rewards: np.ndarray,
        gathered_this_step: np.ndarray,
        diagnostic_counts: dict[str, int],
    ) -> None:
        for agent_id, action in enumerate(actions):
            if action != GATHER or not self.alive[agent_id]:
                continue
            row, col = (int(value) for value in self.positions[agent_id])
            resource_type = self._resource_to_gather(row, col)
            if resource_type is None:
                continue
            hook_state = {
                "phase": "pre_gather",
                "agent_id": agent_id,
                "position": (row, col),
                "resource_type": resource_type,
                "allowed": True,
                "penalty": 0.0,
            }
            hook_state = self.institution.apply(hook_state)
            rewards[agent_id] -= float(hook_state.get("penalty", 0.0))
            if not bool(hook_state.get("allowed", True)):
                if int(hook_state.get("property_violation", 0)) > 0:
                    self.property_violation_count += 1
                    diagnostic_counts["property_violations_step"] += 1
                continue

            self.resources[row, col, resource_type] -= 1
            self.inventory[agent_id, resource_type] += 1
            self.gathered_totals[agent_id, resource_type] += 1
            gathered_this_step[agent_id, resource_type] += 1
            rewards[agent_id] += self.config.gather_reward
            post_gather = self.institution.apply(
                {
                    "phase": "post_gather",
                    "agent_id": agent_id,
                    "position": (row, col),
                    "resource_type": resource_type,
                    "gathered_amount": 1,
                }
            )
            if int(post_gather.get("property_claim_created", 0)) > 0:
                self.property_claim_count += 1
                diagnostic_counts["property_claims_step"] += 1

    def _resource_to_gather(self, row: int, col: int) -> int | None:
        for resource_type in (FOOD, WOOD):
            if self.resources[row, col, resource_type] > 0:
                return resource_type
        return None

    def _apply_trading(
        self,
        actions: list[int],
        rewards: np.ndarray,
        trades_this_step: np.ndarray,
        diagnostic_counts: dict[str, int],
    ) -> None:
        used: set[int] = set()
        for left in range(self.config.n_agents):
            if left in used or not self.alive[left]:
                continue
            for right in range(left + 1, self.config.n_agents):
                if right in used or not self.alive[right]:
                    continue
                if not within_trade_radius(
                    tuple(self.positions[left]),
                    tuple(self.positions[right]),
                    int(self.config.trade_radius),
                ):
                    continue
                if actions[left] not in (OFFER_FOOD_FOR_WOOD, OFFER_WOOD_FOR_FOOD) and actions[right] not in (
                    OFFER_FOOD_FOR_WOOD,
                    OFFER_WOOD_FOR_FOOD,
                ):
                    continue
                self.trade_attempt_count += 1
                diagnostic_counts["trade_attempts_step"] += 1
                trade = self._trade_offer(actions[left], actions[right], left, right)
                if trade is None:
                    self.trade_blocked_count += 1
                    self.trade_inventory_blocked_count += 1
                    diagnostic_counts["trade_blocked_step"] += 1
                    diagnostic_counts["trade_inventory_blocked_step"] += 1
                    continue
                trade = self.institution.apply(trade)
                if not bool(trade.get("allowed", True)):
                    self.trade_blocked_count += 1
                    self.trade_institution_blocked_count += 1
                    diagnostic_counts["trade_blocked_step"] += 1
                    diagnostic_counts["trade_institution_blocked_step"] += 1
                    continue
                giver_food = int(trade["food_giver"])
                giver_wood = int(trade["wood_giver"])
                if self.inventory[giver_food, FOOD] < 1 or self.inventory[giver_wood, WOOD] < 1:
                    self.trade_blocked_count += 1
                    self.trade_inventory_blocked_count += 1
                    diagnostic_counts["trade_blocked_step"] += 1
                    diagnostic_counts["trade_inventory_blocked_step"] += 1
                    continue
                self.inventory[giver_food, FOOD] -= 1
                self.inventory[giver_food, WOOD] += 1
                self.inventory[giver_wood, WOOD] -= 1
                self.inventory[giver_wood, FOOD] += 1
                self.trade_counts[[giver_food, giver_wood]] += 1
                trades_this_step[[giver_food, giver_wood]] += 1
                rewards[[giver_food, giver_wood]] += self.config.trade_reward
                self.institution.apply(
                    {
                        "phase": "post_trade",
                        "participants": (giver_food, giver_wood),
                        "food_units": 1,
                        "wood_units": 1,
                    }
                )
                used.update({giver_food, giver_wood})
                break

    def _trade_offer(self, left_action: int, right_action: int, left: int, right: int) -> dict[str, Any] | None:
        candidates: list[tuple[int, int]] = []
        if left_action == OFFER_FOOD_FOR_WOOD:
            candidates.append((left, right))
        elif left_action == OFFER_WOOD_FOR_FOOD:
            candidates.append((right, left))
        if right_action == OFFER_FOOD_FOR_WOOD:
            candidates.append((right, left))
        elif right_action == OFFER_WOOD_FOR_FOOD:
            candidates.append((left, right))

        for food_giver, wood_giver in candidates:
            if self.inventory[food_giver, FOOD] >= 1 and self.inventory[wood_giver, WOOD] >= 1:
                return {
                    "phase": "pre_trade",
                    "participants": (food_giver, wood_giver),
                    "food_giver": food_giver,
                    "wood_giver": wood_giver,
                    "food_units": 1,
                    "wood_units": 1,
                    "allowed": True,
                }
        return None

    def _apply_energy_and_death(self, rewards: np.ndarray) -> None:
        for agent_id in range(self.config.n_agents):
            if not self.alive[agent_id]:
                continue
            if self.energy[agent_id] <= 0.0 and self.inventory[agent_id, FOOD] > 0:
                self.inventory[agent_id, FOOD] -= 1
                self.energy[agent_id] = min(self.config.max_energy, self.energy[agent_id] + self.config.energy_per_food)
            if self.energy[agent_id] <= 0.0:
                self.alive[agent_id] = False
                rewards[agent_id] -= self.config.death_penalty

    def _regenerate_resources(self) -> None:
        spawned = regenerate_resource(
            self.resources,
            self.rng,
            spawn_probability=self.config.resource_spawn_probability,
            resource_capacity=self.config.resource_capacity,
        )
        if spawned:
            self.cumulative_resource_spawned += 1

    def _info(
        self,
        actions: list[int],
        rewards: np.ndarray,
        gathered_this_step: np.ndarray,
        trades_this_step: np.ndarray,
        diagnostic_counts: dict[str, int],
        done: bool,
    ) -> dict[str, Any]:
        alive_count = int(np.sum(self.alive))
        total_inventory = np.sum(self.inventory, axis=0)
        total_resources = np.sum(self.resources, axis=(0, 1))
        current_resource_stock = float(np.sum(total_resources))
        total_introduced = float(self.initial_resource_total + self.cumulative_resource_spawned)
        holdings = np.sum(self.inventory, axis=1) + np.maximum(self.energy, 0.0) / self.config.energy_per_food
        pairwise_distance = self.mean_pairwise_distance()
        alive_entries = [record["alive"] for record in self.history]
        alive_entries.append(tuple(bool(value) for value in self.alive))
        return {
            "step": float(self.step_idx),
            "done": bool(done),
            "actions": tuple(actions),
            "alive": tuple(bool(value) for value in self.alive),
            "alive_count": float(alive_count),
            "survival_rate": survival_rate(alive_entries),
            "mean_energy": float(np.mean(self.energy)),
            "reward_total": float(np.sum(rewards)),
            "welfare": float(np.sum(rewards)),
            "food_inventory": float(total_inventory[FOOD]),
            "wood_inventory": float(total_inventory[WOOD]),
            "mean_pairwise_distance": pairwise_distance,
            "contact_rate": float(pairwise_distance <= float(self.config.trade_radius)),
            "food_on_map": float(total_resources[FOOD]),
            "wood_on_map": float(total_resources[WOOD]),
            "resource_stock": current_resource_stock,
            "resource_units_introduced": total_introduced,
            "resource_sustainability": resource_sustainability(current_resource_stock, total_introduced),
            "inequality_over_time": gini(holdings),
            "gathered_food": float(np.sum(self.gathered_totals[:, FOOD])),
            "gathered_wood": float(np.sum(self.gathered_totals[:, WOOD])),
            "gathered_food_step": float(np.sum(gathered_this_step[:, FOOD])),
            "gathered_wood_step": float(np.sum(gathered_this_step[:, WOOD])),
            "trade_count": float(np.sum(self.trade_counts) / 2.0),
            "trade_count_step": float(np.sum(trades_this_step) / 2.0),
            "trade_attempt_count": float(self.trade_attempt_count),
            "trade_attempt_count_step": float(diagnostic_counts["trade_attempts_step"]),
            "trade_blocked_count": float(self.trade_blocked_count),
            "trade_blocked_count_step": float(diagnostic_counts["trade_blocked_step"]),
            "trade_inventory_blocked_count": float(self.trade_inventory_blocked_count),
            "trade_inventory_blocked_count_step": float(diagnostic_counts["trade_inventory_blocked_step"]),
            "trade_institution_blocked_count": float(self.trade_institution_blocked_count),
            "trade_institution_blocked_count_step": float(diagnostic_counts["trade_institution_blocked_step"]),
            "property_claims": float(self.property_claim_count),
            "property_claims_step": float(diagnostic_counts["property_claims_step"]),
            "property_violations": float(self.property_violation_count),
            "property_violations_step": float(diagnostic_counts["property_violations_step"]),
            "specialization_index": specialization_index(self.gathered_totals),
        }

    def mean_pairwise_distance(self) -> float:
        """Mean Manhattan distance over alive agent pairs; NaN when fewer than two agents are alive."""
        alive_ids = [idx for idx, is_alive in enumerate(self.alive) if bool(is_alive)]
        distances: list[float] = []
        for left_index, left in enumerate(alive_ids):
            for right in alive_ids[left_index + 1 :]:
                distances.append(
                    float(
                        manhattan_distance(
                            tuple(int(value) for value in self.positions[left]),
                            tuple(int(value) for value in self.positions[right]),
                        )
                    )
                )
        if not distances:
            return float("nan")
        return float(np.mean(distances))

    def get_metrics(self) -> dict[str, Any]:
        if not self.history:
            return {}
        numeric_keys = [
            key
            for key, value in self.history[0].items()
            if isinstance(value, (float, int, np.floating, np.integer, bool))
        ]
        return {
            key: finite_mean(record[key] for record in self.history)
            for key in numeric_keys
        }

    def render_state(self) -> dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "positions": self.positions.tolist(),
            "energy": self.energy.tolist(),
            "inventory": self.inventory.tolist(),
            "gathered_totals": self.gathered_totals.tolist(),
            "trade_counts": self.trade_counts.tolist(),
            "trade_attempt_count": self.trade_attempt_count,
            "trade_blocked_count": self.trade_blocked_count,
            "trade_inventory_blocked_count": self.trade_inventory_blocked_count,
            "trade_institution_blocked_count": self.trade_institution_blocked_count,
            "property_claim_count": self.property_claim_count,
            "property_violation_count": self.property_violation_count,
            "alive": self.alive.tolist(),
            "resources": self.resources.tolist(),
            "last_info": self.history[-1] if self.history else None,
        }
