import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.metrics import (
    resource_sustainability,
    robustness_under_shock,
    specialization_index,
    stability,
    survival_rate,
)
from core.registry import build_experiment
from institutions.resource_island import PropertyRights, Redistribution, ReputationSystem, TradePriceControls
from minds.q_learning import QLearningMind
from run_resource_island_smoke import parse_args as parse_smoke_args
from run_resource_island_smoke import run as run_resource_smoke
from worlds.resource_island.benchmarks import (
    efficient_gather_upper_bound,
    greedy_full_information_gather_plan,
)
from worlds.resource_island.env import (
    FOOD,
    GATHER,
    MOVE_LEFT,
    MOVE_RIGHT,
    N_ACTIONS,
    OFFER_FOOD_FOR_WOOD,
    OFFER_WOOD_FOR_FOOD,
    STAY,
    WOOD,
    ResourceIslandConfig,
    ResourceIslandWorld,
)
from worlds.resource_island.training import train_resource_island


def zero_resources(grid_size: int = 3) -> np.ndarray:
    return np.zeros((grid_size, grid_size, 2), dtype=int)


class ResourceIslandMetricTests(unittest.TestCase):
    def test_resource_metrics_have_expected_values(self):
        self.assertAlmostEqual(survival_rate([[True, False], [True, True]]), 0.75)
        self.assertAlmostEqual(specialization_index([[3.0, 0.0], [1.0, 1.0]]), 0.5)
        self.assertEqual(stability([2.0, 2.0, 2.0]), 1.0)
        self.assertAlmostEqual(robustness_under_shock(10.0, 6.0), 0.6)
        self.assertTrue(math.isnan(robustness_under_shock(0.0, 6.0)))
        self.assertAlmostEqual(resource_sustainability(3.0, 12.0), 0.25)
        self.assertTrue(math.isnan(resource_sustainability(3.0, 0.0)))


class ResourceIslandBenchmarkTests(unittest.TestCase):
    def test_efficient_upper_bound_caps_by_agents_steps_and_resources(self):
        resources = zero_resources()
        resources[0, 0, FOOD] = 5
        resources[1, 1, WOOD] = 2
        self.assertEqual(efficient_gather_upper_bound(resources, n_agents=2, steps=2), 4)
        self.assertEqual(efficient_gather_upper_bound(resources, n_agents=4, steps=3), 7)

    def test_greedy_full_information_plan_assigns_reachable_resources(self):
        resources = zero_resources()
        resources[0, 1, FOOD] = 1
        resources[2, 2, WOOD] = 1
        plan = greedy_full_information_gather_plan([(0, 0), (2, 1)], resources, steps=2)
        self.assertEqual(plan.estimated_gathered, 2)
        self.assertEqual(len(plan.assignments), 2)


class ResourceIslandWorldTests(unittest.TestCase):
    def test_reset_returns_tabular_observations_for_each_agent(self):
        cfg = ResourceIslandConfig(grid_size=3, n_agents=2, initial_resources=zero_resources())
        world = ResourceIslandWorld(config=cfg, seed=3)
        obs = world.reset()
        self.assertEqual(len(obs), 2)
        self.assertTrue(all(isinstance(item, tuple) and len(item) == 3 for item in obs))
        self.assertTrue(all(0 <= value < N_ACTIONS for item in obs for value in item))

    def test_tabular_observation_includes_inventory_imbalance(self):
        cfg = ResourceIslandConfig(grid_size=3, n_agents=1, initial_resources=zero_resources())
        world = ResourceIslandWorld(config=cfg, seed=3)
        balanced = world.discretize_obs(0)[2]
        world.inventory[0, FOOD] = 2
        self.assertEqual(world.discretize_obs(0)[2], 0)
        world.inventory[0, FOOD] = 0
        world.inventory[0, WOOD] = 2
        self.assertEqual(world.discretize_obs(0)[2], N_ACTIONS - 1)
        world.inventory[0, WOOD] = 0
        self.assertEqual(world.discretize_obs(0)[2], balanced)

    def test_simultaneous_move_collision_keeps_agents_in_place(self):
        cfg = ResourceIslandConfig(
            grid_size=3,
            n_agents=2,
            start_positions=((1, 0), (1, 2)),
            initial_resources=zero_resources(),
            resource_spawn_probability=0.0,
        )
        world = ResourceIslandWorld(config=cfg, seed=1)
        world.step([MOVE_RIGHT, MOVE_LEFT])
        self.assertEqual(world.positions.tolist(), [[1, 0], [1, 2]])

    def test_gather_increases_inventory_and_depletes_cell(self):
        resources = zero_resources()
        resources[0, 0, FOOD] = 1
        cfg = ResourceIslandConfig(
            grid_size=3,
            n_agents=1,
            start_positions=((0, 0),),
            initial_resources=resources,
            resource_spawn_probability=0.0,
        )
        world = ResourceIslandWorld(config=cfg, seed=1)
        _, rewards, _, info = world.step([GATHER])
        self.assertEqual(world.inventory[0, FOOD], 1)
        self.assertEqual(world.resources[0, 0, FOOD], 0)
        self.assertGreater(rewards[0], 0.0)
        self.assertEqual(info["gathered_food_step"], 1.0)

    def test_lone_agent_can_survive_by_gathering_food(self):
        resources = zero_resources()
        resources[0, 0, FOOD] = 2
        cfg = ResourceIslandConfig(
            grid_size=3,
            n_agents=1,
            start_positions=((0, 0),),
            initial_energy=0.4,
            energy_per_food=3.0,
            gather_cost=1.0,
            initial_resources=resources,
            resource_spawn_probability=0.0,
        )
        world = ResourceIslandWorld(config=cfg, seed=1)
        world.step([GATHER])
        self.assertTrue(world.alive[0])
        self.assertGreater(world.energy[0], 0.0)

    def test_complementary_adjacent_trade_swaps_food_and_wood(self):
        cfg = ResourceIslandConfig(
            grid_size=3,
            n_agents=2,
            start_positions=((1, 1), (1, 2)),
            initial_resources=zero_resources(),
            resource_spawn_probability=0.0,
        )
        world = ResourceIslandWorld(config=cfg, seed=1)
        world.inventory[0, FOOD] = 1
        world.inventory[1, WOOD] = 1
        _, _, _, info = world.step([OFFER_FOOD_FOR_WOOD, OFFER_WOOD_FOR_FOOD])
        self.assertEqual(world.inventory.tolist(), [[0, 1], [1, 0]])
        self.assertEqual(info["trade_count_step"], 1.0)
        self.assertEqual(info["trade_attempt_count_step"], 1.0)
        self.assertEqual(info["trade_blocked_count_step"], 0.0)

    def test_default_trade_radius_allows_non_adjacent_market_trade(self):
        cfg = ResourceIslandConfig(
            grid_size=5,
            n_agents=2,
            start_positions=((0, 0), (4, 4)),
            initial_resources=zero_resources(grid_size=5),
            resource_spawn_probability=0.0,
        )
        world = ResourceIslandWorld(config=cfg, seed=1)
        world.inventory[0, FOOD] = 1
        world.inventory[1, WOOD] = 1
        _, _, _, info = world.step([OFFER_FOOD_FOR_WOOD, OFFER_WOOD_FOR_FOOD])
        self.assertEqual(info["trade_count_step"], 1.0)
        self.assertEqual(info["contact_rate"], 1.0)
        self.assertEqual(info["mean_pairwise_distance"], 8.0)

    def test_unilateral_valid_trade_offer_clears_with_passive_counterparty(self):
        cfg = ResourceIslandConfig(
            grid_size=3,
            n_agents=2,
            start_positions=((1, 1), (1, 2)),
            initial_resources=zero_resources(),
            resource_spawn_probability=0.0,
        )
        world = ResourceIslandWorld(config=cfg, seed=1)
        world.inventory[0, FOOD] = 1
        world.inventory[1, WOOD] = 1
        _, _, _, info = world.step([OFFER_FOOD_FOR_WOOD, STAY])
        self.assertEqual(world.inventory.tolist(), [[0, 1], [1, 0]])
        self.assertEqual(info["trade_count_step"], 1.0)
        self.assertEqual(info["trade_inventory_blocked_count_step"], 0.0)

    def test_strict_trade_radius_blocks_non_adjacent_trade_attempts(self):
        cfg = ResourceIslandConfig(
            grid_size=5,
            n_agents=2,
            start_positions=((0, 0), (4, 4)),
            initial_resources=zero_resources(grid_size=5),
            resource_spawn_probability=0.0,
            trade_radius=1,
        )
        world = ResourceIslandWorld(config=cfg, seed=1)
        world.inventory[0, FOOD] = 1
        world.inventory[1, WOOD] = 1
        _, _, _, info = world.step([OFFER_FOOD_FOR_WOOD, OFFER_WOOD_FOR_FOOD])
        self.assertEqual(info["trade_count_step"], 0.0)
        self.assertEqual(info["trade_attempt_count_step"], 0.0)
        self.assertEqual(info["contact_rate"], 0.0)

    def test_get_metrics_returns_numeric_summary_after_step(self):
        cfg = ResourceIslandConfig(grid_size=3, n_agents=1, initial_resources=zero_resources())
        world = ResourceIslandWorld(config=cfg, seed=1)
        world.step([STAY])
        metrics = world.get_metrics()
        self.assertIn("survival_rate", metrics)
        self.assertIn("welfare", metrics)
        self.assertIn("inequality_over_time", metrics)
        self.assertIn("resource_sustainability", metrics)


class ResourceIslandInstitutionTests(unittest.TestCase):
    def test_property_rights_blocks_unowned_gathering_from_claimed_cell(self):
        resources = zero_resources()
        resources[1, 1, FOOD] = 2
        cfg = ResourceIslandConfig(
            grid_size=3,
            n_agents=2,
            start_positions=((1, 1), (1, 1)),
            initial_resources=resources,
            resource_spawn_probability=0.0,
        )
        world = ResourceIslandWorld(config=cfg, institution=PropertyRights(violation_penalty=2.0), seed=1)
        _, _, _, first_info = world.step([GATHER, STAY])
        _, rewards, _, second_info = world.step([STAY, GATHER])
        self.assertEqual(first_info["property_claims_step"], 1.0)
        self.assertEqual(second_info["property_violations_step"], 1.0)
        self.assertEqual(world.inventory[1, FOOD], 0)
        self.assertLess(rewards[1], rewards[0])

    def test_redistribution_taxes_positive_rewards(self):
        institution = Redistribution(tax_rate=0.5)
        state = {
            "phase": "post_rewards",
            "rewards": np.array([10.0, 0.0]),
            "alive": np.array([True, True]),
        }
        out = institution.apply(state)
        self.assertTrue(np.allclose(out["rewards"], np.array([7.5, 2.5])))

    def test_trade_price_controls_blocks_unequal_exchange(self):
        institution = TradePriceControls(max_exchange_ratio=1.5)
        out = institution.apply({"phase": "pre_trade", "food_units": 2.0, "wood_units": 1.0, "allowed": True})
        self.assertFalse(out["allowed"])

    def test_reputation_system_adds_trade_bonus(self):
        institution = ReputationSystem(trade_reputation_gain=2.0, reward_bonus=0.5)
        institution.apply({"phase": "post_trade", "participants": (0, 1)})
        out = institution.apply(
            {
                "phase": "post_rewards",
                "rewards": np.array([1.0, 1.0]),
                "alive": np.array([True, True]),
            }
        )
        self.assertTrue(np.allclose(out["rewards"], np.array([2.0, 2.0])))


class ResourceIslandIntegrationTests(unittest.TestCase):
    def test_registry_builds_resource_island_with_unchanged_q_learning_mind(self):
        cfg = ResourceIslandConfig(grid_size=3, n_agents=2, initial_resources=zero_resources())
        world = build_experiment(
            {
                "world": "resource_island",
                "institution": "none",
                "seed": 4,
                "world_params": {"config": cfg},
                "agents": [
                    {
                        "mind": "q_learning",
                        "params": {"n_prices": N_ACTIONS, "seed": 4, "state_shape": (N_ACTIONS, N_ACTIONS, N_ACTIONS)},
                    },
                    {
                        "mind": "q_learning",
                        "params": {"n_prices": N_ACTIONS, "seed": 5, "state_shape": (N_ACTIONS, N_ACTIONS, N_ACTIONS)},
                    },
                ],
            }
        )
        self.assertIsInstance(world.agents[0], QLearningMind)
        obs = world.observations()
        next_obs, rewards, done, _ = world.step([STAY, STAY])
        world.agents[0].update(obs[0], STAY, float(rewards[0]), next_obs[0], done)
        self.assertNotEqual(float(np.sum(world.agents[0].q_values)), 0.0)

    def test_training_loop_runs_q_learning_without_mind_changes(self):
        cfg = ResourceIslandConfig(
            grid_size=3,
            n_agents=2,
            max_steps=20,
            initial_resource_units=4,
            resource_spawn_probability=0.0,
        )
        result = train_resource_island(steps=10, seed=2, config=cfg)
        self.assertEqual(len(result.records), 10)
        self.assertTrue(all("survival_rate" in record for record in result.records))
        self.assertTrue(all("inequality_over_time" in record for record in result.records))
        self.assertTrue(all("resource_sustainability" in record for record in result.records))
        self.assertTrue(all("contact_rate" in record for record in result.records))
        self.assertTrue(all("trade_attempt_count" in record for record in result.records))
        self.assertTrue(all("trade_inventory_blocked_count" in record for record in result.records))
        self.assertTrue(all("trade_institution_blocked_count" in record for record in result.records))
        self.assertTrue(all("property_claims" in record for record in result.records))

    def test_smoke_runner_writes_summary_csvs_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parse_smoke_args(
                [
                    "--steps",
                    "8",
                    "--n-seeds",
                    "2",
                    "--final-window",
                    "4",
                    "--institutions",
                    "none",
                    "property_rights",
                    "--save-dir",
                    tmpdir,
                ]
            )
            outputs = run_resource_smoke(args)
            for path in outputs.values():
                self.assertTrue(Path(path).exists())

            with outputs["summary_by_seed"].open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertIn("inequality_over_time", rows[0])
            self.assertIn("contact_rate", rows[0])
            self.assertIn("trade_attempt_count", rows[0])
            self.assertIn("trade_inventory_blocked_count", rows[0])
            self.assertIn("property_claims", rows[0])
            self.assertIn("resource_sustainability", rows[0])

            manifest = json.loads(outputs["manifest"].read_text())
            self.assertEqual(manifest["world"], "resource_island")
            self.assertEqual(manifest["mind"], "q_learning")
            self.assertIn("inequality_over_time", manifest["summary_metrics"])
            self.assertIn("resource_sustainability", manifest["summary_metrics"])


if __name__ == "__main__":
    unittest.main()
