import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_combined_table import combined_rows, write_csv as write_combined_csv
from arena_v0 import DuopolyMarket, MarketConfig, QAgent
from core.logger import finalize_run, log_step, new_run_manifest
from core.metrics import collusion_index, exploitability, gini, profit_collusion_index, victim_loss, welfare_damage
from core.registry import build_experiment
from institutions.anti_collusion import AntiCollusion
from institutions.demand_shock import DemandShock
from institutions.none import NoInstitution
from institutions.price_cap import PriceCap
from institutions.random_audit import RandomAudit
from institutions.tax_high_price import TaxHighPrice
from minds.heuristic_mind import HeuristicPricingMind
from minds.q_learning import QLearningMind
from minds.random_mind import RandomMind
from worlds.pricing_arena.env import PricingArenaWorld


class CoreMetricsTests(unittest.TestCase):
    def test_core_exploitability_formulas_match_current_definitions(self):
        self.assertEqual(exploitability(12.0, 8.0), 4.0)
        self.assertEqual(victim_loss(10.0, 3.0), 7.0)
        self.assertEqual(welfare_damage(100.0, 80.0), 20.0)

    def test_collusion_index_matches_existing_scaled_price_definition(self):
        self.assertAlmostEqual(collusion_index(5.25, 2.5, 8.0), 0.5)
        self.assertEqual(collusion_index(20.0, 2.5, 8.0), 1.0)
        self.assertTrue(math.isnan(collusion_index(5.0, None, 8.0)))

    def test_profit_collusion_index_matches_calvano_style_formula(self):
        self.assertAlmostEqual(profit_collusion_index(15.0, 10.0, 20.0), 0.5)
        self.assertEqual(profit_collusion_index(40.0, 10.0, 20.0), 1.0)
        self.assertTrue(math.isnan(profit_collusion_index(15.0, None, 20.0)))

    def test_gini_handles_equal_zero_and_unequal_values(self):
        self.assertEqual(gini([0.0, 0.0]), 0.0)
        self.assertEqual(gini([3.0, 3.0, 3.0]), 0.0)
        self.assertGreater(gini([0.0, 0.0, 10.0]), 0.0)


class MindWrapperTests(unittest.TestCase):
    def test_q_learning_mind_preserves_q_agent_action_and_update_logic(self):
        mind = QLearningMind(n_prices=4, seed=11)
        original = QAgent(n_prices=4, seed=11)
        state = (1, 2)
        next_state = (2, 3)

        self.assertEqual(mind.act(state, epsilon=0.0), original.act(state, epsilon=0.0))
        mind.update(state, 3, 7.0, next_state, done=False)
        original.update(state, 3, 7.0, next_state)
        self.assertTrue(np.array_equal(mind.q_values, original.q_values))

    def test_q_learning_mind_supports_explicit_discrete_state_shape(self):
        mind = QLearningMind(n_prices=4, seed=11, state_shape=(4, 4, 4))
        state = (1, 2, 3)
        next_state = (2, 3, 1)
        mind.update(state, 2, 5.0, next_state, done=False)
        self.assertEqual(mind.q_values.shape, (4, 4, 4, 4))
        self.assertGreater(mind.q_values[state + (2,)], 0.0)

    def test_random_mind_outputs_valid_actions_and_resets_seed(self):
        mind = RandomMind(n_actions=5, seed=123)
        actions = [mind.act(None) for _ in range(8)]
        self.assertTrue(all(0 <= action < 5 for action in actions))

        mind.reset()
        replay = [mind.act(None) for _ in range(8)]
        self.assertEqual(actions, replay)

    def test_heuristic_pricing_mind_undercuts_opponent_action(self):
        mind = HeuristicPricingMind(n_actions=10, undercut=1)
        self.assertEqual(mind.act((5, 7)), 6)
        self.assertEqual(mind.act((5, 0)), 0)


class InstitutionWrapperTests(unittest.TestCase):
    def test_price_cap_clamps_prices(self):
        state = {"prices": np.array([9.0, 4.0])}
        out = PriceCap(price_cap=5.5).apply(state)
        self.assertTrue(np.array_equal(out["prices"], np.array([5.5, 4.0])))

    def test_tax_high_price_matches_existing_penalty_formula(self):
        state = {
            "prices": np.array([6.0, 4.0]),
            "quantities": np.array([10.0, 10.0]),
            "rewards": np.array([50.0, 30.0]),
            "penalties": np.zeros(2),
        }
        out = TaxHighPrice(tax_threshold=5.5, tax_rate=0.3).apply(state)
        self.assertAlmostEqual(out["penalties"][0], 1.5)
        self.assertAlmostEqual(out["rewards"][0], 48.5)
        self.assertAlmostEqual(out["penalties"][1], 0.0)

    def test_random_audit_applies_penalty_when_probability_is_one(self):
        state = {
            "prices": np.array([6.0, 6.0]),
            "rewards": np.array([50.0, 50.0]),
            "penalties": np.zeros(2),
            "rng": np.random.default_rng(1),
        }
        out = RandomAudit(audit_probability=1.0, audit_threshold=5.5, audit_penalty=35.0).apply(state)
        self.assertEqual(out["audit_hit"], 1.0)
        self.assertTrue(np.array_equal(out["penalties"], np.array([35.0, 35.0])))

    def test_anti_collusion_penalizes_close_high_prices(self):
        state = {
            "prices": np.array([6.0, 6.5]),
            "rewards": np.array([50.0, 50.0]),
            "penalties": np.zeros(2),
        }
        out = AntiCollusion(collusion_threshold=5.5, collusion_window=0.75, collusion_penalty=20.0).apply(state)
        self.assertTrue(np.array_equal(out["penalties"], np.array([20.0, 20.0])))

    def test_demand_shock_leaves_market_size_when_probability_zero(self):
        state = {"market_size": 100.0, "rng": np.random.default_rng(1)}
        out = DemandShock(shock_probability=0.0).apply(state)
        self.assertEqual(out["market_size"], 100.0)

    def test_no_institution_is_noop_copy(self):
        state = {"prices": np.array([1.0, 2.0])}
        out = NoInstitution().apply(state)
        self.assertIsNot(out, state)
        self.assertTrue(np.array_equal(out["prices"], state["prices"]))


class PricingArenaWrapperTests(unittest.TestCase):
    def test_pricing_arena_world_step_matches_existing_market_step(self):
        cfg = MarketConfig(mechanism="none")
        world = PricingArenaWorld(config=cfg, seed=9)
        market = DuopolyMarket(cfg, seed=9)

        world_next, world_rewards, world_done, world_info = world.step([3, 4])
        market_next, market_rewards, market_info = market.step((3, 4))

        self.assertEqual(world_next, market_next)
        self.assertFalse(world_done)
        self.assertTrue(np.array_equal(world_rewards, market_rewards))
        self.assertEqual(world_info["avg_price"], market_info["avg_price"])
        self.assertIn("collusion_index", world_info)

    def test_registry_builds_pricing_world_with_registered_components(self):
        world = build_experiment(
            {
                "world": "pricing_arena",
                "mind": "random",
                "institution": "price_cap",
                "seed": 5,
                "mind_params": {"n_actions": 19, "seed": 5},
                "institution_params": {"price_cap": 5.5},
            }
        )

        self.assertIsInstance(world, PricingArenaWorld)
        self.assertEqual(len(world.agents), 2)
        _, _, _, info = world.step([18, 18])
        self.assertEqual(info["p1"], 5.5)


class CoreLoggerTests(unittest.TestCase):
    def test_logger_writes_manifest_steps_and_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = new_run_manifest({"run_id": "test_run", "world": "pricing_arena"}, tmpdir)
            log_step(run_id, 0, {"welfare": 1.25}, tmpdir)
            finalize_run(run_id, {"welfare": 1.25}, tmpdir)

            output_dir = Path(tmpdir)
            manifest = json.loads((output_dir / "experiment_manifest.json").read_text())
            self.assertEqual(manifest["run_id"], "test_run")

            with (output_dir / "test_run_steps.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["welfare"], "1.25")

            summary = json.loads((output_dir / "test_run_summary.json").read_text())
            self.assertEqual(summary["summary_metrics"]["welfare"], 1.25)


class CombinedTableTests(unittest.TestCase):
    def test_combined_table_merges_multiseed_and_exploitability_by_mechanism(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            multi = root / "multi"
            exploit = root / "exploit"
            multi.mkdir()
            exploit.mkdir()
            with (multi / "summary_aggregate.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "mechanism",
                        "n_seeds",
                        "avg_price_mean",
                        "avg_price_ci95_low",
                        "avg_price_ci95_high",
                        "welfare_mean",
                        "welfare_ci95_low",
                        "welfare_ci95_high",
                        "consumer_surplus_mean",
                        "consumer_surplus_ci95_low",
                        "consumer_surplus_ci95_high",
                        "collusion_index_mean",
                        "collusion_index_ci95_low",
                        "collusion_index_ci95_high",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "mechanism": "none",
                        "n_seeds": "2",
                        "avg_price_mean": "5.0",
                        "avg_price_ci95_low": "4.0",
                        "avg_price_ci95_high": "6.0",
                        "welfare_mean": "100.0",
                        "welfare_ci95_low": "90.0",
                        "welfare_ci95_high": "110.0",
                        "consumer_surplus_mean": "50.0",
                        "consumer_surplus_ci95_low": "45.0",
                        "consumer_surplus_ci95_high": "55.0",
                        "collusion_index_mean": "0.5",
                        "collusion_index_ci95_low": "0.4",
                        "collusion_index_ci95_high": "0.6",
                    }
                )
            with (exploit / "summary_aggregate.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "mechanism",
                        "n_seeds",
                        "exploitability_mean",
                        "exploitability_ci95_low",
                        "exploitability_ci95_high",
                        "victim_loss_mean",
                        "victim_loss_ci95_low",
                        "victim_loss_ci95_high",
                        "welfare_damage_mean",
                        "welfare_damage_ci95_low",
                        "welfare_damage_ci95_high",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "mechanism": "none",
                        "n_seeds": "2",
                        "exploitability_mean": "7.0",
                        "exploitability_ci95_low": "6.0",
                        "exploitability_ci95_high": "8.0",
                        "victim_loss_mean": "3.0",
                        "victim_loss_ci95_low": "2.0",
                        "victim_loss_ci95_high": "4.0",
                        "welfare_damage_mean": "1.0",
                        "welfare_damage_ci95_low": "0.0",
                        "welfare_damage_ci95_high": "2.0",
                    }
                )

            rows = combined_rows(multi, exploit, "q_learning")
            output = root / "combined.csv"
            write_combined_csv(output, rows)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["mind"], "q_learning")
            self.assertEqual(rows[0]["mechanism"], "none")
            self.assertEqual(rows[0]["welfare_mean"], "100.0")
            self.assertEqual(rows[0]["exploitability_mean"], "7.0")
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
