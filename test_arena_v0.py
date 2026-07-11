import math
import unittest

import numpy as np

from arena_v0 import (
    DuopolyMarket,
    MarketConfig,
    compute_static_benchmarks,
    evaluate_policy_pair,
    train_adversary_against_frozen_firm1,
    train_market,
    train_market_with_agents,
)


class ArenaV0Tests(unittest.TestCase):
    def test_demand_is_non_negative_and_bounded_by_market_size(self):
        env = DuopolyMarket(MarketConfig())
        quantities = env.demand(np.array([3.0, 4.0]))
        self.assertEqual(quantities.shape, (2,))
        self.assertTrue(np.all(quantities >= 0.0))
        self.assertLessEqual(float(np.sum(quantities)), env.cfg.market_size)

    def test_price_cap_changes_effective_price(self):
        cfg = MarketConfig(mechanism="price_cap", price_cap=5.5)
        env = DuopolyMarket(cfg)
        _, _, info = env.step((18, 18))
        self.assertEqual(info["p1"], 5.5)
        self.assertEqual(info["p2"], 5.5)
        self.assertEqual(info["raw_p1"], 10.0)
        self.assertEqual(info["raw_p2"], 10.0)

    def test_static_benchmarks_are_available(self):
        benchmarks = compute_static_benchmarks(np.linspace(1.0, 10.0, 19))
        self.assertTrue(benchmarks["nash_pairs"])
        self.assertIsNotNone(benchmarks["nash_price"])
        self.assertGreaterEqual(benchmarks["monopoly_price"], 1.0)

    def test_short_training_run_produces_finite_metrics(self):
        data, _ = train_market(mechanism="none", steps=25, seed=3)
        self.assertEqual(len(data["avg_price"]), 25)
        self.assertTrue(math.isfinite(float(data["welfare"][-1])))

    def test_training_run_exposes_agents_without_breaking_old_api(self):
        run = train_market_with_agents(mechanism="none", steps=25, seed=3)
        self.assertEqual(len(run.data["avg_price"]), 25)
        self.assertEqual(run.agent1.q_values.shape, run.agent2.q_values.shape)
        self.assertEqual(run.final_state[0] >= 0, True)
        self.assertIn("nash_price", run.benchmarks)

    def test_policy_evaluation_does_not_mutate_q_tables(self):
        run = train_market_with_agents(mechanism="none", steps=25, seed=4)
        q1_before = run.agent1.q_values.copy()
        q2_before = run.agent2.q_values.copy()

        data = evaluate_policy_pair(run.agent1, run.agent2, mechanism="none", steps=10, seed=100)

        self.assertEqual(len(data["avg_price"]), 10)
        self.assertTrue(np.array_equal(q1_before, run.agent1.q_values))
        self.assertTrue(np.array_equal(q2_before, run.agent2.q_values))

    def test_adversary_training_leaves_frozen_firm_unchanged(self):
        run = train_market_with_agents(mechanism="none", steps=25, seed=5)
        frozen_q_before = run.agent1.q_values.copy()

        adversary_run = train_adversary_against_frozen_firm1(
            run.agent1,
            mechanism="none",
            steps=25,
            seed=500,
        )

        self.assertEqual(len(adversary_run.data["avg_price"]), 25)
        self.assertTrue(np.array_equal(frozen_q_before, run.agent1.q_values))
        self.assertFalse(np.array_equal(adversary_run.agent2.q_values, np.zeros_like(adversary_run.agent2.q_values)))


if __name__ == "__main__":
    unittest.main()
