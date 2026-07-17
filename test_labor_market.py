import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.registry import build_experiment
from run_labor_market_smoke import parse_args, run
from worlds.labor_market.benchmarks import (
    best_worker_report_gains,
    blocking_pairs,
    canonical_matching_cases,
    deferred_acceptance,
    matching_welfare,
    preference_order,
    reported_preferences_from_top,
    truthful_matching,
)
from worlds.labor_market.env import LaborMarketConfig, LaborMarketWorld
from worlds.labor_market.features import labor_market_obs_dim, structured_observations
from worlds.labor_market.training import summarize_records, train_labor_market


class LaborMarketTests(unittest.TestCase):
    def test_deferred_acceptance_hand_computed_case(self):
        worker_prefs = np.asarray([[0, 1], [0, 1]])
        employer_prefs = np.asarray([[1, 0], [0, 1]])
        matches = deferred_acceptance(worker_prefs, employer_prefs)
        self.assertEqual(matches.tolist(), [1, 0])

    def test_blocking_pairs_detect_instability(self):
        worker_prefs = np.asarray([[0, 1], [1, 0]])
        employer_prefs = np.asarray([[0, 1], [1, 0]])
        unstable = np.asarray([1, 0])
        self.assertEqual(blocking_pairs(unstable, worker_prefs, employer_prefs), [(0, 0), (1, 1)])

    def test_reported_preferences_put_action_first(self):
        true_prefs = np.asarray([[0, 1, 2]])
        reported = reported_preferences_from_top(true_prefs, np.asarray([2]))
        self.assertEqual(reported.tolist(), [[2, 0, 1]])

    def test_truthful_matching_benchmark_is_stable(self):
        worker_values = np.asarray([[3.0, 1.0], [2.0, 4.0]])
        employer_values = np.asarray([[3.0, 1.0], [1.0, 4.0]])
        bench = truthful_matching(worker_values, employer_values)
        self.assertEqual(len(bench["blocking_pairs"]), 0)

    def test_canonical_matching_cases_cover_stable_and_unstable_profiles(self):
        cases = canonical_matching_cases()
        stable = cases["stable_truthful_2x2"]
        stable_bench = truthful_matching(stable["worker_values"], stable["employer_values"])
        self.assertEqual(stable_bench["matches"].tolist(), stable["expected_truthful_matches"].tolist())
        self.assertEqual(stable_bench["blocking_pairs"], stable["expected_blocking_pairs"])

        unstable = cases["unstable_forced_2x2"]
        worker_prefs = preference_order(unstable["worker_values"])
        employer_prefs = preference_order(unstable["employer_values"])
        self.assertEqual(
            blocking_pairs(unstable["forced_matches"], worker_prefs, employer_prefs),
            unstable["expected_blocking_pairs"],
        )

    def test_worker_proposing_da_has_no_profitable_worker_top_report_deviation(self):
        case = canonical_matching_cases()["contested_strategyproof_3x3"]
        gains = best_worker_report_gains(case["worker_values"], case["employer_values"])
        self.assertLessEqual(float(gains.max()), case["expected_max_worker_report_gain"] + 1e-12)

    def test_matching_welfare_accounts_for_both_sides(self):
        matches = np.asarray([0, 1])
        worker_values = np.asarray([[3.0, 1.0], [1.0, 4.0]])
        employer_values = np.asarray([[5.0, 1.0], [1.0, 6.0]])
        welfare = matching_welfare(matches, worker_values, employer_values)
        self.assertAlmostEqual(welfare["worker_welfare"], 7.0)
        self.assertAlmostEqual(welfare["employer_welfare"], 11.0)
        self.assertAlmostEqual(welfare["total_welfare"], 18.0)

    def test_world_step_reports_stability_and_rewards(self):
        cfg = LaborMarketConfig(
            n_workers=2,
            n_employers=2,
            worker_values=((3.0, 1.0), (2.0, 4.0)),
            employer_values=((3.0, 1.0), (1.0, 4.0)),
        )
        world = LaborMarketWorld(config=cfg, seed=1)
        _, rewards, _, info = world.step([0, 1])
        self.assertEqual(info["match_rate"], 1.0)
        self.assertEqual(info["stability"], 1.0)
        self.assertEqual(rewards.tolist(), [3.0, 4.0])

    def test_registry_builds_labor_market_world(self):
        world = build_experiment({"world": "labor_market", "institution": "deferred_acceptance", "seed": 4})
        self.assertIsInstance(world, LaborMarketWorld)

    def test_preference_order_descends_values(self):
        prefs = preference_order(np.asarray([[1.0, 3.0, 2.0]]))
        self.assertEqual(prefs.tolist(), [[1, 2, 0]])

    def test_q_learning_training_smoke_emits_metrics(self):
        cfg = LaborMarketConfig(n_workers=2, n_employers=2, max_rounds=30)
        result = train_labor_market(steps=30, seed=2, config=cfg)
        summary = summarize_records(result.records, final_window=10)
        self.assertIn("total_welfare", summary)
        self.assertTrue(math.isfinite(summary["match_rate"]))

    def test_structured_features_are_fixed_width_and_finite(self):
        cfg = LaborMarketConfig(n_workers=2, n_employers=3, max_rounds=5)
        world = LaborMarketWorld(config=cfg, seed=2)
        observations = structured_observations(world)

        self.assertEqual(len(observations), 2)
        self.assertEqual(len(observations[0]), labor_market_obs_dim(cfg.n_employers))
        self.assertTrue(np.all(np.isfinite(observations[0])))

    def test_deep_and_marl_minds_run_on_labor_market_workers(self):
        cfg = LaborMarketConfig(n_workers=2, n_employers=2, max_rounds=6)
        for mind in ("dqn", "ppo", "independent_dqn", "centralized_critic"):
            with self.subTest(mind=mind):
                result = train_labor_market(steps=6, seed=3, config=cfg, mind=mind)
                self.assertEqual(len(result.records), 6)
                self.assertEqual(result.mind, mind)
                self.assertEqual(result.obs_dim, labor_market_obs_dim(cfg.n_employers))

    def test_runner_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(["--steps", "20", "--n-seeds", "2", "--final-window", "5", "--save-dir", tmp])
            outputs = run(args)
            with Path(outputs["summary_by_seed"]).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertTrue(Path(outputs["summary_aggregate"]).exists())


if __name__ == "__main__":
    unittest.main()
