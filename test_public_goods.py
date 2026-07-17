import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.registry import build_experiment
from institutions.public_goods import ContributionMatching, InformationRestriction, PublicGoodsPenalty
from institutions.tax_schedule import TaxSchedule
from run_public_goods_smoke import parse_args, run
from validate_public_goods_effects import validate_effects
from worlds.public_goods.benchmarks import (
    free_rider_benchmark,
    ration_extractions,
    social_optimum_benchmark,
)
from worlds.public_goods.env import (
    CONTRIBUTE_HIGH,
    EXTRACT_HIGH,
    EXTRACT_LOW,
    NOOP,
    PublicGoodsConfig,
    PublicGoodsWorld,
)
from worlds.public_goods.features import public_goods_obs_dim, structured_observations
from worlds.public_goods.training import summarize_records, train_public_goods


class PublicGoodsTests(unittest.TestCase):
    def test_rationing_is_proportional_when_pool_is_short(self):
        realized = ration_extractions([2.0, 2.0], available_pool=2.0)
        self.assertEqual(realized.tolist(), [1.0, 1.0])

    def test_contribution_and_extraction_update_pool_and_rewards(self):
        cfg = PublicGoodsConfig(n_agents=2, initial_pool=4.0, pool_capacity=10.0, regeneration_rate=0.0)
        world = PublicGoodsWorld(config=cfg, seed=1)
        _, rewards, _, info = world.step([CONTRIBUTE_HIGH, EXTRACT_LOW])

        self.assertAlmostEqual(info["contribution_total"], 2.0)
        self.assertAlmostEqual(info["extraction_total"], 1.0)
        self.assertAlmostEqual(world.pool_stock, 4.0 - 1.0 + 2.0 * cfg.public_multiplier)
        self.assertLess(rewards[0], 0.0)
        self.assertGreater(rewards[1], 0.0)

    def test_penalty_institution_binds_on_high_extraction(self):
        cfg = PublicGoodsConfig(n_agents=1, initial_pool=10.0, regeneration_rate=0.0)
        world = PublicGoodsWorld(config=cfg, institution=PublicGoodsPenalty(sustainable_extraction=0.5), seed=1)
        _, rewards, _, info = world.step([EXTRACT_HIGH])

        self.assertGreater(info["penalty_total"], 0.0)
        self.assertLess(rewards[0], 2.0)

    def test_contribution_matching_adds_public_stock(self):
        cfg = PublicGoodsConfig(n_agents=1, initial_pool=1.0, pool_capacity=10.0, regeneration_rate=0.0)
        unmatched = PublicGoodsWorld(config=cfg, seed=1)
        matched = PublicGoodsWorld(config=cfg, institution=ContributionMatching(match_rate=1.0), seed=1)
        unmatched.step([CONTRIBUTE_HIGH])
        _, _, _, info = matched.step([CONTRIBUTE_HIGH])

        self.assertGreater(matched.pool_stock, unmatched.pool_stock)
        self.assertAlmostEqual(info["matched_contribution"], 2.0)

    def test_information_restriction_hides_pool_bin(self):
        cfg = PublicGoodsConfig(n_agents=1, initial_pool=10.0, pool_capacity=10.0)
        world = PublicGoodsWorld(config=cfg, institution=InformationRestriction(hidden_pool_bin=2), seed=1)
        self.assertEqual(world.observations()[0][0], 2)

    def test_tax_schedule_redistributes_positive_rewards(self):
        tax = TaxSchedule(brackets=((0.0, 0.5),), redistribute=True)
        out = tax.apply({"phase": "post_rewards", "rewards": [2.0, 0.0]})
        self.assertAlmostEqual(sum(out["rewards"]), 2.0)
        self.assertGreater(out["rewards"][1], 0.0)

    def test_benchmark_brackets_are_finite(self):
        cfg = PublicGoodsConfig(n_agents=2)
        free = free_rider_benchmark(cfg, steps=5)
        social = social_optimum_benchmark(cfg, steps=5)
        self.assertTrue(math.isfinite(free["welfare"]))
        self.assertTrue(math.isfinite(social["sustainability"]))

    def test_registry_builds_public_goods_world(self):
        world = build_experiment({"world": "public_goods", "institution": "none", "seed": 4})
        self.assertIsInstance(world, PublicGoodsWorld)

    def test_q_learning_training_smoke_emits_metrics(self):
        cfg = PublicGoodsConfig(n_agents=2, max_rounds=30)
        result = train_public_goods(steps=30, seed=2, config=cfg)
        summary = summarize_records(result.records, final_window=10)
        self.assertIn("welfare", summary)
        self.assertTrue(math.isfinite(summary["sustainability"]))

    def test_structured_features_are_fixed_width_and_finite(self):
        cfg = PublicGoodsConfig(n_agents=3, max_rounds=5)
        world = PublicGoodsWorld(config=cfg, seed=2)
        observations = structured_observations(world)

        self.assertEqual(len(observations), 3)
        self.assertEqual(len(observations[0]), public_goods_obs_dim())
        self.assertTrue(np.all(np.isfinite(observations[0])))

    def test_deep_and_marl_minds_run_on_public_goods(self):
        cfg = PublicGoodsConfig(n_agents=2, max_rounds=6)
        for mind in ("dqn", "ppo", "independent_dqn", "centralized_critic"):
            with self.subTest(mind=mind):
                result = train_public_goods(steps=6, seed=3, config=cfg, mind=mind)
                self.assertEqual(len(result.records), 6)
                self.assertEqual(result.mind, mind)
                self.assertEqual(result.obs_dim, public_goods_obs_dim())

    def test_runner_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(
                [
                    "--steps",
                    "20",
                    "--n-seeds",
                    "2",
                    "--final-window",
                    "5",
                    "--institutions",
                    "none",
                    "public_goods_penalty",
                    "--save-dir",
                    tmp,
                ]
            )
            outputs = run(args)
            with Path(outputs["summary_by_seed"]).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertTrue(Path(outputs["summary_aggregate"]).exists())

    def test_effect_validator_distinguishes_state_from_reward_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            aggregate = Path(tmp) / "summary_aggregate.csv"
            aggregate.write_text(
                "\n".join(
                    [
                        "institution,n_seeds,sustainability_mean,contribution_total_mean,extraction_total_mean,collapse_rate_mean,welfare_mean,reward_total_mean,penalty_total_mean,reputation_bonus_total_mean,tax_revenue_mean",
                        "none,3,0.10,0.10,1.0,0.8,1.0,1.0,0.0,0.0,0.0",
                        "contribution_matching,3,0.13,0.20,1.1,0.7,1.2,1.1,0.0,0.0,0.0",
                        "tax_schedule,3,0.10,0.10,1.0,0.8,1.0,1.0,0.0,0.0,0.3",
                    ]
                )
                + "\n"
            )
            report = validate_effects(aggregate, state_threshold=0.01, reward_threshold=0.01)

        self.assertEqual(report["institutions"]["contribution_matching"]["classification"], "state_and_reward")
        self.assertEqual(report["institutions"]["tax_schedule"]["classification"], "reward_or_accounting_only")


if __name__ == "__main__":
    unittest.main()
