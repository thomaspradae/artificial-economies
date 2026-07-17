import unittest

import numpy as np

from build_world_mind_comparison import build_comparison
from core.registry import build_experiment
from institutions.auction_house import AuctionInformationPolicy
from institutions.none import NoInstitution
from worlds.auction_house.benchmarks import (
    ex_post_bidder_regret,
    expected_outcome_over_grid,
    first_price_bid_benchmark,
    first_price_equilibrium_bid,
    truthful_bid_benchmark,
)
from worlds.auction_house.env import AuctionHouseConfig, AuctionHouseWorld
from worlds.auction_house.features import auction_house_obs_dim, structured_observations
from worlds.auction_house.training import benchmark_for_config, summarize_records, train_auction_house


class AuctionHouseMechanicsTests(unittest.TestCase):
    def test_second_price_allocates_to_highest_bidder_and_charges_second_bid(self):
        cfg = AuctionHouseConfig(
            n_bidders=3,
            auction_format="second_price",
            bid_grid=(0.0, 2.0, 5.0, 7.0),
            fixed_valuations=(6.0, 9.0, 4.0),
        )
        world = AuctionHouseWorld(config=cfg, seed=1)
        _, rewards, done, info = world.step([1, 3, 2])

        self.assertTrue(done)
        self.assertEqual(info["winner"], 1.0)
        self.assertEqual(info["payment"], 5.0)
        self.assertEqual(rewards.tolist(), [0.0, 4.0, 0.0])
        self.assertEqual(info["allocative_efficiency"], 1.0)

    def test_first_price_charges_winning_bid(self):
        cfg = AuctionHouseConfig(
            n_bidders=2,
            auction_format="first_price",
            bid_grid=(0.0, 3.0, 8.0),
            fixed_valuations=(9.0, 4.0),
        )
        world = AuctionHouseWorld(config=cfg, seed=1)
        _, rewards, _, info = world.step([2, 1])

        self.assertEqual(info["winner"], 0.0)
        self.assertEqual(info["payment"], 8.0)
        self.assertEqual(rewards.tolist(), [1.0, 0.0])

    def test_clock_auction_uses_second_price_clearing_rule(self):
        cfg = AuctionHouseConfig(
            n_bidders=3,
            auction_format="clock",
            bid_grid=(0.0, 3.0, 6.0, 9.0),
            fixed_valuations=(5.0, 9.0, 7.0),
        )
        world = AuctionHouseWorld(config=cfg, seed=1)
        _, rewards, _, info = world.step([1, 3, 2])

        self.assertEqual(info["winner"], 1.0)
        self.assertEqual(info["payment"], 6.0)
        self.assertEqual(rewards.tolist(), [0.0, 3.0, 0.0])
        self.assertEqual(info["allocative_efficiency"], 1.0)

    def test_information_policy_public_signal_changes_value_observation(self):
        cfg = AuctionHouseConfig(
            n_bidders=2,
            valuation_grid=(0.0, 5.0, 10.0),
            bid_grid=(0.0, 5.0, 10.0),
            fixed_valuations=(0.0, 10.0),
        )
        world = AuctionHouseWorld(
            config=cfg,
            institution=AuctionInformationPolicy(public_signal_weight=1.0),
            seed=1,
        )

        self.assertEqual(world.observations(), [(1,), (1,)])

    def test_tie_breaking_is_deterministic_lowest_bidder_id(self):
        cfg = AuctionHouseConfig(n_bidders=3, fixed_valuations=(1.0, 5.0, 9.0))
        world = AuctionHouseWorld(config=cfg, seed=1)
        _, _, _, info = world.step([4.0, 4.0, 4.0])

        self.assertEqual(info["winner"], 0.0)
        self.assertEqual(info["efficient_winner"], 2.0)
        self.assertEqual(info["allocative_efficiency"], 0.0)

    def test_reserve_price_allows_no_sale(self):
        cfg = AuctionHouseConfig(
            n_bidders=2,
            auction_format="second_price",
            reserve_price=5.0,
            fixed_valuations=(3.0, 4.0),
        )
        world = AuctionHouseWorld(config=cfg, seed=1)
        _, rewards, _, info = world.step([2.0, 4.0])

        self.assertEqual(info["winner"], -1.0)
        self.assertEqual(info["payment"], 0.0)
        self.assertEqual(info["revenue"], 0.0)
        self.assertEqual(info["welfare"], 0.0)
        self.assertEqual(rewards.tolist(), [0.0, 0.0])

    def test_registry_builds_auction_house_world(self):
        cfg = AuctionHouseConfig(fixed_valuations=(3.0, 7.0))
        world = build_experiment(
            {
                "world": "auction_house",
                "institution": "none",
                "seed": 3,
                "world_params": {"config": cfg},
            }
        )

        self.assertIsInstance(world, AuctionHouseWorld)
        self.assertIsInstance(world.institution, NoInstitution)

    def test_second_price_truthful_benchmark_is_hand_computable(self):
        self.assertEqual(truthful_bid_benchmark((0.0, 5.0, 10.0)), (0.0, 5.0, 10.0))
        outcome = expected_outcome_over_grid(
            valuation_grid=(0.0, 10.0),
            n_bidders=2,
            bid_strategy=lambda value: value,
            auction_format="second_price",
            reserve_price=0.0,
        )

        self.assertAlmostEqual(outcome["revenue"], 2.5)
        self.assertAlmostEqual(outcome["bidder_surplus"], 5.0)
        self.assertAlmostEqual(outcome["welfare"], 7.5)
        self.assertAlmostEqual(outcome["allocative_efficiency"], 1.0)

    def test_first_price_bid_shading_benchmark(self):
        self.assertAlmostEqual(first_price_equilibrium_bid(10.0, n_bidders=2), 5.0)
        self.assertEqual(
            first_price_bid_benchmark((0.0, 5.0, 10.0), n_bidders=2, bid_grid=(0.0, 2.0, 5.0, 10.0)),
            (0.0, 2.0, 5.0),
        )

    def test_truthful_second_price_has_zero_grid_regret(self):
        regret_0 = ex_post_bidder_regret(
            valuations=(4.0, 8.0),
            bids=(4.0, 8.0),
            bidder=0,
            bid_grid=(0.0, 4.0, 8.0),
            auction_format="second_price",
        )
        regret_1 = ex_post_bidder_regret(
            valuations=(4.0, 8.0),
            bids=(4.0, 8.0),
            bidder=1,
            bid_grid=(0.0, 4.0, 8.0),
            auction_format="second_price",
        )

        self.assertAlmostEqual(regret_0, 0.0)
        self.assertAlmostEqual(regret_1, 0.0)

    def test_benchmark_for_config_reports_expected_reference(self):
        cfg = AuctionHouseConfig(
            n_bidders=2,
            auction_format="second_price",
            valuation_grid=(0.0, 10.0),
            bid_grid=(0.0, 10.0),
        )
        benchmark = benchmark_for_config(cfg)

        self.assertAlmostEqual(benchmark["revenue"], 2.5)
        self.assertAlmostEqual(benchmark["welfare"], 7.5)

    def test_q_learning_training_smoke_emits_finite_metrics(self):
        cfg = AuctionHouseConfig(
            n_bidders=2,
            auction_format="second_price",
            valuation_grid=(0.0, 5.0, 10.0),
            bid_grid=(0.0, 5.0, 10.0),
        )
        result = train_auction_house(steps=50, seed=3, config=cfg, epsilon_decay=0.98)
        summary = summarize_records(result.records, final_window=20)

        self.assertEqual(len(result.records), 50)
        self.assertIn("revenue", summary)
        self.assertIn("ex_post_regret_mean", summary)
        self.assertTrue(summary["revenue"] == summary["revenue"])

    def test_structured_features_are_fixed_width_and_finite(self):
        cfg = AuctionHouseConfig(n_bidders=2, fixed_valuations=(2.0, 8.0))
        world = AuctionHouseWorld(config=cfg, seed=3)
        observations = structured_observations(world)

        self.assertEqual(len(observations), 2)
        self.assertEqual(len(observations[0]), auction_house_obs_dim())
        self.assertTrue(np.all(np.isfinite(observations[0])))

    def test_deep_and_marl_minds_run_on_auction_house(self):
        cfg = AuctionHouseConfig(
            n_bidders=2,
            auction_format="second_price",
            valuation_grid=(0.0, 5.0, 10.0),
            bid_grid=(0.0, 5.0, 10.0),
        )
        for mind in ("dqn", "ppo", "independent_dqn", "centralized_critic"):
            with self.subTest(mind=mind):
                result = train_auction_house(steps=6, seed=4, config=cfg, mind=mind)
                self.assertEqual(len(result.records), 6)
                self.assertEqual(result.mind, mind)
                self.assertEqual(result.obs_dim, auction_house_obs_dim())

    def test_generic_comparison_builder_handles_auction_scenarios(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            d = root / "auction"
            d.mkdir()
            (d / "summary_aggregate.csv").write_text(
                "scenario,n_seeds,welfare_mean,welfare_std,allocative_efficiency_mean,allocative_efficiency_std\n"
                "second_price,2,1.0,0.1,0.8,0.2\n"
            )
            rows = build_comparison(world="auction_house", results=[("dqn", d)])

        self.assertEqual(rows[0]["institution"], "second_price")
        self.assertAlmostEqual(rows[0]["welfare_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
