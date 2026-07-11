import unittest

from core.registry import build_experiment
from institutions.none import NoInstitution
from worlds.auction_house.benchmarks import first_price_equilibrium_bid, truthful_bid_benchmark
from worlds.auction_house.env import AuctionHouseConfig, AuctionHouseWorld


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

    def test_tie_breaking_is_deterministic_lowest_bidder_id(self):
        cfg = AuctionHouseConfig(n_bidders=3, fixed_valuations=(1.0, 5.0, 9.0))
        world = AuctionHouseWorld(config=cfg, seed=1)
        _, _, _, info = world.step([4.0, 4.0, 4.0])

        self.assertEqual(info["winner"], 0.0)
        self.assertEqual(info["efficient_winner"], 2.0)
        self.assertEqual(info["allocative_efficiency"], 0.0)

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

    def test_benchmarks_are_explicitly_deferred(self):
        with self.assertRaises(NotImplementedError):
            truthful_bid_benchmark()
        with self.assertRaises(NotImplementedError):
            first_price_equilibrium_bid()


if __name__ == "__main__":
    unittest.main()
