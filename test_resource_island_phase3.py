import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_resource_island_mind_comparison import build_comparison, write_csv as write_comparison_csv
from run_resource_island_phase3_validation import parse_args as parse_validation_args
from run_resource_island_phase3_validation import run as run_validation
from run_resource_island_smoke import parse_args as parse_smoke_args
from run_resource_island_smoke import run as run_smoke
from worlds.resource_island.env import FOOD, ResourceIslandConfig, ResourceIslandWorld
from worlds.resource_island.features import (
    encode_resource_island_observation,
    resource_island_obs_dim,
    structured_observations,
)
from worlds.resource_island.training import train_resource_island


def zero_resources(grid_size: int = 3) -> np.ndarray:
    return np.zeros((grid_size, grid_size, 2), dtype=int)


class ResourceIslandFeatureTests(unittest.TestCase):
    def test_structured_feature_width_is_grid_size_independent(self):
        small = ResourceIslandWorld(
            config=ResourceIslandConfig(grid_size=3, n_agents=2, initial_resources=zero_resources(3)),
            seed=1,
        )
        large = ResourceIslandWorld(
            config=ResourceIslandConfig(grid_size=6, n_agents=2, initial_resources=zero_resources(6)),
            seed=1,
        )
        small_obs = encode_resource_island_observation(small, 0, radius=1)
        large_obs = encode_resource_island_observation(large, 0, radius=1)
        self.assertEqual(len(small_obs), resource_island_obs_dim(radius=1))
        self.assertEqual(len(large_obs), resource_island_obs_dim(radius=1))
        self.assertEqual(len(small_obs), len(large_obs))
        self.assertTrue(np.all(np.isfinite(small_obs)))
        self.assertTrue(np.all(np.isfinite(large_obs)))

    def test_structured_features_include_inventory_and_nearby_context(self):
        cfg = ResourceIslandConfig(
            grid_size=3,
            n_agents=2,
            start_positions=((1, 1), (1, 2)),
            initial_resources=zero_resources(3),
        )
        world = ResourceIslandWorld(config=cfg, seed=1)
        world.inventory[0, FOOD] = 2
        observations = structured_observations(world, radius=1)
        self.assertEqual(len(observations), 2)
        self.assertGreater(observations[0][1], 0.0)
        self.assertGreater(observations[0][-3], 0.0)


class ResourceIslandPhase3TrainingTests(unittest.TestCase):
    def test_deep_and_marl_minds_run_on_resource_island(self):
        cfg = ResourceIslandConfig(
            grid_size=3,
            n_agents=2,
            max_steps=8,
            initial_resource_units=4,
            resource_spawn_probability=0.0,
        )
        for mind in ("dqn", "ppo", "independent_dqn", "centralized_critic"):
            with self.subTest(mind=mind):
                result = train_resource_island(steps=6, seed=2, config=cfg, mind=mind)
                self.assertEqual(len(result.records), 6)
                self.assertEqual(result.mind, mind)
                self.assertEqual(result.obs_dim, resource_island_obs_dim(radius=1))
                self.assertTrue(all("survival_rate" in record for record in result.records))

    def test_smoke_runner_accepts_dqn_mind(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parse_smoke_args(
                [
                    "--mind",
                    "dqn",
                    "--steps",
                    "5",
                    "--n-seeds",
                    "1",
                    "--final-window",
                    "3",
                    "--institutions",
                    "none",
                    "--save-dir",
                    tmpdir,
                ]
            )
            outputs = run_smoke(args)
            manifest = json.loads(outputs["manifest"].read_text())
            self.assertEqual(manifest["mind"], "dqn")
            with outputs["summary_by_seed"].open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)

    def test_phase3_validation_runner_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parse_validation_args(
                [
                    "--save-dir",
                    tmpdir,
                    "--single-agent-steps",
                    "5",
                    "--comparison-steps",
                    "5",
                    "--final-window",
                    "3",
                ]
            )
            outputs = run_validation(args)
            for path in outputs.values():
                self.assertTrue(Path(path).exists())
            with outputs["single_agent_validation"].open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["mind"] for row in rows}, {"dqn", "ppo"})


class ResourceIslandComparisonTests(unittest.TestCase):
    def test_comparison_builder_merges_aggregate_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            q_dir = root / "q"
            dqn_dir = root / "dqn"
            q_dir.mkdir()
            dqn_dir.mkdir()
            for directory, value in ((q_dir, 0.8), (dqn_dir, 0.9)):
                (directory / "summary_aggregate.csv").write_text(
                    "institution,n_seeds,survival_rate_mean,survival_rate_std,welfare_mean,welfare_std,"
                    "trade_count_mean,trade_count_std\n"
                    f"none,2,{value},0.1,1.0,0.2,3.0,0.3\n"
                )
            rows = build_comparison([("q_learning", q_dir), ("dqn", dqn_dir)])
            self.assertEqual(len(rows), 2)
            out = root / "comparison.csv"
            write_comparison_csv(out, rows)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
