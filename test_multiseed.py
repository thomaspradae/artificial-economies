import csv
import importlib.util
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from run_multiseed import (
    SUMMARY_METRICS,
    aggregate_rows,
    mean_ci95,
    parse_args,
    run_experiment,
    summarize_final_window,
)


class MultiSeedStatsTests(unittest.TestCase):
    def test_mean_ci95_uses_student_t_interval(self):
        stats = mean_ci95([1.0, 2.0, 3.0])
        expected_half_width = 4.303 / math.sqrt(3.0)

        self.assertEqual(stats["n"], 3)
        self.assertAlmostEqual(stats["mean"], 2.0)
        self.assertAlmostEqual(stats["std"], 1.0)
        self.assertAlmostEqual(stats["sem"], 1.0 / math.sqrt(3.0))
        self.assertAlmostEqual(stats["ci95_half_width"], expected_half_width)

    def test_mean_ci95_does_not_fake_uncertainty_for_one_seed(self):
        stats = mean_ci95([4.0])
        self.assertEqual(stats["n"], 1)
        self.assertEqual(stats["mean"], 4.0)
        self.assertTrue(math.isnan(stats["ci95_half_width"]))

    def test_summarize_final_window_uses_tail_only(self):
        data = {
            "avg_price": np.array([1.0, 2.0, 3.0, 5.0]),
            "p1": np.array([1.0, 2.0, 3.0, 4.0]),
            "p2": np.array([1.5, 2.5, 4.0, 7.0]),
            "profit1": np.array([10.0, 20.0, 30.0, 40.0]),
            "profit2": np.array([1.0, 2.0, 3.0, 4.0]),
            "reward1": np.array([9.0, 19.0, 29.0, 39.0]),
            "reward2": np.array([0.5, 1.5, 2.5, 3.5]),
            "consumer_surplus": np.array([100.0, 90.0, 80.0, 70.0]),
            "welfare": np.array([111.0, 112.0, 113.0, 114.0]),
            "collusion_index": np.array([0.0, 0.1, 0.4, 0.8]),
            "penalty1": np.array([1.0, 1.0, 1.0, 1.0]),
            "penalty2": np.array([0.0, 0.0, 2.0, 2.0]),
            "quantity1": np.array([20.0, 21.0, 22.0, 23.0]),
            "quantity2": np.array([10.0, 11.0, 12.0, 13.0]),
            "audit_hit": np.array([0.0, 0.0, 1.0, 0.0]),
            "market_size": np.array([100.0, 100.0, 90.0, 110.0]),
        }
        benchmarks = {"nash_price": 2.5, "monopoly_price": 8.0}

        summary = summarize_final_window(data, benchmarks, final_window=2)

        self.assertEqual(summary["effective_final_window"], 2.0)
        self.assertAlmostEqual(summary["avg_price"], 4.0)
        self.assertAlmostEqual(summary["price_dispersion"], 2.0)
        self.assertAlmostEqual(summary["profit_total"], 38.5)
        self.assertAlmostEqual(summary["reward_total"], 37.0)
        self.assertAlmostEqual(summary["penalty_total"], 3.0)
        self.assertAlmostEqual(summary["audit_rate"], 0.5)
        self.assertAlmostEqual(summary["nash_price_gap"], 1.5)

    def test_aggregate_rows_keeps_requested_mechanism_order(self):
        rows = []
        for mechanism, values in (("price_cap", [2.0, 4.0]), ("none", [5.0, 7.0])):
            for seed_index, value in enumerate(values):
                row = {
                    "mechanism": mechanism,
                    "seed_index": seed_index,
                    "seed": seed_index,
                    "steps": 10,
                    "final_window": 5,
                    "effective_final_window": 5,
                    "nash_price": 2.5,
                    "monopoly_price": 8.0,
                }
                for metric in SUMMARY_METRICS:
                    row[metric] = value
                rows.append(row)

        aggregate = aggregate_rows(rows, mechanisms=["none", "price_cap"])

        self.assertEqual([row["mechanism"] for row in aggregate], ["none", "price_cap"])
        self.assertAlmostEqual(aggregate[0]["avg_price_mean"], 6.0)
        self.assertAlmostEqual(aggregate[1]["avg_price_mean"], 3.0)
        self.assertEqual(aggregate[0]["avg_price_n"], 2)


class MultiSeedEndToEndTests(unittest.TestCase):
    def test_small_no_plot_run_writes_core_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parse_args(
                [
                    "--steps",
                    "25",
                    "--n-seeds",
                    "2",
                    "--final-window",
                    "5",
                    "--save-dir",
                    tmpdir,
                    "--mechanisms",
                    "none",
                    "price_cap",
                    "--no-plots",
                ]
            )
            with redirect_stdout(io.StringIO()):
                run_experiment(args)

            output_dir = Path(tmpdir)
            by_seed = output_dir / "summary_by_seed.csv"
            aggregate = output_dir / "summary_aggregate.csv"
            rankings = output_dir / "mechanism_rankings.csv"
            manifest = output_dir / "experiment_manifest.json"

            self.assertTrue(by_seed.exists())
            self.assertTrue(aggregate.exists())
            self.assertTrue(rankings.exists())
            self.assertTrue(manifest.exists())

            with by_seed.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)

            with aggregate.open(newline="") as handle:
                aggregate_rows = list(csv.DictReader(handle))
            self.assertEqual([row["mechanism"] for row in aggregate_rows], ["none", "price_cap"])
            self.assertIn("welfare_ci95_low", aggregate_rows[0])

            payload = json.loads(manifest.read_text())
            self.assertEqual(payload["config"]["n_seeds"], 2)
            self.assertEqual(payload["mechanisms"], ["none", "price_cap"])
            self.assertEqual(payload["ci_method"], "Student-t two-sided 95% confidence interval over seed-level means")

    def test_small_plot_run_writes_expected_pngs(self):
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            args = parse_args(
                [
                    "--steps",
                    "25",
                    "--n-seeds",
                    "2",
                    "--final-window",
                    "5",
                    "--smoothing",
                    "5",
                    "--save-dir",
                    tmpdir,
                    "--mechanisms",
                    "none",
                ]
            )
            with redirect_stdout(io.StringIO()):
                run_experiment(args)

            output_dir = Path(tmpdir)
            self.assertTrue((output_dir / "avg_price_ci.png").exists())
            self.assertTrue((output_dir / "welfare_ci.png").exists())
            self.assertTrue((output_dir / "consumer_surplus_ci.png").exists())
            self.assertTrue((output_dir / "collusion_ci.png").exists())
            self.assertTrue((output_dir / "collusion_index_ci.png").exists())


if __name__ == "__main__":
    unittest.main()
