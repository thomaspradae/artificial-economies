import csv
import tempfile
import unittest
from pathlib import Path

from build_cross_world_synthesis import build_coverage_report, build_protocol_comparability_report
from build_world_mind_comparison import build_paired_mind_effects, build_uncertainty


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class WorldComparisonBuilderTests(unittest.TestCase):
    def test_paired_mind_effects_compare_common_seeds_to_q_learning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = Path(tmpdir) / "outputs"
            results = []
            for mind, offset in (("q_learning", 0.0), ("dqn", 2.0)):
                result = outputs / mind
                rows = []
                for seed in (0, 1):
                    rows.append(
                        {
                            "institution": "deferred_acceptance",
                            "seed": seed,
                            "match_rate": 1.0,
                            "stability": 0.8,
                            "truthful_report_rate": 0.7,
                            "total_welfare": 3.0 + seed + offset,
                            "manipulation_gain_mean": 0.0,
                        }
                    )
                write_csv(result / "summary_by_seed.csv", list(rows[0]), rows)
                results.append((mind, result))
            paired = build_paired_mind_effects(
                world="labor_market", results=results, outputs_dir=outputs
            )

        effect = next(row for row in paired if row["metric"] == "total_welfare")
        self.assertEqual(effect["baseline_mind"], "q_learning")
        self.assertEqual(effect["n"], 2)
        self.assertAlmostEqual(effect["mean"], 2.0)
        self.assertAlmostEqual(effect["std"], 0.0)

    def test_pricing_uncertainty_pairs_exploitability_by_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = Path(tmpdir) / "outputs"
            result = outputs / "full_v0_multiseed"
            exploit = outputs / "v1_exploitability"
            multi_rows = []
            exploit_rows = []
            for mechanism, offset in (("none", 0.0), ("price_cap", -2.0)):
                for seed in (0, 1):
                    multi_rows.append(
                        {
                            "mechanism": mechanism,
                            "seed": seed,
                            "avg_price": 3.0,
                            "price_dispersion": 0.0,
                            "profit_total": 5.0,
                            "quantity_total": 5.0,
                            "welfare": 5.0,
                            "collusion_index": 0.0,
                        }
                    )
                    exploit_rows.append(
                        {
                            "mechanism": mechanism,
                            "seed": seed,
                            "exploitability": 10.0 + seed + offset,
                            "victim_loss": 2.0 + offset,
                            "welfare_damage": 1.0 + offset,
                        }
                    )
            write_csv(result / "summary_by_seed.csv", list(multi_rows[0]), multi_rows)
            write_csv(exploit / "summary_by_seed.csv", list(exploit_rows[0]), exploit_rows)
            _, paired = build_uncertainty(
                world="pricing_arena", results=[("q_learning", result)], outputs_dir=outputs
            )

        effect = next(
            row
            for row in paired
            if row["institution"] == "price_cap" and row["metric"] == "exploitability"
        )
        self.assertEqual(effect["n"], 2)
        self.assertAlmostEqual(effect["mean"], -2.0)
        self.assertAlmostEqual(effect["std"], 0.0)

    def test_auction_uncertainty_uses_seed_level_benchmark_gaps_and_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = Path(tmpdir) / "outputs"
            result = outputs / "auction_house_phase3_full" / "dqn"
            rows = []
            for scenario, values in (("second_price", (1.0, 3.0)), ("first_price", (2.0, 4.0))):
                for seed, welfare in enumerate(values):
                    rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "revenue": welfare,
                            "bidder_surplus": welfare,
                            "welfare": welfare,
                            "allocative_efficiency": welfare / 4.0,
                            "ex_post_regret_mean": welfare / 10.0,
                            "truthful_bid_distance_mean": welfare / 10.0,
                            "first_price_shading_distance_mean": welfare / 10.0,
                            "overbid_rate": 0.0,
                            "underbid_rate": 0.0,
                            "no_sale": 0.0,
                            "benchmark_revenue": 1.0,
                            "benchmark_bidder_surplus": 1.0,
                            "benchmark_welfare": 1.0,
                            "benchmark_allocative_efficiency": 1.0,
                        }
                    )
            write_csv(result / "summary_by_seed.csv", list(rows[0]), rows)
            uncertainty, paired = build_uncertainty(
                world="auction_house", results=[("dqn", result)], outputs_dir=outputs
            )

        observed = next(
            row
            for row in uncertainty
            if row["institution"] == "second_price"
            and row["metric"] == "welfare"
            and row["estimate_type"] == "observed"
        )
        gap = next(
            row
            for row in uncertainty
            if row["institution"] == "second_price"
            and row["metric"] == "welfare"
            and row["estimate_type"] == "benchmark_gap"
        )
        effect = next(
            row
            for row in paired
            if row["institution"] == "first_price" and row["metric"] == "welfare"
        )
        self.assertEqual(observed["n"], 2)
        self.assertAlmostEqual(observed["mean"], 2.0)
        self.assertAlmostEqual(gap["mean"], 1.0)
        self.assertAlmostEqual(effect["mean"], 1.0)
        self.assertAlmostEqual(effect["std"], 0.0)

    def test_coverage_report_flags_unbalanced_historical_scenarios(self):
        rows = [
            {"world": "auction_house", "mind": "q_learning", "institution": "second_price"},
            {"world": "auction_house", "mind": "dqn", "institution": "second_price"},
            {"world": "auction_house", "mind": "dqn", "institution": "clock"},
        ]
        report = build_coverage_report(rows)["auction_house"]
        self.assertFalse(report["balanced"])
        self.assertEqual(report["common_institutions"], ["second_price"])

    def test_protocol_report_flags_resource_v1_vs_default_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = []
            for mind, config in (
                (
                    "q_learning",
                    {
                        "steps": 40000,
                        "n_seeds": 20,
                        "resource_layout": "contested",
                        "activation_preset": "pressure",
                        "specialization_preset": "complementary",
                        "trade_food_units": 2,
                    },
                ),
                ("dqn", {"steps": 40000, "n_seeds": 20}),
            ):
                source = root / mind
                source.mkdir()
                (source / "experiment_manifest.json").write_text(
                    __import__("json").dumps({"config": config})
                )
                rows.append(
                    {
                        "world": "resource_island",
                        "mind": mind,
                        "raw_source_dir": str(source),
                    }
                )
            report = build_protocol_comparability_report(rows)["resource_island"]

        self.assertEqual(report["status"], "mismatch")
        self.assertFalse(report["cross_mind_capability_claims_valid"])
        self.assertEqual(report["mismatched_minds"], ["dqn"])


if __name__ == "__main__":
    unittest.main()
