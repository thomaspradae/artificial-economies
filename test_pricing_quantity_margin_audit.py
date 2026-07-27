import csv
import tempfile
import unittest
from pathlib import Path

from run_pricing_quantity_margin_audit import aggregate_delta_rows, existing_full_seed_deltas


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mechanism",
        "seed",
        "steps",
        "final_window",
        "avg_price",
        "quantity_total",
        "profit_total",
        "welfare",
        "collusion_index",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class PricingQuantityMarginAuditTests(unittest.TestCase):
    def test_existing_full_seed_deltas_pair_none_and_price_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_summary(
                tmp_path / "dqn_v0_multiseed" / "summary_by_seed.csv",
                [
                    {
                        "mechanism": "none",
                        "seed": 0,
                        "steps": 40000,
                        "final_window": 1000,
                        "avg_price": 4.0,
                        "quantity_total": 90.0,
                        "profit_total": 100.0,
                        "welfare": 200.0,
                        "collusion_index": 0.5,
                    },
                    {
                        "mechanism": "price_cap",
                        "seed": 0,
                        "steps": 40000,
                        "final_window": 1000,
                        "avg_price": 3.8,
                        "quantity_total": 95.0,
                        "profit_total": 120.0,
                        "welfare": 210.0,
                        "collusion_index": 0.4,
                    },
                    {
                        "mechanism": "none",
                        "seed": 1,
                        "steps": 40000,
                        "final_window": 1000,
                        "avg_price": 4.2,
                        "quantity_total": 91.0,
                        "profit_total": 110.0,
                        "welfare": 205.0,
                        "collusion_index": 0.6,
                    },
                    {
                        "mechanism": "price_cap",
                        "seed": 1,
                        "steps": 40000,
                        "final_window": 1000,
                        "avg_price": 3.9,
                        "quantity_total": 94.0,
                        "profit_total": 100.0,
                        "welfare": 204.0,
                        "collusion_index": 0.45,
                    },
                ],
            )

            rows = existing_full_seed_deltas(tmp_path, ["dqn"])
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["delta_profit_total"], 20.0)
            self.assertEqual(rows[0]["delta_quantity_total"], 5.0)
            self.assertEqual(rows[0]["price_cap_profit_higher"], 1.0)
            self.assertEqual(rows[1]["delta_profit_total"], -10.0)
            self.assertEqual(rows[1]["price_cap_profit_higher"], 0.0)

            aggregate = aggregate_delta_rows(rows)
            self.assertEqual(len(aggregate), 1)
            self.assertEqual(aggregate[0]["n"], 2)
            self.assertEqual(aggregate[0]["positive_profit_delta_count"], 1)
            self.assertEqual(aggregate[0]["positive_profit_delta_share"], 0.5)
            self.assertEqual(aggregate[0]["delta_profit_total_mean"], 5.0)


if __name__ == "__main__":
    unittest.main()
