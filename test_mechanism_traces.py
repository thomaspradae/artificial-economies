import csv
import tempfile
import unittest
from pathlib import Path

from run_mechanism_traces import parse_args, run


class MechanismTraceTests(unittest.TestCase):
    def test_pricing_trace_runner_writes_n_firm_decomposition(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "trace"
            outputs = run(
                parse_args(
                    [
                        "--worlds",
                        "pricing_arena",
                        "--minds",
                        "q_learning",
                        "--steps",
                        "8",
                        "--final-window",
                        "4",
                        "--max-trace-rows",
                        "3",
                        "--n-agents",
                        "3",
                        "--save-dir",
                        str(save_dir),
                    ]
                )
            )
            with outputs["decomposition"].open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            with outputs["trace_steps"].open(newline="") as handle:
                trace_rows = list(csv.DictReader(handle))

        self.assertTrue(trace_rows)
        self.assertTrue(all(row["n_agents"] == "3" for row in rows))
        self.assertIn("price_cap_vs_none", {row["institution"] for row in rows})
        self.assertIn("delta_profit_total", {row["component"] for row in rows})


if __name__ == "__main__":
    unittest.main()
