from __future__ import annotations

import json
import math
import os
import unittest
from pathlib import Path
from typing import Any


FLOAT_TOL = 1e-7


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def assert_close(testcase: unittest.TestCase, left: Any, right: Any, path: str) -> None:
    if is_number(left) and is_number(right):
        if math.isnan(float(left)) or math.isnan(float(right)):
            testcase.assertTrue(math.isnan(float(left)) and math.isnan(float(right)), path)
        else:
            testcase.assertLessEqual(abs(float(left) - float(right)), FLOAT_TOL, path)
        return
    if isinstance(left, list) and isinstance(right, list):
        testcase.assertEqual(len(left), len(right), path)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            assert_close(testcase, left_item, right_item, f"{path}[{index}]")
        return
    testcase.assertEqual(left, right, path)


class BellmanTargetMatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        stage5b_dir = Path(os.getenv("AUDIT_STAGE5B_DIR", "audit_outputs/stage5_bellman"))
        cls.stage5b_dir = stage5b_dir
        cls.pytorch_path = stage5b_dir / "pytorch_bellman.jsonl"
        cls.deepmind_path = stage5b_dir / "deepmind_bellman.jsonl"
        cls.compare_path = stage5b_dir / "bellman_compare.txt"
        required = [cls.pytorch_path, cls.deepmind_path, cls.compare_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise unittest.SkipTest(f"Stage 5B artifacts missing: {missing}")
        cls.pytorch_rows = read_jsonl(cls.pytorch_path)
        cls.deepmind_rows = read_jsonl(cls.deepmind_path)
        cls.compare_text = cls.compare_path.read_text(encoding="utf-8")

    def test_stage5b_compare_passed(self) -> None:
        self.assertIn("MATCH", self.compare_text)

    def test_bellman_rows_match(self) -> None:
        self.assertEqual(len(self.pytorch_rows), len(self.deepmind_rows))
        exact_fields = {
            "phase",
            "step",
            "batch_name",
            "batch_position",
            "batch_size",
            "replay_index",
            "action",
            "action_one_based",
            "reward",
            "terminal_flag",
            "true_terminal",
            "life_loss_terminal",
            "continuation_mask",
            "target_network_used",
            "max_dimension",
            "terminal_target_equals_reward",
            "nonterminal_gamma_applied",
            "replay_indices",
            "actions",
            "rewards",
            "terminal_flags",
            "maximizing_next_action",
            "maximizing_next_actions",
            "target_shape",
        }
        float_fields = {
            "gamma",
            "online_q_values",
            "selected_q",
            "target_next_q_values",
            "max_next_q",
            "discounted_continuation",
            "bellman_target",
            "td_error",
            "targets",
            "td_errors",
        }
        for row_index, (pytorch, deepmind) in enumerate(zip(self.pytorch_rows, self.deepmind_rows)):
            for field in exact_fields & pytorch.keys() & deepmind.keys():
                self.assertEqual(pytorch[field], deepmind[field], f"row {row_index} field {field}")
            for field in float_fields & pytorch.keys() & deepmind.keys():
                assert_close(self, pytorch[field], deepmind[field], f"row {row_index} field {field}")

    def test_required_controls_present(self) -> None:
        sample_rows = [row for row in self.pytorch_rows if row.get("phase") == "bellman_sample"]
        batch_rows = [row for row in self.pytorch_rows if row.get("phase") == "bellman_batch"]
        batch_sizes = {int(row["batch_size"]) for row in batch_rows}
        batch_names = {row["batch_name"] for row in sample_rows}

        self.assertIn(1, batch_sizes)
        self.assertIn(4, batch_sizes)
        self.assertIn(32, batch_sizes)
        self.assertIn("ordinary_nonterminal", batch_names)
        self.assertIn("true_terminal", batch_names)
        self.assertIn("life_loss_terminal", batch_names)
        self.assertIn("zero_reward", batch_names)
        self.assertIn("positive_reward", batch_names)
        self.assertTrue(any(row["true_terminal"] for row in sample_rows))
        self.assertTrue(any(row["life_loss_terminal"] for row in sample_rows))
        self.assertTrue(any(row["reward"] == 0 for row in sample_rows))
        self.assertTrue(any(row["reward"] > 0 for row in sample_rows))

    def test_terminal_and_nonterminal_contracts(self) -> None:
        sample_rows = [row for row in self.pytorch_rows if row.get("phase") == "bellman_sample"]
        terminal_rows = [row for row in sample_rows if row["terminal_flag"]]
        nonterminal_rows = [row for row in sample_rows if not row["terminal_flag"]]
        self.assertTrue(terminal_rows)
        self.assertTrue(nonterminal_rows)

        for row in terminal_rows:
            self.assertTrue(row["terminal_target_equals_reward"])
            self.assertAlmostEqual(float(row["discounted_continuation"]), 0.0, delta=FLOAT_TOL)
            self.assertAlmostEqual(float(row["bellman_target"]), float(row["reward"]), delta=FLOAT_TOL)
            self.assertEqual(float(row["continuation_mask"]), 0.0)

        for row in nonterminal_rows:
            self.assertTrue(row["nonterminal_gamma_applied"])
            self.assertEqual(float(row["continuation_mask"]), 1.0)
            expected = float(row["gamma"]) * float(row["max_next_q"])
            self.assertAlmostEqual(float(row["discounted_continuation"]), expected, delta=FLOAT_TOL)


if __name__ == "__main__":
    unittest.main()
