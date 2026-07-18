from __future__ import annotations

import json
import math
import os
import unittest
from pathlib import Path
from typing import Any


FLOAT_TOL = 1e-7
THRESHOLD_DELTAS = [-2.0, -1.001, -1.0, -0.999, -0.5, 0.0, 0.5, 0.999, 1.0, 1.001, 2.0]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def close(left: float, right: float, tol: float = FLOAT_TOL) -> bool:
    return abs(float(left) - float(right)) <= tol


class LossSemanticsMatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        stage5c_dir = Path(os.getenv("AUDIT_STAGE5C_DIR", "audit_outputs/stage5_loss"))
        cls.stage5c_dir = stage5c_dir
        cls.pytorch_path = stage5c_dir / "pytorch_loss.jsonl"
        cls.deepmind_path = stage5c_dir / "deepmind_loss.jsonl"
        cls.compare_path = stage5c_dir / "loss_compare.txt"
        cls.fixture_path = stage5c_dir / "loss_fixture.json"
        cls.threshold_path = stage5c_dir / "loss_threshold_controls.tsv"
        required = [cls.pytorch_path, cls.deepmind_path, cls.compare_path, cls.fixture_path, cls.threshold_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise unittest.SkipTest(f"Stage 5C artifacts missing: {missing}")
        cls.pytorch_rows = read_jsonl(cls.pytorch_path)
        cls.deepmind_rows = read_jsonl(cls.deepmind_path)
        cls.compare_text = cls.compare_path.read_text(encoding="utf-8")
        cls.fixture = json.loads(cls.fixture_path.read_text(encoding="utf-8"))

    def test_stage5c_compare_passed(self) -> None:
        self.assertIn("MATCH", self.compare_text)
        self.assertEqual(len(self.pytorch_rows), len(self.deepmind_rows))

    def test_required_batches_present(self) -> None:
        batch_rows = [row for row in self.pytorch_rows if row.get("phase") == "loss_batch"]
        names = {row["batch_name"] for row in batch_rows}
        required = {
            "batch_1",
            "batch_4",
            "batch_32",
            "ordinary_nonterminal",
            "true_terminal",
            "life_loss_terminal",
            "zero_reward",
            "positive_reward",
            "threshold_all",
            "threshold_mixed_4",
            "threshold_duplicated",
            "threshold_batch_32",
        }
        self.assertTrue(required.issubset(names), sorted(required - names))
        self.assertTrue(any(name.startswith("threshold_single_") for name in names))
        self.assertTrue(any(name.startswith("real_single_") for name in names))

    def test_threshold_clipping_contract(self) -> None:
        rows = [
            row
            for row in self.pytorch_rows
            if row.get("phase") == "loss_sample" and row.get("batch_name") == "threshold_all"
        ]
        self.assertEqual(len(rows), len(THRESHOLD_DELTAS))
        for row, expected_delta in zip(rows, THRESHOLD_DELTAS):
            clipped = max(-1.0, min(1.0, expected_delta))
            self.assertTrue(close(row["raw_td_error"], expected_delta), row)
            self.assertTrue(close(row["clipped_td_error"], clipped), row)
            self.assertEqual(row["strictly_inside_clip_region"], abs(expected_delta) < 1.0)
            self.assertEqual(row["at_clip_threshold"], abs(expected_delta) == 1.0)
            self.assertTrue(close(row["selected_action_output_gradient"], clipped), row)

    def test_sparse_selected_action_output_gradient(self) -> None:
        sample_rows = [row for row in self.pytorch_rows if row.get("phase") == "loss_sample"]
        self.assertTrue(sample_rows)
        for row in sample_rows:
            gradient = row["output_gradient_row"]
            action = int(row["action"])
            self.assertEqual(len(gradient), int(self.fixture["action_count"]))
            for index, value in enumerate(gradient):
                if index == action:
                    self.assertTrue(close(value, row["clipped_td_error"]), row)
                else:
                    self.assertEqual(float(value), 0.0, row)
            self.assertTrue(row["unselected_actions_zero"], row)

    def test_deepmind_no_batch_mean_reduction(self) -> None:
        batches = {row["batch_name"]: row for row in self.pytorch_rows if row.get("phase") == "loss_batch"}
        all_batch = batches["threshold_all"]
        duplicate = batches["threshold_duplicated"]

        self.assertEqual(all_batch["deepmind_effective_reduction"], "sparse_clipped_delta_no_batch_mean")
        self.assertEqual(all_batch["deepmind_batch_normalization_factor"], 1.0)
        self.assertEqual(all_batch["scalar_loss_reported_by_deepmind"], "none")
        self.assertTrue(close(duplicate["huber_loss_sum_proxy"], 2.0 * float(all_batch["huber_loss_sum_proxy"])))
        self.assertTrue(close(duplicate["huber_loss_mean_proxy"], all_batch["huber_loss_mean_proxy"]))
        self.assertTrue(
            close(
                duplicate["output_gradient_tensor"]["abs_sum"],
                2.0 * float(all_batch["output_gradient_tensor"]["abs_sum"]),
            )
        )

    def test_pytorch_and_deepmind_effective_rows_match(self) -> None:
        exact_fields = {
            "phase",
            "step",
            "batch_name",
            "batch_kind",
            "batch_size",
            "batch_position",
            "sample_id",
            "sample_kind",
            "replay_index",
            "action",
            "action_one_based",
            "strictly_inside_clip_region",
            "at_clip_threshold",
            "output_gradient_nonzero_count",
            "unselected_actions_zero",
            "terminal_flag",
            "true_terminal",
            "life_loss_terminal",
            "deepmind_source_path",
            "scalar_loss_reported_by_deepmind",
            "sample_ids",
            "actions",
            "deepmind_batch_normalization_factor",
            "deepmind_effective_reduction",
        }
        numeric_fields = {
            "selected_q",
            "target",
            "raw_td_error",
            "pred_minus_target",
            "abs_td_error",
            "clip_delta",
            "clipped_td_error",
            "per_sample_huber_loss_proxy",
            "selected_action_output_gradient",
            "smooth_l1_mean_dloss_d_selected_q",
            "output_gradient_row",
            "selected_q_values",
            "targets",
            "raw_td_errors",
            "clipped_td_errors",
            "per_sample_huber_losses_proxy",
            "huber_loss_sum_proxy",
            "huber_loss_mean_proxy",
            "smooth_l1_mean_scalar_proxy",
            "selected_action_output_gradients",
        }
        for row_index, (pytorch, deepmind) in enumerate(zip(self.pytorch_rows, self.deepmind_rows)):
            for field in exact_fields & pytorch.keys() & deepmind.keys():
                self.assertEqual(pytorch[field], deepmind[field], f"row {row_index} field {field}")
            for field in numeric_fields & pytorch.keys() & deepmind.keys():
                self._assert_close_value(pytorch[field], deepmind[field], f"row {row_index} field {field}")

    def _assert_close_value(self, left: Any, right: Any, path: str) -> None:
        if isinstance(left, list):
            self.assertEqual(len(left), len(right), path)
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                self._assert_close_value(left_item, right_item, f"{path}[{index}]")
            return
        self.assertTrue(math.isfinite(float(left)), path)
        self.assertTrue(math.isfinite(float(right)), path)
        self.assertLessEqual(abs(float(left) - float(right)), FLOAT_TOL, path)


if __name__ == "__main__":
    unittest.main()
