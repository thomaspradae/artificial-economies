from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local audit host may not have torch.
    torch = None  # type: ignore[assignment]

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if torch is not None:
    from atari.deepmind_faithful.network import DeepMindAtariQNetwork  # noqa: E402
else:
    DeepMindAtariQNetwork = None  # type: ignore[assignment]


def read_raw_tensor(model_dir: Path, layer: dict, field: str) -> np.ndarray:
    shape = [int(dim) for dim in layer[f"{field}_shape"]]
    return np.fromfile(model_dir / layer[f"{field}_file"], dtype=np.float32).reshape(shape)


def _load_old_qnetwork(atari_dir: Path):
    network_path = atari_dir / "network.py"
    if not network_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("stage5_old_network_test", network_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.QNetwork


def _load_weights(model, model_dir: Path, manifest: dict, faithful: bool) -> None:
    if faithful:
        mapping = {
            "conv1": ("conv1.weight", "conv1.bias"),
            "conv2": ("conv2.weight", "conv2.bias"),
            "conv3": ("conv3.weight", "conv3.bias"),
            "fc1": ("fc1.weight", "fc1.bias"),
            "fc2": ("fc2.weight", "fc2.bias"),
        }
    else:
        mapping = {
            "conv1": ("conv.0.weight", "conv.0.bias"),
            "conv2": ("conv.2.weight", "conv.2.bias"),
            "conv3": ("conv.4.weight", "conv.4.bias"),
            "fc1": ("fc.0.weight", "fc.0.bias"),
            "fc2": ("fc.2.weight", "fc.2.bias"),
        }
    state = model.state_dict()
    for layer in manifest["layers"]:
        weight_key, bias_key = mapping[layer["name"]]
        state[weight_key] = torch.from_numpy(read_raw_tensor(model_dir, layer, "weight").copy())
        state[bias_key] = torch.from_numpy(read_raw_tensor(model_dir, layer, "bias").copy())
    model.load_state_dict(state)


class DeepMindNetworkForwardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if torch is None:
            raise unittest.SkipTest("torch is not installed")
        stage5_dir = Path(os.getenv("AUDIT_STAGE5_DIR", "audit_outputs/stage5_learner"))
        cls.stage5_dir = stage5_dir
        cls.fixture_dir = stage5_dir / "learner_fixture"
        cls.model_dir = stage5_dir / "model_exchange"
        cls.oracle = stage5_dir / "deepmind_forward_layers.jsonl"
        required = [
            cls.fixture_dir / "states_uint8.bin",
            cls.fixture_dir / "manifest.json",
            cls.model_dir / "deepmind_model_manifest.json",
            cls.oracle,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise unittest.SkipTest(f"Stage 5A artifacts missing: {missing}")
        cls.fixture_manifest = json.loads((cls.fixture_dir / "manifest.json").read_text())
        cls.model_manifest = json.loads((cls.model_dir / "deepmind_model_manifest.json").read_text())
        tensors = {entry["name"]: entry for entry in cls.fixture_manifest["tensors"]}
        entry = tensors["states_uint8"]
        cls.states = np.fromfile(cls.fixture_dir / entry["path"], dtype=np.uint8).reshape(entry["shape"])
        rows = [json.loads(line) for line in cls.oracle.read_text().splitlines() if line.strip()]
        oracle_row = next(
            row
            for row in rows
            if row.get("phase") == "forward_layer"
            and row.get("batch_source") == "state"
            and row.get("layer") == "q_values"
        )
        cls.oracle_q = np.fromfile(stage5_dir / "forward_layers" / "deepmind" / oracle_row["tensor_file"], dtype=np.float32).reshape(
            oracle_row["shape"]
        )

    def test_faithful_network_matches_deepmind_oracle(self) -> None:
        model = DeepMindAtariQNetwork(int(self.model_manifest["action_count"]))
        self.assertEqual(tuple(model.conv1.padding), (1, 1))
        _load_weights(model, self.model_dir, self.model_manifest, faithful=True)
        model.eval()
        with torch.no_grad():
            q_values = model(torch.as_tensor(self.states, dtype=torch.float32)).cpu().numpy()
        self.assertLessEqual(float(np.max(np.abs(q_values - self.oracle_q))), 1e-7)
        np.testing.assert_array_equal(q_values.argmax(axis=1), self.oracle_q.argmax(axis=1))

    def test_old_padding_zero_network_fails_forward_gate(self) -> None:
        atari_dir = Path(os.getenv("ATARI_DIR", "."))
        old_cls = _load_old_qnetwork(atari_dir)
        if old_cls is None:
            raise unittest.SkipTest("old Atari QNetwork is not available")
        model = old_cls(int(self.model_manifest["action_count"]))
        first_conv = next(module for module in model.modules() if isinstance(module, torch.nn.Conv2d))
        self.assertEqual(tuple(first_conv.padding), (0, 0))
        _load_weights(model, self.model_dir, self.model_manifest, faithful=False)
        model.eval()
        with torch.no_grad():
            q_values = model(torch.as_tensor(self.states, dtype=torch.float32)).cpu().numpy()
        self.assertGreater(float(np.max(np.abs(q_values - self.oracle_q))), 1e-5)


if __name__ == "__main__":
    unittest.main()
