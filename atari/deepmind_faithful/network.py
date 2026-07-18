from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn


class DeepMindAtariQNetwork(nn.Module):
    """DeepMind Nature Atari DQN convnet with the released Torch7 padding.

    This intentionally does not replace the historical local ``QNetwork`` class.
    The confirmed architectural difference is the first convolution's zero
    padding of one pixel.
    """

    def __init__(self, num_actions: int):
        super().__init__()
        self.num_actions = int(num_actions)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=1, padding_mode="zeros")
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, self.num_actions)

    def forward_layers(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        layers: OrderedDict[str, torch.Tensor] = OrderedDict()
        normalized = x / 255.0
        layers["input"] = normalized
        z = self.conv1(normalized)
        layers["conv1_pre"] = z
        z = self.relu1(z)
        layers["conv1_post"] = z
        z = self.conv2(z)
        layers["conv2_pre"] = z
        z = self.relu2(z)
        layers["conv2_post"] = z
        z = self.conv3(z)
        layers["conv3_pre"] = z
        z = self.relu3(z)
        layers["conv3_post"] = z
        z = torch.flatten(z, start_dim=1)
        layers["flatten"] = z
        z = self.fc1(z)
        layers["fc1_pre"] = z
        z = self.relu4(z)
        layers["fc1_post"] = z
        z = self.fc2(z)
        layers["q_values"] = z
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_layers(x)["q_values"]

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "input_scaling": "float32_0_255_divided_by_255_inside_forward",
            "input_shape": [4, 84, 84],
            "flattened_feature_dim": 64 * 7 * 7,
            "action_count": self.num_actions,
            "layers": {
                "conv1": {
                    "in_channels": 4,
                    "out_channels": 32,
                    "kernel": [8, 8],
                    "stride": [4, 4],
                    "padding": [1, 1],
                    "padding_mode": "zeros",
                    "activation": "relu1",
                },
                "conv2": {
                    "in_channels": 32,
                    "out_channels": 64,
                    "kernel": [4, 4],
                    "stride": [2, 2],
                    "padding": [0, 0],
                    "padding_mode": "zeros",
                    "activation": "relu2",
                },
                "conv3": {
                    "in_channels": 64,
                    "out_channels": 64,
                    "kernel": [3, 3],
                    "stride": [1, 1],
                    "padding": [0, 0],
                    "padding_mode": "zeros",
                    "activation": "relu3",
                },
                "fc1": {
                    "in_features": 64 * 7 * 7,
                    "out_features": 512,
                    "activation": "relu4",
                },
                "fc2": {
                    "in_features": 512,
                    "out_features": self.num_actions,
                    "activation": None,
                },
            },
            "activation_order": [
                "conv1_pre",
                "conv1_post",
                "conv2_pre",
                "conv2_post",
                "conv3_pre",
                "conv3_post",
                "flatten",
                "fc1_pre",
                "fc1_post",
                "q_values",
            ],
        }


if __name__ == "__main__":
    print(json.dumps(DeepMindAtariQNetwork(4).architecture_manifest(), indent=2, sort_keys=True))
