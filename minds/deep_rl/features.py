from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _flatten_value(value: Any) -> list[float]:
    if value is None:
        return [0.0]
    if isinstance(value, (bool, np.bool_)):
        return [1.0 if value else 0.0]
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [float(value)]
    if isinstance(value, str):
        return [float(abs(hash(value)) % 10_000) / 10_000.0]
    if isinstance(value, Mapping):
        out: list[float] = []
        for key in sorted(value):
            out.extend(_flatten_value(value[key]))
        return out
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float).ravel().tolist()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_value(item))
        return out
    return [0.0]


def encode_observation(
    obs: Any,
    obs_dim: int | None = None,
    action_dim: int | None = None,
) -> np.ndarray:
    """Convert structured observations into a stable float vector.

    Pricing Arena states are two previous discrete actions; when `action_dim`
    is supplied those are encoded as two one-hot vectors. Other structured
    observations are recursively flattened, then padded/truncated if `obs_dim`
    is fixed by the caller.
    """
    if action_dim is not None and isinstance(obs, tuple) and len(obs) == 2:
        vector = np.zeros(2 * action_dim, dtype=float)
        for offset, item in enumerate(obs):
            index = int(item)
            if 0 <= index < action_dim:
                vector[offset * action_dim + index] = 1.0
    else:
        vector = np.asarray(_flatten_value(obs), dtype=float)

    if obs_dim is not None:
        if obs_dim < 1:
            raise ValueError("obs_dim must be positive")
        if len(vector) < obs_dim:
            vector = np.pad(vector, (0, obs_dim - len(vector)))
        elif len(vector) > obs_dim:
            vector = vector[:obs_dim]

    scale = max(1.0, float(np.max(np.abs(vector))) if vector.size else 1.0)
    return (vector / scale).astype(float, copy=False)


def infer_obs_dim(obs: Any, action_dim: int | None = None) -> int:
    """Feature dimension implied by one observation."""
    return int(len(encode_observation(obs, action_dim=action_dim)))


def one_hot(index: int, size: int) -> np.ndarray:
    """One-hot vector for a discrete action."""
    out = np.zeros(size, dtype=float)
    if 0 <= index < size:
        out[index] = 1.0
    return out
