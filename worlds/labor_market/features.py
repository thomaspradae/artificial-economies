from __future__ import annotations

import numpy as np

from worlds.labor_market.env import LaborMarketWorld


def labor_market_obs_dim(n_employers: int) -> int:
    """Feature width for one learning worker."""
    return 6 + 3 * int(n_employers)


def _safe_scale(value: float, denominator: float) -> float:
    return 0.0 if denominator <= 0.0 else float(value) / float(denominator)


def encode_labor_market_observation(world: LaborMarketWorld, worker: int) -> np.ndarray:
    """Encode true worker preferences and last matching outcome for neural minds."""
    cfg = world.config
    obs = world.observations()[worker]
    values = world.worker_values[worker].astype(float)
    max_value = max(float(np.max(world.worker_values)), 1.0)
    ranks = np.empty(cfg.n_employers, dtype=float)
    for rank, employer in enumerate(world.worker_preferences[worker]):
        ranks[int(employer)] = float(rank)
    last = world.history[-1] if world.history else {}
    matches = last.get("matches", ())
    reported = last.get("reported_tops", ())
    matched = int(matches[worker]) if worker < len(matches) else -1
    last_report = int(reported[worker]) if worker < len(reported) else -1
    matched_one_hot = [1.0 if employer == matched else 0.0 for employer in range(cfg.n_employers)]
    features = np.asarray(
        [
            _safe_scale(float(worker), float(max(cfg.n_workers - 1, 1))),
            _safe_scale(float(obs[0]), float(max(cfg.n_employers - 1, 1))),
            _safe_scale(float(obs[1]), float(max(cfg.n_employers - 1, 1))),
            _safe_scale(float(world.round_idx), float(max(cfg.max_rounds - 1, 1))),
            1.0 if matched >= 0 else 0.0,
            *[_safe_scale(value, max_value) for value in values],
            *[_safe_scale(rank, float(max(cfg.n_employers - 1, 1))) for rank in ranks],
            *matched_one_hot,
            _safe_scale(float(last_report), float(max(cfg.n_employers - 1, 1))),
        ],
        dtype=np.float32,
    )
    if len(features) != labor_market_obs_dim(cfg.n_employers):
        raise ValueError("Labor Market feature width changed unexpectedly")
    return features


def structured_observations(world: LaborMarketWorld) -> list[np.ndarray]:
    return [encode_labor_market_observation(world, worker) for worker in range(world.config.n_workers)]
