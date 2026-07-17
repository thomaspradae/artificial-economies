from __future__ import annotations

import numpy as np

from worlds.public_goods.env import N_ACTIONS, PublicGoodsWorld


def public_goods_obs_dim() -> int:
    """Fixed feature width for one Public Goods learner."""
    return 12


def _safe_scale(value: float, denominator: float) -> float:
    return 0.0 if denominator <= 0.0 else float(value) / float(denominator)


def encode_public_goods_observation(world: PublicGoodsWorld, agent_id: int) -> np.ndarray:
    """Encode public pool state, recent commons activity, and agent identity."""
    cfg = world.config
    tabular = world.observations()[agent_id]
    last = world.history[-1] if world.history else {}
    contribution_capacity = max(cfg.n_agents * 2.0 * cfg.contribution_unit, 1.0)
    extraction_capacity = max(cfg.n_agents * 2.0 * cfg.extraction_unit, 1.0)
    reward_scale = max(cfg.extraction_reward * 2.0 * cfg.extraction_unit, 1.0)
    features = np.asarray(
        [
            _safe_scale(world.pool_stock, cfg.pool_capacity),
            _safe_scale(float(tabular[0]), float(max(N_ACTIONS - 1, 1))),
            _safe_scale(float(tabular[1]), float(max(N_ACTIONS - 1, 1))),
            _safe_scale(float(agent_id), float(max(cfg.n_agents - 1, 1))),
            _safe_scale(float(world.round_idx), float(max(cfg.max_rounds - 1, 1))),
            np.tanh(float(world.cumulative_rewards[agent_id]) / reward_scale),
            _safe_scale(float(last.get("contribution_total", 0.0)), contribution_capacity),
            _safe_scale(float(last.get("extraction_total", 0.0)), extraction_capacity),
            float(last.get("collapse_rate", 0.0)),
            _safe_scale(float(last.get("contributor_count", 0.0)), float(max(cfg.n_agents, 1))),
            _safe_scale(float(last.get("extractor_count", 0.0)), float(max(cfg.n_agents, 1))),
            _safe_scale(float(last.get("reward_total", 0.0)), reward_scale * max(cfg.n_agents, 1)),
        ],
        dtype=np.float32,
    )
    if len(features) != public_goods_obs_dim():
        raise ValueError("Public Goods feature width changed unexpectedly")
    return features


def structured_observations(world: PublicGoodsWorld) -> list[np.ndarray]:
    return [encode_public_goods_observation(world, agent_id) for agent_id in range(world.config.n_agents)]
