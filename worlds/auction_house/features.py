from __future__ import annotations

import numpy as np

from worlds.auction_house.env import AuctionHouseWorld


AUCTION_FORMATS = ("first_price", "second_price", "clock")


def auction_house_obs_dim() -> int:
    """Fixed feature width for one bidder in Auction House."""
    return 11


def _safe_scale(value: float, denominator: float) -> float:
    return 0.0 if denominator <= 0.0 else float(value) / float(denominator)


def encode_auction_house_observation(world: AuctionHouseWorld, bidder_id: int) -> np.ndarray:
    """Encode bidder-private Auction House state for structured neural minds."""
    cfg = world.config
    value_bins = world.observations()
    observed_bin = int(value_bins[bidder_id][0])
    max_value = max(float(max(cfg.valuation_grid)), 1.0)
    max_bid = max(float(max(cfg.bid_grid)), 1.0)
    last_info = world.history[-1] if world.history else {}
    last_bids = last_info.get("bids", ())
    last_bid = float(last_bids[bidder_id]) if bidder_id < len(last_bids) else 0.0
    last_winner = int(last_info.get("winner", -1.0)) if last_info else -1
    format_one_hot = [1.0 if cfg.auction_format == item else 0.0 for item in AUCTION_FORMATS]
    denom_bins = max(len(cfg.valuation_grid) - 1, 1)
    features = np.asarray(
        [
            _safe_scale(float(world.valuations[bidder_id]), max_value),
            _safe_scale(float(observed_bin), float(denom_bins)),
            _safe_scale(float(cfg.valuation_grid[observed_bin]), max_value),
            _safe_scale(float(world.round_idx), float(max(cfg.max_rounds - 1, 1))),
            _safe_scale(last_bid, max_bid),
            1.0 if last_winner == bidder_id else 0.0,
            _safe_scale(float(last_info.get("payment", 0.0)), max_bid),
            _safe_scale(float(cfg.reserve_price), max_bid),
            *format_one_hot,
        ],
        dtype=np.float32,
    )
    if len(features) != auction_house_obs_dim():
        raise ValueError("Auction House feature width changed unexpectedly")
    return features


def structured_observations(world: AuctionHouseWorld) -> list[np.ndarray]:
    return [encode_auction_house_observation(world, bidder_id) for bidder_id in range(world.config.n_bidders)]
