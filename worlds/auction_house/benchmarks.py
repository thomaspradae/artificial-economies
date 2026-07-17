from __future__ import annotations

from itertools import product
from typing import Callable, Literal

import numpy as np


AuctionFormat = Literal["first_price", "second_price", "clock"]


def nearest_bid(value: float, bid_grid: tuple[float, ...], *, prefer_floor: bool = False) -> float:
    """Return the nearest bid-grid value, optionally flooring ties/overshoots."""
    grid = np.asarray(bid_grid, dtype=float)
    if grid.ndim != 1 or len(grid) == 0:
        raise ValueError("bid_grid must be a non-empty one-dimensional grid")
    if prefer_floor:
        feasible = grid[grid <= float(value) + 1e-12]
        if len(feasible) == 0:
            return float(grid[0])
        return float(feasible[-1])
    return float(grid[int(np.argmin(np.abs(grid - float(value))))])


def truthful_bid(value: float, bid_grid: tuple[float, ...] | None = None) -> float:
    """Second-price private-value benchmark bid."""
    if bid_grid is None:
        return float(value)
    return nearest_bid(float(value), bid_grid)


def truthful_bid_benchmark(
    valuations: tuple[float, ...] | list[float] | np.ndarray,
    bid_grid: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    """Truthful second-price benchmark bids for a valuation profile."""
    return tuple(truthful_bid(float(value), bid_grid) for value in valuations)


def first_price_equilibrium_bid(
    value: float,
    n_bidders: int = 2,
    valuation_low: float = 0.0,
    bid_grid: tuple[float, ...] | None = None,
) -> float:
    """Symmetric first-price IPV uniform benchmark.

    For risk-neutral bidders with values uniformly distributed on
    [valuation_low, valuation_high], the continuous equilibrium is
    b(v) = v - (v - valuation_low) / n_bidders.
    """
    if n_bidders < 2:
        raise ValueError("n_bidders must be at least 2")
    value = float(value)
    valuation_low = float(valuation_low)
    shaded = value - (value - valuation_low) / float(n_bidders)
    if bid_grid is None:
        return float(shaded)
    return nearest_bid(shaded, bid_grid, prefer_floor=True)


def first_price_bid_benchmark(
    valuations: tuple[float, ...] | list[float] | np.ndarray,
    n_bidders: int,
    valuation_low: float = 0.0,
    bid_grid: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    """First-price bid-shading benchmark bids for a valuation profile."""
    return tuple(
        first_price_equilibrium_bid(
            float(value),
            n_bidders=n_bidders,
            valuation_low=valuation_low,
            bid_grid=bid_grid,
        )
        for value in valuations
    )


def allocate_winner(bids: tuple[float, ...] | np.ndarray, reserve_price: float = 0.0) -> int | None:
    """Allocate to the lowest-id highest eligible bidder, or no sale."""
    bids_array = np.asarray(bids, dtype=float)
    eligible = np.flatnonzero(bids_array >= float(reserve_price) - 1e-12)
    if len(eligible) == 0:
        return None
    eligible_bids = bids_array[eligible]
    max_bid = float(np.max(eligible_bids))
    tied = eligible[eligible_bids == max_bid]
    return int(tied[0])


def payment_for_winner(
    bids: tuple[float, ...] | np.ndarray,
    winner: int | None,
    auction_format: AuctionFormat,
    reserve_price: float = 0.0,
) -> float:
    """Compute first-price or second-price payment for a sealed-bid auction."""
    if winner is None:
        return 0.0
    bids_array = np.asarray(bids, dtype=float)
    if auction_format == "first_price":
        return float(bids_array[winner])
    if auction_format not in ("second_price", "clock"):
        raise ValueError("auction_format must be first_price, second_price, or clock")
    other_bids = np.delete(bids_array, winner)
    second_bid = float(np.max(other_bids)) if len(other_bids) else 0.0
    return float(max(float(reserve_price), second_bid))


def bidder_utility(
    valuations: tuple[float, ...] | np.ndarray,
    bids: tuple[float, ...] | np.ndarray,
    bidder: int,
    auction_format: AuctionFormat,
    reserve_price: float = 0.0,
) -> float:
    """Realized surplus for one bidder under deterministic tie-breaking."""
    valuations_array = np.asarray(valuations, dtype=float)
    winner = allocate_winner(bids, reserve_price=reserve_price)
    if winner != int(bidder):
        return 0.0
    payment = payment_for_winner(
        bids,
        winner,
        auction_format=auction_format,
        reserve_price=reserve_price,
    )
    return float(valuations_array[int(bidder)] - payment)


def ex_post_bidder_regret(
    valuations: tuple[float, ...] | np.ndarray,
    bids: tuple[float, ...] | np.ndarray,
    bidder: int,
    bid_grid: tuple[float, ...],
    auction_format: AuctionFormat,
    reserve_price: float = 0.0,
) -> float:
    """Best unilateral grid-bid gain holding other bids fixed."""
    bids_array = np.asarray(bids, dtype=float)
    current = bidder_utility(
        valuations,
        bids_array,
        bidder,
        auction_format=auction_format,
        reserve_price=reserve_price,
    )
    best = current
    for candidate in bid_grid:
        trial = bids_array.copy()
        trial[int(bidder)] = float(candidate)
        best = max(
            best,
            bidder_utility(
                valuations,
                trial,
                bidder,
                auction_format=auction_format,
                reserve_price=reserve_price,
            ),
        )
    return float(max(0.0, best - current))


def auction_outcome(
    valuations: tuple[float, ...] | np.ndarray,
    bids: tuple[float, ...] | np.ndarray,
    auction_format: AuctionFormat,
    reserve_price: float = 0.0,
) -> dict[str, float | int | None]:
    """Resolve a one-item sealed-bid auction into economic accounting terms."""
    valuations_array = np.asarray(valuations, dtype=float)
    winner = allocate_winner(bids, reserve_price=reserve_price)
    payment = payment_for_winner(
        bids,
        winner,
        auction_format=auction_format,
        reserve_price=reserve_price,
    )
    max_value = float(np.max(valuations_array))
    efficient_winner: int | None = None
    if max_value >= float(reserve_price) - 1e-12:
        efficient_winner = int(np.flatnonzero(valuations_array == max_value)[0])
    welfare = 0.0 if winner is None else float(valuations_array[winner])
    max_possible_welfare = max_value if efficient_winner is not None else 0.0
    efficient = (winner == efficient_winner) or (winner is None and efficient_winner is None)
    welfare_efficiency = 1.0 if max_possible_welfare <= 1e-12 else welfare / max_possible_welfare
    return {
        "winner": winner,
        "payment": float(payment),
        "revenue": float(payment),
        "welfare": float(welfare),
        "max_possible_welfare": float(max_possible_welfare),
        "efficient_winner": efficient_winner,
        "allocative_efficiency": float(efficient),
        "welfare_efficiency": float(welfare_efficiency),
    }


def expected_outcome_over_grid(
    valuation_grid: tuple[float, ...],
    n_bidders: int,
    bid_strategy: Callable[[float], float],
    auction_format: AuctionFormat,
    reserve_price: float = 0.0,
) -> dict[str, float]:
    """Enumerate a discrete independent-private-values benchmark exactly."""
    if n_bidders < 2:
        raise ValueError("n_bidders must be at least 2")
    profiles = list(product(tuple(float(v) for v in valuation_grid), repeat=n_bidders))
    if not profiles:
        raise ValueError("valuation_grid must be non-empty")
    totals = {
        "revenue": 0.0,
        "bidder_surplus": 0.0,
        "welfare": 0.0,
        "max_possible_welfare": 0.0,
        "allocative_efficiency": 0.0,
        "welfare_efficiency": 0.0,
    }
    for valuations in profiles:
        bids = tuple(float(bid_strategy(value)) for value in valuations)
        outcome = auction_outcome(
            valuations,
            bids,
            auction_format=auction_format,
            reserve_price=reserve_price,
        )
        winner = outcome["winner"]
        payment = float(outcome["payment"])
        bidder_surplus = 0.0 if winner is None else float(valuations[int(winner)] - payment)
        totals["revenue"] += float(outcome["revenue"])
        totals["bidder_surplus"] += bidder_surplus
        totals["welfare"] += float(outcome["welfare"])
        totals["max_possible_welfare"] += float(outcome["max_possible_welfare"])
        totals["allocative_efficiency"] += float(outcome["allocative_efficiency"])
        totals["welfare_efficiency"] += float(outcome["welfare_efficiency"])
    count = float(len(profiles))
    return {key: value / count for key, value in totals.items()}
