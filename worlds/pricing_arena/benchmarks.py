from __future__ import annotations

import numpy as np

from arena_v0 import compute_static_benchmarks as _compute_static_benchmarks, logsumexp


def _demand(prices: np.ndarray, *, market_size: float, alpha: float, tau: float, quality: np.ndarray) -> np.ndarray:
    utilities = quality - alpha * prices
    logits = np.concatenate(([0.0], utilities)) / tau
    probabilities = np.exp(logits - logsumexp(logits))
    return market_size * probabilities[1:]


def compute_static_benchmarks(
    price_grid: np.ndarray,
    *,
    n_firms: int = 2,
    cost: float = 1.0,
    market_size: float = 100.0,
    alpha: float = 0.9,
    tau: float = 0.7,
    quality: np.ndarray | None = None,
) -> dict[str, object]:
    """Grid-search static benchmarks for symmetric N-firm logit pricing.

    For the default duopoly configuration this delegates to the legacy helper
    so Phase 0 parity stays byte-identical. For N>2 it reports symmetric Nash
    and symmetric joint-profit reference prices over the same discrete grid.
    """
    if n_firms == 2 and quality is None and cost == 1.0 and market_size == 100.0 and alpha == 0.9 and tau == 0.7:
        return _compute_static_benchmarks(price_grid)
    if n_firms < 2:
        raise ValueError("n_firms must be at least 2")
    qualities = np.full(n_firms, 8.0, dtype=float) if quality is None else np.asarray(quality, dtype=float)
    if len(qualities) != n_firms:
        raise ValueError("quality length must match n_firms")

    symmetric_profits: list[float] = []
    for price in price_grid:
        prices = np.full(n_firms, float(price), dtype=float)
        quantities = _demand(prices, market_size=market_size, alpha=alpha, tau=tau, quality=qualities)
        symmetric_profits.append(float(np.sum((prices - cost) * quantities)))
    monopoly_index = int(np.argmax(symmetric_profits))
    monopoly_price = float(price_grid[monopoly_index])
    monopoly_profit = float(symmetric_profits[monopoly_index])

    nash_prices: list[float] = []
    for candidate in price_grid:
        profile = np.full(n_firms, float(candidate), dtype=float)
        base_quantities = _demand(profile, market_size=market_size, alpha=alpha, tau=tau, quality=qualities)
        base_profit = float((candidate - cost) * base_quantities[0])
        best_response_profit = -float("inf")
        for deviation in price_grid:
            deviated = profile.copy()
            deviated[0] = float(deviation)
            quantities = _demand(deviated, market_size=market_size, alpha=alpha, tau=tau, quality=qualities)
            best_response_profit = max(best_response_profit, float((deviation - cost) * quantities[0]))
        if base_profit >= best_response_profit - 1e-9:
            nash_prices.append(float(candidate))

    nash_price = float(nash_prices[0]) if nash_prices else None
    if nash_price is None:
        nash_profit = None
    else:
        prices = np.full(n_firms, nash_price, dtype=float)
        quantities = _demand(prices, market_size=market_size, alpha=alpha, tau=tau, quality=qualities)
        nash_profit = float(np.sum((prices - cost) * quantities))

    nash_profiles = [(price,) * n_firms for price in nash_prices]
    return {
        "nash_price": nash_price,
        "nash_pairs": nash_profiles,
        "nash_price_pairs": nash_profiles,
        "monopoly_price": monopoly_price,
        "monopoly_profit": monopoly_profit,
        "nash_profit": nash_profit,
        "n_firms": n_firms,
    }
