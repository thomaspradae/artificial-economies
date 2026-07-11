from __future__ import annotations

import math
from typing import Callable, Iterable

import numpy as np


def finite_mean(values: Iterable[float]) -> float:
    """Mean over finite values only; returns NaN if no finite values exist."""
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan")
    return float(np.mean(array))


def average_price(price_history: Iterable[float]) -> float:
    """Average realized price over a run or window."""
    return finite_mean(price_history)


def welfare(consumer_surplus: Iterable[float], profits: Iterable[float]) -> float:
    """Mean total welfare proxy: consumer surplus plus producer profits."""
    surplus = np.asarray(list(consumer_surplus), dtype=float)
    producer_profit = np.asarray(list(profits), dtype=float)
    return finite_mean(surplus + producer_profit)


def consumer_surplus_proxy(prices: Iterable[float], demand_fn: Callable[[np.ndarray], float]) -> float:
    """World-supplied inclusive-value or surplus proxy evaluated at prices."""
    return float(demand_fn(np.asarray(list(prices), dtype=float)))


def collusion_index(avg_price: float, nash_price: float | None, monopoly_price: float) -> float:
    """Price-normalized collusion proxy: 0 at Nash price, 1 at symmetric joint-profit price.

    This is not the Calvano et al. profit-normalized index
    (profit - Nash profit) / (monopoly profit - Nash profit); use
    profit_collusion_index when profit benchmarks are available.
    """
    if nash_price is None or abs(monopoly_price - nash_price) < 1e-12:
        return float("nan")
    scaled = (avg_price - nash_price) / (monopoly_price - nash_price)
    return float(np.clip(scaled, 0.0, 1.0))


def profit_collusion_index(profit: float, nash_profit: float | None, monopoly_profit: float) -> float:
    """Calvano-style profit-normalized collusion index.

    Formula: (observed profit - Nash profit) / (monopoly profit - Nash profit).
    """
    if nash_profit is None or abs(monopoly_profit - nash_profit) < 1e-12:
        return float("nan")
    scaled = (profit - nash_profit) / (monopoly_profit - nash_profit)
    return float(np.clip(scaled, 0.0, 1.0))


def gini(values: Iterable[float]) -> float:
    """Gini coefficient for a non-negative distribution; NaN for invalid input."""
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan")
    if np.any(array < 0.0):
        return float("nan")
    total = float(np.sum(array))
    if math.isclose(total, 0.0):
        return 0.0
    sorted_values = np.sort(array)
    n = len(sorted_values)
    weights = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(weights * sorted_values)) / (n * total) - (n + 1.0) / n)


def exploitability(adversary_profit: float, baseline_profit: float) -> float:
    """Extra profit earned by an adversary relative to the original agent."""
    return float(adversary_profit - baseline_profit)


def victim_loss(original_profit: float, frozen_profit: float) -> float:
    """Profit lost by a frozen victim when facing the trained adversary."""
    return float(original_profit - frozen_profit)


def welfare_damage(original_welfare: float, exploited_welfare: float) -> float:
    """Welfare lost after replacing an original agent with the adversary."""
    return float(original_welfare - exploited_welfare)


def survival_rate(alive_history: Iterable[Iterable[bool] | bool]) -> float:
    """Fraction of observed agent-time in which agents are alive."""
    values: list[float] = []
    for entry in alive_history:
        if isinstance(entry, (bool, np.bool_)):
            values.append(float(entry))
        else:
            values.extend(float(value) for value in entry)
    if not values:
        return float("nan")
    return finite_mean(values)


def specialization_index(resource_totals: Iterable[Iterable[float]]) -> float:
    """Mean normalized concentration of each agent's gathered resources."""
    rows = np.asarray(list(resource_totals), dtype=float)
    if rows.size == 0:
        return float("nan")
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.shape[1] < 2:
        return 0.0
    scores: list[float] = []
    uniform = np.full(rows.shape[1], 1.0 / rows.shape[1])
    normalizer = 2.0 * (1.0 - 1.0 / rows.shape[1])
    for row in rows:
        row = row[np.isfinite(row)]
        if len(row) != rows.shape[1] or np.any(row < 0.0):
            continue
        total = float(np.sum(row))
        if math.isclose(total, 0.0):
            scores.append(0.0)
            continue
        shares = row / total
        scores.append(float(np.sum(np.abs(shares - uniform)) / normalizer))
    return finite_mean(scores)


def stability(values: Iterable[float]) -> float:
    """Inverse volatility score: 1/(1 + finite-series standard deviation)."""
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan")
    return float(1.0 / (1.0 + np.std(array)))


def robustness_under_shock(baseline_metric: float, shocked_metric: float) -> float:
    """Performance retained under shock, measured as shocked divided by baseline."""
    if not math.isfinite(baseline_metric) or math.isclose(baseline_metric, 0.0):
        return float("nan")
    return float(shocked_metric / baseline_metric)


def resource_sustainability(current_stock: float, total_introduced: float) -> float:
    """Resource stock retained: current map stock divided by total introduced stock."""
    if not math.isfinite(total_introduced) or total_introduced <= 0.0:
        return float("nan")
    return float(current_stock / total_introduced)
