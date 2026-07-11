from __future__ import annotations

import numpy as np

from arena_v0 import compute_static_benchmarks as _compute_static_benchmarks


def compute_static_benchmarks(price_grid: np.ndarray) -> dict[str, object]:
    """Grid-search one-shot Nash pairs and symmetric joint-profit price."""
    return _compute_static_benchmarks(price_grid)

