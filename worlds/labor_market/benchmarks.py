from __future__ import annotations

import numpy as np


def preference_order(values: np.ndarray, axis: int = 1) -> np.ndarray:
    """Descending preference order for each row of a value matrix."""
    array = np.asarray(values, dtype=float)
    if axis != 1:
        raise ValueError("only row-wise preference order is supported")
    return np.argsort(-array, axis=1)


def rank_matrix(preferences: np.ndarray, n_items: int) -> np.ndarray:
    """Convert preference order rows into rank lookup rows."""
    prefs = np.asarray(preferences, dtype=int)
    ranks = np.full((prefs.shape[0], n_items), n_items + 1, dtype=int)
    for owner, row in enumerate(prefs):
        for rank, item in enumerate(row):
            ranks[owner, int(item)] = rank
    return ranks


def deferred_acceptance(worker_preferences: np.ndarray, employer_preferences: np.ndarray) -> np.ndarray:
    """Worker-proposing deferred acceptance.

    Returns an array of length n_workers where each entry is the matched
    employer id or -1.
    """
    worker_prefs = np.asarray(worker_preferences, dtype=int)
    employer_prefs = np.asarray(employer_preferences, dtype=int)
    n_workers = worker_prefs.shape[0]
    n_employers = employer_prefs.shape[0]
    employer_ranks = rank_matrix(employer_prefs, n_workers)
    next_proposal = np.zeros(n_workers, dtype=int)
    worker_match = np.full(n_workers, -1, dtype=int)
    employer_match = np.full(n_employers, -1, dtype=int)

    while True:
        proposals: dict[int, list[int]] = {employer: [] for employer in range(n_employers)}
        active = False
        for worker in range(n_workers):
            if worker_match[worker] != -1 or next_proposal[worker] >= worker_prefs.shape[1]:
                continue
            employer = int(worker_prefs[worker, next_proposal[worker]])
            next_proposal[worker] += 1
            if 0 <= employer < n_employers:
                proposals[employer].append(worker)
                active = True
        if not active:
            break
        for employer, workers in proposals.items():
            if employer_match[employer] != -1:
                workers.append(int(employer_match[employer]))
            if not workers:
                continue
            best = min(workers, key=lambda worker: employer_ranks[employer, worker])
            employer_match[employer] = best
            worker_match[best] = employer
            for worker in workers:
                if worker != best and worker_match[worker] == employer:
                    worker_match[worker] = -1
    return worker_match


def blocking_pairs(
    matches: np.ndarray,
    worker_preferences: np.ndarray,
    employer_preferences: np.ndarray,
) -> list[tuple[int, int]]:
    """Return true-preference blocking pairs for a matching."""
    match = np.asarray(matches, dtype=int)
    worker_prefs = np.asarray(worker_preferences, dtype=int)
    employer_prefs = np.asarray(employer_preferences, dtype=int)
    n_workers = worker_prefs.shape[0]
    n_employers = employer_prefs.shape[0]
    worker_ranks = rank_matrix(worker_prefs, n_employers)
    employer_ranks = rank_matrix(employer_prefs, n_workers)
    employer_to_worker = {int(employer): int(worker) for worker, employer in enumerate(match) if employer >= 0}
    pairs: list[tuple[int, int]] = []
    for worker in range(n_workers):
        current_employer = int(match[worker])
        current_worker_rank = worker_ranks[worker, current_employer] if current_employer >= 0 else n_employers + 1
        for employer in range(n_employers):
            if worker_ranks[worker, employer] >= current_worker_rank:
                continue
            incumbent = employer_to_worker.get(employer)
            if incumbent is None or employer_ranks[employer, worker] < employer_ranks[employer, incumbent]:
                pairs.append((worker, employer))
    return pairs


def reported_preferences_from_top(true_preferences: np.ndarray, reported_tops: np.ndarray) -> np.ndarray:
    """Build full reported preference lists from one reported top choice."""
    true_prefs = np.asarray(true_preferences, dtype=int)
    tops = np.asarray(reported_tops, dtype=int)
    reported = np.zeros_like(true_prefs)
    for worker, top in enumerate(tops):
        remaining = [int(item) for item in true_prefs[worker] if int(item) != int(top)]
        reported[worker] = np.asarray([int(top), *remaining], dtype=int)
    return reported


def truthful_matching(worker_values: np.ndarray, employer_values: np.ndarray) -> dict[str, object]:
    """Truthful worker-proposing DA benchmark."""
    worker_prefs = preference_order(worker_values)
    employer_prefs = preference_order(employer_values)
    matches = deferred_acceptance(worker_prefs, employer_prefs)
    return {
        "matches": matches,
        "worker_preferences": worker_prefs,
        "employer_preferences": employer_prefs,
        "blocking_pairs": blocking_pairs(matches, worker_prefs, employer_prefs),
    }


def matching_welfare(
    matches: np.ndarray,
    worker_values: np.ndarray,
    employer_values: np.ndarray,
) -> dict[str, float]:
    """Payoff accounting for a fixed matching."""
    match = np.asarray(matches, dtype=int)
    worker_array = np.asarray(worker_values, dtype=float)
    employer_array = np.asarray(employer_values, dtype=float)
    worker_welfare = 0.0
    employer_welfare = 0.0
    matched = 0
    for worker, employer in enumerate(match):
        if employer < 0:
            continue
        matched += 1
        worker_welfare += float(worker_array[worker, employer])
        employer_welfare += float(employer_array[employer, worker])
    return {
        "worker_welfare": worker_welfare,
        "employer_welfare": employer_welfare,
        "total_welfare": worker_welfare + employer_welfare,
        "match_rate": float(matched / len(match)) if len(match) else 0.0,
    }


def best_worker_report_gains(worker_values: np.ndarray, employer_values: np.ndarray) -> np.ndarray:
    """Best gain from unilateral top-report deviations under worker-proposing DA.

    For the proposing side of deferred acceptance this should be zero up to
    numerical precision when other workers report truthfully.
    """
    worker_array = np.asarray(worker_values, dtype=float)
    employer_array = np.asarray(employer_values, dtype=float)
    worker_prefs = preference_order(worker_array)
    employer_prefs = preference_order(employer_array)
    truthful_tops = worker_prefs[:, 0]
    truthful_reports = reported_preferences_from_top(worker_prefs, truthful_tops)
    truthful_matches = deferred_acceptance(truthful_reports, employer_prefs)
    truthful_payoffs = np.zeros(worker_array.shape[0], dtype=float)
    for worker, employer in enumerate(truthful_matches):
        if employer >= 0:
            truthful_payoffs[worker] = worker_array[worker, employer]

    gains = np.zeros(worker_array.shape[0], dtype=float)
    for worker in range(worker_array.shape[0]):
        best = truthful_payoffs[worker]
        for report in range(worker_array.shape[1]):
            tops = truthful_tops.copy()
            tops[worker] = report
            reported = reported_preferences_from_top(worker_prefs, tops)
            matches = deferred_acceptance(reported, employer_prefs)
            employer = int(matches[worker])
            payoff = 0.0 if employer < 0 else float(worker_array[worker, employer])
            best = max(best, payoff)
        gains[worker] = best - truthful_payoffs[worker]
    return gains


def canonical_matching_cases() -> dict[str, dict[str, object]]:
    """Fixed profiles used as regression benchmarks for matching economics."""
    stable_worker_values = np.asarray([[3.0, 1.0], [1.0, 3.0]], dtype=float)
    stable_employer_values = np.asarray([[3.0, 1.0], [1.0, 3.0]], dtype=float)
    unstable_forced_matches = np.asarray([1, 0], dtype=int)
    contested_worker_values = np.asarray(
        [
            [4.0, 3.0, 1.0],
            [4.0, 2.0, 1.0],
            [1.0, 3.0, 4.0],
        ],
        dtype=float,
    )
    contested_employer_values = np.asarray(
        [
            [2.0, 4.0, 1.0],
            [4.0, 1.0, 3.0],
            [1.0, 2.0, 4.0],
        ],
        dtype=float,
    )
    return {
        "stable_truthful_2x2": {
            "worker_values": stable_worker_values,
            "employer_values": stable_employer_values,
            "expected_truthful_matches": np.asarray([0, 1], dtype=int),
            "expected_blocking_pairs": [],
        },
        "unstable_forced_2x2": {
            "worker_values": stable_worker_values,
            "employer_values": stable_employer_values,
            "forced_matches": unstable_forced_matches,
            "expected_blocking_pairs": [(0, 0), (1, 1)],
        },
        "contested_strategyproof_3x3": {
            "worker_values": contested_worker_values,
            "employer_values": contested_employer_values,
            "expected_max_worker_report_gain": 0.0,
        },
    }
