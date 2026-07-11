from __future__ import annotations

from typing import Any


def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Manhattan distance between two grid cells."""
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Whether two grid cells are Manhattan-adjacent or co-located."""
    return manhattan_distance(a, b) <= 1


def within_trade_radius(a: tuple[int, int], b: tuple[int, int], trade_radius: int) -> bool:
    """Whether two grid cells are close enough for Resource Island trade matching."""
    return manhattan_distance(a, b) <= int(trade_radius)


def one_for_one_trade_offer(
    left_action: int,
    right_action: int,
    left_agent: int,
    right_agent: int,
    offer_food_for_wood_action: int,
    offer_wood_for_food_action: int,
) -> dict[str, Any] | None:
    """Return a 1-for-1 complementary trade state, or None if offers do not match."""
    if left_action == offer_food_for_wood_action and right_action == offer_wood_for_food_action:
        return {
            "phase": "pre_trade",
            "participants": (left_agent, right_agent),
            "food_giver": left_agent,
            "wood_giver": right_agent,
            "food_units": 1,
            "wood_units": 1,
            "allowed": True,
        }
    if left_action == offer_wood_for_food_action and right_action == offer_food_for_wood_action:
        return {
            "phase": "pre_trade",
            "participants": (left_agent, right_agent),
            "food_giver": right_agent,
            "wood_giver": left_agent,
            "food_units": 1,
            "wood_units": 1,
            "allowed": True,
        }
    return None
