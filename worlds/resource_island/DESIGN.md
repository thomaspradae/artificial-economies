# Resource Island Design Lock

Resource Island is the first non-pricing world. Its purpose is to test whether the Phase 0 `World`/`Agent`/`Institution` abstraction survives a spatial, inventory-based economy instead of another two-firm price game.

## Agents

- Agents live on a bounded square grid. The world is not a torus.
- Agents have position, energy, food inventory, wood inventory, alive/dead state, and cumulative gathered-resource counters.
- Dead agents stay in the world state but are inactive and receive zero reward after death.

## Observation

The world exposes two observation formats:

- Structured state in `render_state()` and `info`, containing positions, inventories, resource maps, alive flags, trade counts, and resource totals.
- Tabular observation for `QLearningMind`: `(energy_bin, local_resource_bin, inventory_imbalance_bin)`, where all integers are in `[0, n_actions - 1]`.
- `inventory_imbalance_bin` is `0` for food surplus, midpoint for balanced inventory, and `n_actions - 1` for wood surplus.

The tabular observation intentionally preserves the existing `QLearningMind` class and tabular update rule while making trade-relevant inventory imbalance observable. Richer minds can later use structured observations without changing this v0 tabular proof.

## Actions

The v0 discrete action set has 8 actions:

```text
0 stay
1 move_up
2 move_down
3 move_left
4 move_right
5 gather
6 offer_food_for_wood
7 offer_wood_for_food
```

Movement is simultaneous. If multiple agents propose the same destination, all agents involved in that destination conflict remain in their original cells. Grid boundaries are hard walls; a move into a wall leaves the agent in place.

## Resources

There are two resource types:

- `food`
- `wood`

Cells hold integer resource stocks by type. `gather` takes one unit from the current cell. If both food and wood are present, food is gathered first; this deterministic priority avoids hidden random tie-breaking in tests.

Resources regenerate stochastically after each step up to per-cell capacity. A resource spawn adds one unit to a random cell/type with probability `resource_spawn_probability`.

V1 hardening adds deterministic resource-layout options:

- `random`: original v0 random placement.
- `contested`: scarce resources are clustered around the island center so property claims face repeated access pressure.
- `split`: food and wood are placed in opposite corners for specialization/trade ablations.

## Energy, Starvation, And Death

Each active action has an energy cost:

- stay: low cost
- movement: movement cost
- gather: gather cost
- trade offer: trade cost

After costs are paid, an agent with low energy automatically consumes one food unit if available, converting it into energy. If energy remains at or below zero, the agent dies and receives a death penalty on that step.

## Trading

Trading is one-shot and simultaneous:

- `offer_food_for_wood` means "give 1 food, receive 1 wood."
- `offer_wood_for_food` means "give 1 wood, receive 1 food."
- A trade attempt is any within-radius pair where at least one agent posts a trade offer.
- A one-for-one trade clears when the posted offer is inventory-valid: the food giver has at least 1 food and the wood giver has at least 1 wood.
- The default `trade_radius` spans the whole island, making v0 trade a simple matching-market institution. Set `trade_radius=1` for strict adjacency-only trade ablations.
- This is a deliberate v0 simplification, not an emergent result: movement and gathering remain spatial, but trade itself is centralized by default so the trade/reputation institutions are mechanically exercisable. Spatially constrained trade should be reported separately with `trade_radius=1` or other finite radii before making claims about spatial trade institutions.
- V1 can set `trade_food_units` and `trade_wood_units` above 1, creating unequal exchange ratios. This is required before `trade_price_controls` can be economically evaluated rather than only mechanically tested.
- Each agent participates in at most one trade per step.

There is no multi-round bargaining in v0.

## Reward

Per-agent reward is:

```text
survival_reward
+ gather_reward * gathered_units
+ trade_reward * successful_trades
- action_energy_cost
- death_penalty_if_agent_dies_this_step
+ institution reward modifications
```

Food converted to energy is not counted as a new gather reward.

V1 specialization pressure can set per-agent `resource_preferences`, scaling gather and trade-acquisition rewards by resource type. This creates agents that value food and wood differently without changing the shared `Agent` interface.

## Institutions

Resource Island v0 institutions are:

- `property_rights`: the first agent to gather from a cell claims that cell; unauthorized later gathering from claimed cells is blocked and penalized.
- `redistribution`: taxes positive step rewards and redistributes the pool equally among alive agents.
- `trade_price_controls`: blocks trades whose proposed exchange ratio exceeds a configured maximum. The default 1-for-1 trade is allowed.
- `reputation_system`: increments reputation on successful trades and adds a small reward bonus proportional to reputation.

These are intentionally simple institution hooks. They are validated for mechanics, not yet claimed as full research-grade mechanisms.

Current v0 limitations:

- `property_rights` can only bind when a non-owner tries to gather from a claimed cell. The corrected full run created claims but almost never put non-owners near resource-bearing claimed cells, so zero violations should be read as weak activation, not institutional success.
- `trade_price_controls` cannot bind under the current one-for-one-only trade action space because every trade has exchange ratio 1. Unequal exchange offers are required before this mechanism has a behavioral test.

## Benchmark

There is no clean one-shot Nash analogue for this spatial gather/trade game. The v0 benchmark is therefore an oracle-style efficient allocation upper bound:

- `efficient_gather_upper_bound`: the maximum number of resource units that could be gathered over a finite horizon, ignoring travel and strategic conflict.
- `greedy_full_information_gather_plan`: a deterministic full-information greedy planner that assigns agents to nearest visible resources and estimates reachable gathers under movement costs.

These are benchmarks for scale and sanity, not equilibrium predictions.

## Metrics

Core metrics added for this world:

- `survival_rate`: fraction of agent-time alive.
- `specialization_index`: normalized concentration of each agent's gathered resources across resource types.
- `inequality_over_time`: stepwise Gini coefficient over each agent's current holdings, averaged over the final summary window. Holdings are `food_inventory + wood_inventory + max(energy, 0) / energy_per_food`.
- `resource_sustainability`: current resource stock on the map divided by total resource units introduced so far, where introduced units are initial stock plus cumulative regeneration events. A value near 1 means most introduced resources remain stocked; a value near 0 means the map has been depleted.
- `mean_pairwise_distance`: mean Manhattan distance over alive agent pairs.
- `contact_rate`: final-window fraction of steps in which alive agents were within the configured trade radius.
- `trade_attempt_count`: cumulative within-radius posted trade offers, whether or not inventory/institution checks allowed them.
- `trade_blocked_count`: cumulative attempted trades blocked by inventory or institution checks.
- `trade_inventory_blocked_count`: cumulative attempted trades blocked because no posted offer had sufficient food/wood inventory behind it.
- `trade_institution_blocked_count`: cumulative attempted trades blocked by an institution.
- `property_claims`: cumulative property-right claims created after successful gathering under the property-rights institution.
- `property_violations`: cumulative denied gathers from another agent's claimed cell.
- `property_opportunities`: cumulative non-owner opportunities to observe another agent's claimed cell.
- `property_resource_opportunities`: cumulative non-owner opportunities to observe another agent's claimed cell while resources remain on that cell.
- `property_gather_opportunities`: cumulative non-owner gather attempts on claimed resource cells before the institution blocks or allows the action.
- `stability`: inverse volatility score for finite metric series.
- `robustness_under_shock`: shocked-performance divided by baseline-performance.

## Validation Bar

An item is checked in `ROADMAP_STATUS.md` only after code exists and tests or smoke outputs validate it. For Resource Island v0, validation requires:

- deterministic reset/step mechanics tests,
- movement and collision tests,
- gather/starvation/trade tests,
- institution hook tests,
- v1 activation-pressure tests for contested layouts, unequal trades, price-control blocks, property-right opportunity counters, and specialization rewards,
- static benchmark tests,
- proof that the existing `QLearningMind` class and update rule run through an explicit Resource Island tabular state shape,
- an explicit note that v0 tabular compatibility is achieved by giving `QLearningMind` an inventory-aware discrete observation, so it validates interface compatibility by explicit state-shape configuration rather than unconstrained interface generality,
- a short smoke run that writes CSV and manifest outputs.
