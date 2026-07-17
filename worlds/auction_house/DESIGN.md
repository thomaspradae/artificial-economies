# Auction House Design Lock

Auction House is the third validated economic world target after Pricing Arena
and Resource Island. Its purpose is to test whether learning agents recover or
violate standard auction-design predictions in repeated sealed-bid auctions.

## Economic Scope

The v1 world is a single-item independent-private-values auction. Each bidder
privately observes its own value for the item, submits one sealed bid, and then
receives payoff equal to value minus payment if it wins, zero otherwise.

The first research-grade target is not a complicated auction catalog. It is a
clean test of whether learning agents recover:

- truthful bidding in second-price/Vickrey auctions,
- bid shading in first-price auctions,
- revenue/efficiency tradeoffs under reserve prices,
- overbidding and regret when learning dynamics fail the textbook prediction.

## Agents

- There are `n_bidders >= 2`.
- Each bidder has a private value independently drawn from the valuation grid.
- Bidders do not observe other bidders' values or bids before acting.
- A bidder's reward is its realized auction surplus: `value - payment` if it
  wins, otherwise `0`.

## Observation

The tabular observation for `QLearningMind` is:

```text
(valuation_bin,)
```

where `valuation_bin` is the nearest index on the configured valuation grid.
This keeps the existing `QLearningMind` class unchanged while giving it the
economically relevant private information.

The richer state in `info` and `render_state()` includes valuations, bids,
winner, payment, revenue, welfare, regret, and bidding diagnostics.

## Actions

Each action is a bid-grid index. The bid grid is an ordered finite set of
non-negative bids. Continuous bid values may still be passed directly in tests,
but learning agents use discrete action indices.

## Mechanisms

The supported mechanisms are:

- `second_price`: highest eligible bidder wins and pays
  `max(reserve_price, second_highest_bid)`.
- `first_price`: highest eligible bidder wins and pays its own bid.
- `clock`: a simple discrete ascending-clock abstraction where each bidder's
  action is its dropout/maximum-stay price; the highest dropout price wins and
  pays the second-highest dropout price, subject to the reserve.

Ties are deterministic: the lowest bidder id among tied highest eligible bids
wins. This avoids hidden randomness in tests and run-to-run comparisons.

If no bid meets the reserve price, the item is not sold, revenue is zero, and
all bidder rewards are zero.

Information variants are modeled as institutions over bidder observations:

- `auction_information_policy` can mix a bidder's private value signal with a
  public rival-value signal while preserving the one-bin tabular observation.
- The same institution can add bounded value-bin noise for noisy-information
  ablations.

These variants change information, not allocation or payment rules.

## Benchmarks

The benchmark layer is part of the economics, not optional decoration:

- In a second-price private-values auction, truthful bidding is weakly dominant.
- The simple clock abstraction shares the second-price clearing benchmark:
  truthful dropout at value is the reference behavior.
- In a symmetric first-price auction with risk-neutral bidders and values
  uniformly distributed on `[low, high]`, the continuous equilibrium bid is
  `b(v) = v - (v - low) / n_bidders`.
- On a discrete bid grid, the first-price reference is the closest feasible
  shaded bid at or below the continuous expression, plus ex-post regret checks
  on the actual grid.

Benchmarks are tested on hand-computed cases before they can be used as
reference lines in experiments.

## Metrics

Core metrics are reported per auction step and averaged over final windows:

- `revenue`: seller payment.
- `bidder_surplus`: sum of bidder rewards.
- `welfare`: realized value of the allocated item, or zero if unsold.
- `max_possible_welfare`: highest value that could be allocated above reserve.
- `allocative_efficiency`: indicator that the item went to the efficient bidder
  or was efficiently unsold.
- `welfare_efficiency`: realized welfare divided by feasible maximum welfare.
- `ex_post_regret_mean`: mean bidder gain from the best unilateral bid on the
  grid holding other bids fixed.
- `truthful_bid_distance_mean`: mean absolute distance from value.
- `first_price_shading_distance_mean`: mean distance from the first-price
  bid-shading reference.
- `overbid_rate`: fraction of bidders bidding above value.
- `underbid_rate`: fraction of bidders bidding below value.
- `allocation_error`: `1 - allocative_efficiency`.

## Validation Bar

Auction House is not research-ready until:

- mechanics tests cover allocation, payments, reserve/no-sale, and tie-breaking,
- benchmark tests cover truthful second-price bidding and first-price shading,
- tabular Q-learning runs without modifying the shared `QLearningMind` class,
- smoke outputs show finite metrics for first-price and second-price auctions,
- variant smoke outputs show finite metrics for clock and information/noise
  scenarios before those variants are included in full comparisons,
- learned bidding is compared against benchmark behavior before any full-run
  institution claim is made.
