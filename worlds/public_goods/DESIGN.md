# Public Goods / Commons Design Lock

The Public Goods / Commons world is a compact common-pool resource economy. It
is meant to test free-riding, over-extraction, contribution incentives, and
commons sustainability under the same `World` / `Agent` / `Institution`
interface used by Pricing Arena, Resource Island, and Auction House.

## Agents

- Agents repeatedly interact with one shared resource pool.
- Agents do not have spatial positions in v0; this world isolates public-pool
  incentives from movement.
- Each agent receives a private reward from extraction and pays a private cost
  for contribution.

## Observation

Tabular Q-learning receives:

```text
(pool_bin, agent_bin)
```

- `pool_bin` discretizes the current pool stock into `[0, n_actions - 1]`.
- `agent_bin` is the agent id modulo `n_actions`, allowing heterogeneous
  learned behavior without changing the shared Q-learning class.
- Information-restriction institutions may replace `pool_bin` with a neutral
  midpoint.

## Actions

The v0 action set has five discrete actions:

```text
0 noop
1 contribute_low
2 contribute_high
3 extract_low
4 extract_high
```

Contributions add to the public pool after being multiplied by a public-return
factor. Extractions remove stock from the pool and pay private reward.

## Pool Dynamics

- The pool is bounded by `pool_capacity`.
- Extractions are rationed proportionally if total requested extraction exceeds
  available stock.
- Contributions enter before stochastic-free deterministic regeneration.
- Regeneration follows a simple linear gap rule:

```text
regen = regeneration_rate * (pool_capacity - pool_after_actions)
```

This avoids hidden randomness and makes benchmark tests reproducible.

## Institutions

Public Goods v0 institutions:

- `public_goods_penalty`: penalizes agents whose extraction exceeds a configured
  sustainable per-agent allowance.
- `contribution_matching`: adds matching public funds for contributions.
- `public_goods_reputation`: gives a small reward bonus to repeated
  contributors.
- `information_restriction`: hides the pool stock in tabular observations.

These institutions are intentionally simple, but each has activation counters
or visible metric effects so inactive mechanisms are not mistaken for null
economic results.

## Benchmarks

There is no single closed-form equilibrium for the repeated learner world. The
v0 benchmark file therefore provides bracketing policies:

- `free_rider_benchmark`: all agents extract high and never contribute.
- `social_optimum_benchmark`: all agents contribute high and extract only a
  conservative sustainable amount.
- `simulate_fixed_policy`: deterministic finite-horizon rollout for simple
  fixed action profiles.

These are sanity bounds, not claims about learned equilibrium.

## Metrics

Core metrics:

- `pool_stock`
- `sustainability`
- `contribution_total`
- `extraction_total`
- `contribution_rate`
- `extraction_rate`
- `collapse`
- `collapse_rate`
- `welfare`
- `reward_total`
- `inequality`
- institution diagnostics such as `penalty_total`, `matched_contribution`, and
  `reputation_bonus_total`.

## Validation Bar

Before this world can produce research evidence:

- mechanics tests must cover contribution, extraction, rationing, regeneration,
  collapse, and observations;
- benchmark tests must cover hand-computed fixed policies;
- each institution must be shown to bind or have a measured reason why it did
  not;
- `QLearningMind` must run without changing its shared update rule;
- a smoke run must write seed summaries, aggregate summaries, and a manifest.
