# Completed Roadmap

This file summarizes what has already been built and validated. The detailed source of truth remains `ROADMAP_STATUS.md`, where each checked item is tied to tests, smoke outputs, or full-run validation.

## 1. Core Platform

- Built the shared `World`, `Agent`, and `Institution` abstractions in `core/`.
- Added a registry for config-driven world, mind, and institution composition.
- Added standardized logging helpers for manifests, step CSVs, and summaries.
- Extracted reusable metrics including welfare, gini, exploitability, collusion indexes, stability, robustness, survival rate, specialization, and resource sustainability.
- Preserved the original `arena_v0.py` path as a compatibility module.
- Confirmed old-path versus new-architecture parity for key Pricing Arena workflows.

Validation completed:

- Full unittest discovery passes.
- Phase 0 architecture tests pass.
- Hot-path parity checks pass for multiseed and exploitability CSVs.

## 2. Pricing Arena

- Refactored the original duopoly pricing game into `worlds/pricing_arena/`.
- Implemented six Pricing Arena institutions:
  - `none`
  - `price_cap`
  - `tax_high_price`
  - `random_audit`
  - `anti_collusion`
  - `demand_shock`
- Added static one-shot benchmark computation for Nash and joint-profit reference points.
- Added replicated multiseed and exploitability runners.
- Built combined institution-level comparison tables.
- Added profit-normalized collusion index alongside the older price-normalized proxy.

Validation completed:

- Full Q-learning multiseed run exists for all six mechanisms.
- Full exploitability run exists for all six mechanisms.
- Random-mind baseline runs exist for multiseed and exploitability.
- Combined Phase 1 institution table exists.

Current research status:

- Price caps reduce exploitability across tested mind classes.
- Stronger learners can sit near the cap and still earn high profit, so price-based and profit-based collusion metrics must be interpreted separately.

## 3. Mind Implementations

- Added `QLearningMind` as a shared tabular learner.
- Added random and heuristic baselines.
- Added PyTorch DQN with replay, target network, Huber TD loss, and DeepMind-style centered RMSProp.
- Added PyTorch PPO with GAE, clipped policy objective, value loss, and entropy bonus.
- Retained lightweight NumPy DQN/PPO baselines as explicit `simple_dqn` and `simple_ppo`.
- Added independent-DQN coordinator with decorrelated per-agent RNG streams.
- Added centralized-critic MARL scaffold.

Validation completed:

- Deep-RL/MARL tests pass.
- Pricing Arena best-response validation exists.
- Pricing Arena smoke and full Phase 3 outputs exist.
- Independent-DQN decorrelation fix is covered by tests and smoke gates.

Important note:

- Some older Phase 3 tables contain stale `independent_dqn` rows from before the decorrelation fix. Those rows should not be treated as distinct evidence until fixed reruns replace them.

## 4. Resource Island

- Created `worlds/resource_island/DESIGN.md` as the design lock.
- Implemented bounded-grid movement, collisions, inventory, energy, starvation, gathering, regeneration, posted-offer trading, and render state.
- Added inventory-aware tabular observations for shared `QLearningMind` compatibility.
- Added Resource Island-specific metrics and diagnostics.
- Added resource helpers, trading helpers, benchmarks, and training loop.
- Added Resource Island institutions:
  - property rights
  - redistribution
  - trade price controls
  - reputation system
- Added fixed-width structured observations for PyTorch DQN/PPO/MARL minds.
- Added Resource Island Phase 3 validation and mind comparison builders.

Validation completed:

- Resource Island unit tests pass.
- Q-learning smoke and corrected full runs exist.
- Trade-fix smoke and corrected full outputs exist.
- Resource Island Phase 3 validation outputs exist.
- Resource Island Phase 3 smoke and full outputs exist for DQN, PPO, independent-DQN, and centralized-critic.

Current research status:

- Corrected Resource Island produces nonzero trade attempts and sparse successful trades under tabular Q-learning.
- Property rights and trade price controls are mechanically under-exercised in v0.
- Neural/MARL minds find higher survival/welfare policies but do not learn successful trade under the current reward, observation, and training setup.
- Resource Island v1 needs stronger institution activation: contested resources, unequal exchange offers, specialization pressure, and stricter activation diagnostics.

## 5. Auction House

- Added an Auction House mechanics skeleton in `worlds/auction_house/env.py`.
- Implemented single-item sealed-bid allocation.
- Implemented first-price and second-price payment rules.
- Added deterministic tie-breaking.
- Added revenue, welfare, efficiency, and registry construction coverage.
- Added benchmark placeholders that explicitly raise `NotImplementedError` until the theory-lock pass is done.

Validation completed:

- Auction House mechanics tests pass.

Remaining before research use:

- Write `worlds/auction_house/DESIGN.md`.
- Lock valuation model, information structure, bid grid, episode protocol, and benchmarks.
- Add benchmark tests and smoke/full runs.

## 6. External Atari DQN Audit

The Atari Breakout DQN replication audit is a separate project from the Artificial Economies platform. Its completed work, current findings, missing stages, and acceptance gates are tracked in `ATARI_DQN_AUDIT_ROADMAP.md`.

Current cross-reference only:

- The audit package lives under `audit/`.
- The current Atari audit boundary is preprocessing: raw frames and BT.601 luminance match, but Torch7 resize is not yet byte-replicated in Python.
- Do not treat the Atari audit as validation evidence for the Artificial Economies roadmap.

## 7. Experiment Outputs And Tables

Completed output families include:

- `outputs/full_v0_multiseed/`
- `outputs/v1_exploitability/`
- `outputs/random_v0_multiseed/`
- `outputs/random_v1_exploitability/`
- `outputs/combined_phase1/`
- `outputs/phase3_validation_torch/`
- `outputs/phase3_smoke_torch/`
- `outputs/phase3_full/`
- `outputs/resource_island_tradefix_full/`
- `outputs/resource_island_phase3_validation/`
- `outputs/resource_island_phase3_smoke/`
- `outputs/resource_island_phase3_full/`

Thesis-facing distinct-mind tables exist for:

- Pricing Arena Phase 3.
- Resource Island Phase 3.

## 8. Main Open Items

- Replace stale independent-DQN full outputs with fixed decorrelated reruns where needed.
- Harden Resource Island v1 so property rights and price controls actually bind.
- Finish Auction House design lock and benchmark pass.
- Add Public Goods / Commons world.
- Add Labor Market world.
- Decide whether Central Planner / Tax Schedule is a standalone world or cross-world institution sweep.
- Write literature section and thesis/proposal draft.
- Continue Atari audit only after resolving or formally documenting the preprocessing mismatch.
