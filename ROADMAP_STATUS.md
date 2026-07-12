# Artificial Economies Roadmap Status

Status rule: an item is checked only if it is implemented in this repo, covered by tests or full-run output validation, and the validation has been observed. Existing behavior in `arena_v0.py` does not count as completion of a requested refactor item unless the requested target file/interface exists.

## Validation Evidence

- [x] Core v0 tests pass: `test_arena_v0.py` ran 7 tests, all passing.
- [x] Multiseed tests pass: `test_multiseed.py` ran 6 tests, all passing.
- [x] Exploitability tests pass: `test_exploitability.py` ran 6 tests, all passing.
- [x] Phase 0 architecture tests pass: `test_phase0_core.py` ran 18 focused tests and is included in full discovery, covering core interfaces, registry composition, pricing-world parity, mind wrappers, institution transforms, logging, and combined-table construction.
- [x] Phase 3 deep-RL/MARL tests pass: `test_phase3_deep_rl.py` ran 10 tests and is included in full discovery, covering feature encoding, PyTorch DQN, PyTorch PPO, explicit NumPy fallback baselines, decorrelated independent learners, centralized critic, Pricing Arena integration, and validation-run output.
- [x] Resource Island tests pass: `test_resource_island.py` ran 20 tests, all passing, covering metrics, benchmarks, movement/collision, gathering/starvation, posted-offer trading, trade-radius behavior, institutions, Q-learning integration, corrected inventory-aware observations, and smoke output.
- [x] Resource Island Phase 3 cross-mind tests pass: `test_resource_island_phase3.py` ran 7 tests, all passing, covering fixed-width structured observations, DQN, PPO, independent-DQN, centralized-critic, validation-run output, smoke-run mind selection, DQN-vs-independent-DQN divergence, and mind-comparison table construction.
- [x] Auction House mechanics tests pass: `test_auction_house.py` ran 5 tests, all passing, covering allocation, first-price and second-price payment resolution, deterministic tie-breaking, registry construction, and explicitly deferred benchmarks.
- [x] Full unittest discovery passes: 79 tests, all passing.
- [x] Python compile check passes for the legacy v0 module, experiment scripts, new `core/`, `worlds/`, `minds/`, `institutions/`, Resource Island modules, Auction House modules, deep-RL/MARL modules, validation runners, combined-table builder, and all test files.
- [x] Phase 0 hot-path parity smoke check passes: old-path vs new-architecture multiseed CSVs are byte-identical for fixed seeds.
- [x] Phase 0 hot-path parity smoke check passes: old-path vs new-architecture exploitability CSVs are byte-identical for fixed seeds.
- [x] Phase 3 guard parity passes: Q-learning multiseed/exploitability CSVs remain byte-identical to the saved Phase 0 new-path parity outputs after adding PyTorch deep-RL/MARL support.
- [x] PyTorch installed and pinned: `.venv` imports `torch==2.12.1+cpu`, CUDA unavailable/unused, and a trivial forward/backward smoke test is covered in `test_phase3_deep_rl.py`.
- [x] Full multiseed output exists at `outputs/full_v0_multiseed/` with `steps=40000`, `n_seeds=20`, all 6 mechanisms, Student-t 95% CI, CSV summaries, manifest, and plots.
- [x] Full exploitability output exists at `outputs/v1_exploitability/` with `incumbent_steps=40000`, `adversary_steps=20000`, `evaluation_steps=5000`, `n_seeds=20`, `adversary_restarts=3`, all 6 mechanisms, CSV summaries, manifest, and plots.
- [x] Full random-mind multiseed baseline exists at `outputs/random_v0_multiseed/` with `steps=40000`, `n_seeds=20`, all 6 mechanisms, CSV summaries, manifest, and plots.
- [x] Full random-mind exploitability baseline exists at `outputs/random_v1_exploitability/` with `incumbent_steps=40000`, `adversary_steps=20000`, `evaluation_steps=5000`, `n_seeds=20`, `adversary_restarts=3`, all 6 mechanisms, CSV summaries, manifest, and plots.
- [x] Combined Phase 1 table exists at `outputs/combined_phase1/institution_summary.csv` with Q-learning and random-mind rows for all 6 mechanisms.
- [x] Phase 3 PyTorch best-response validation exists at `outputs/phase3_validation_torch/best_response_validation.csv`; DQN and PPO learn the computed one-step best response at the validation seed. The `independent_dqn` row is execution evidence for the registered alias, not a distinct algorithmic condition.
- [x] Phase 3 PyTorch Pricing Arena smoke validation exists at `outputs/phase3_validation_torch/pricing_smoke_validation.csv`; DQN, PPO, the `independent_dqn` alias, and centralized-critic all emit finite metrics.
- [x] Phase 3 torch-vs-NumPy qualitative comparison exists at `outputs/phase3_validation_torch/torch_vs_numpy_qualitative.csv`; DQN and PPO families both show price-cap collusion suppression directionally on matched smoke runs.
- [x] Phase 3 PyTorch all-institution smoke outputs exist at `outputs/phase3_smoke_torch/` for DQN, PPO, independent-DQN, and centralized-critic multiseed/exploitability runs, plus `mind_comparison.csv`.
- [x] Phase 3 full-run outputs exist for DQN, PPO, independent-DQN, and centralized-critic: all four have n=20 multiseed outputs and n=20 exploitability outputs with 3 adversary restarts, but `independent_dqn` is code-identical to the per-agent DQN condition and should not be counted as an independent corroborating mind.
- [x] Phase 3 full validation passes: `outputs/phase3_full/validation_report.json` reports `status=pass`, 120 multiseed rows per Phase 3 mind, 120 selected exploitability rows per Phase 3 mind, 360 restart rows per Phase 3 mind, and 36 combined comparison rows across Q-learning, random, DQN, PPO, independent-DQN, and centralized-critic. Interpret the independent-DQN rows as DQN-alias reproducibility rows until a distinct MARL baseline is implemented.
- [x] Phase 3 comparison table includes price, price-dispersion, total-profit, quantity, historical price-normalized `collusion_index_mean`, and literature-comparable `profit_collusion_index_mean` columns in `outputs/phase3_full/mind_comparison.csv`.
- [x] Thesis-facing Phase 3 distinct-mind comparison table exists at `outputs/phase3_full/distinct_mind_comparison.csv`; it excludes the `independent_dqn` alias row and contains Q-learning, random, DQN, PPO, and centralized-critic.
- [x] Resource Island smoke output exists at `outputs/resource_island_smoke/` with `steps=200`, `n_seeds=3`, five institutions, Q-learning agents, CSV summaries, and manifest.
- [x] Resource Island pre-observation-fix full output exists at `outputs/resource_island_full/` with `steps=40000`, `n_seeds=20`, five institutions, Q-learning agents, CSV summaries, and manifest; this is now treated as a diagnostic artifact, not final evidence for the corrected Resource Island implementation.
- [x] Resource Island corrected-observation smoke output exists at `outputs/resource_island_obsfix_smoke/` with `steps=500`, `n_seeds=3`, five institutions, inventory-aware tabular observations, trade/property diagnostic counters, CSV summaries, and manifest.
- [x] Resource Island full-run trigger diagnosis observed: `trade_count` is 0.0 for every institution and every seed in `outputs/resource_island_full/summary_by_seed.csv`; source audit found inventory-imbalance observations and 3D Q-table wiring are now present, but the saved full output predates the current `property_claims`/trade-diagnostic schema and should be rerun before drawing Resource Island institution conclusions.
- [x] Resource Island contact diagnosis observed: corrected-observation smoke had nonzero trade actions but zero successful trades because strict adjacency made contact rare and the complementary-offer protocol blocked on missing inventory.
- [x] Resource Island trade-fix smoke output exists at `outputs/resource_island_tradefix_smoke/` with nonzero successful trades, posted-offer trading, all-island default trade radius, and trade block diagnostics.
- [x] Resource Island corrected full output exists at `outputs/resource_island_tradefix_full/` with `steps=40000`, `n_seeds=20`, five institutions, inventory-aware observations, posted-offer trading, CSV summaries, and manifest.
- [x] Resource Island corrected full validation observed: `summary_by_seed.csv` has 100 rows, `summary_aggregate.csv` has 5 rows, every institution has 20 seeds, and no blank/NaN/inf values were found.
- [x] Resource Island Phase 3 validation output exists at `outputs/resource_island_phase3_validation/` with DQN/PPO single-agent sanity rows, Q-learning-vs-DQN qualitative comparison rows, and manifest.
- [x] Resource Island Phase 3 all-institution smoke outputs exist for DQN, PPO, independent-DQN, and centralized-critic at `outputs/resource_island_*_smoke/`; each has 15 by-seed rows, 5 aggregate rows, and no blank/NaN/inf values. The independent-DQN rows are currently expected to match DQN because the registered class is an alias of the same per-agent learner.
- [x] Resource Island Phase 3 smoke comparison table exists at `outputs/resource_island_phase3_smoke/mind_comparison.csv` with 25 rows across Q-learning, DQN, PPO, independent-DQN, and centralized-critic for all five Resource Island institutions; it should be read as four distinct conditions plus one DQN alias row.
- [x] Resource Island Phase 3 full-run outputs exist for DQN, PPO, independent-DQN, and centralized-critic: all four have `steps=40000`, `n_seeds=20`, five institutions, CSV summaries, and manifests.
- [x] Resource Island Phase 3 full validation observed: each neural/MARL full run has 100 by-seed rows, 5 aggregate rows, exactly 20 seeds per institution, and no blank/NaN/inf values.
- [x] Resource Island Phase 3 full comparison table exists at `outputs/resource_island_phase3_full/mind_comparison.csv` with 25 rows across Q-learning, DQN, PPO, independent-DQN, and centralized-critic for all five Resource Island institutions; it should be read as four distinct conditions plus one DQN alias row.
- [x] Thesis-facing Resource Island distinct-mind comparison table exists at `outputs/resource_island_phase3_full/distinct_mind_comparison.csv`; it excludes the `independent_dqn` alias row and contains Q-learning, DQN, PPO, and centralized-critic.
- [x] Independent-DQN decorrelation fix validated: `test_phase3_deep_rl.py` covers decorrelated independent-DQN Q-values from one base seed, `test_resource_island_phase3.py` confirms Resource Island DQN and independent-DQN rollouts diverge, and `outputs/resource_island_independent_dqn_smoke_fixed_gate/summary_aggregate.csv` differs from the matching DQN smoke gate.

## Running Work

- [ ] Fixed independent-DQN replacement reruns are active remotely: old1 is running Pricing Arena `independent_dqn` multiseed fixed output, queued to run exploitability fixed output afterward; old2 is running Resource Island `independent_dqn` full fixed output.

## Validated Implemented Artifacts

- [x] `arena_v0.py`: repeated duopoly pricing environment with tabular Q-learning firms.
- [x] Six current mechanisms exist in the pricing arena: `none`, `price_cap`, `tax_high_price`, `random_audit`, `anti_collusion`, `demand_shock`.
- [x] Static one-shot benchmark computation exists: grid Nash price pairs and symmetric joint-profit price.
- [x] Single-run CLI exists and writes trace CSVs, summary CSV, and plots.
- [x] `run_multiseed.py`: paired-seed replicated experiment runner using the new pricing-arena training path, with per-seed summaries, aggregate summaries, rankings, manifest, and CI plots.
- [x] `run_exploitability.py`: frozen-policy exploitability runner using the new pricing-arena training path, with adversary restarts, selected restart summaries, aggregate summaries, manifest, and CI plots.
- [x] Experiment manifests record configuration, seed design, metric definitions/method notes, platform, Python version, and output paths.
- [x] `core/agent.py`, `core/world.py`, `core/institution.py`: abstract base interfaces exist and are import-tested.
- [x] `core/metrics.py`: extracted current exploitability, victim-loss, welfare-damage, price-normalized collusion-index, profit-normalized collusion-index, gini, average-price, survival-rate, specialization-index, stability, robustness-under-shock, and resource-sustainability helpers with unit tests.
- [x] `core/registry.py`: minimal world/mind/institution registry exists and can instantiate a pricing-arena experiment in tests.
- [x] `core/logger.py`: minimal manifest, step CSV, and summary JSON helpers exist and are unit-tested.
- [x] `worlds/pricing_arena/env.py`: `PricingArenaWorld` subclasses `World`, applies institutions through `Institution.apply()`, and step output is tested against `DuopolyMarket`.
- [x] `worlds/pricing_arena/benchmarks.py`: static benchmark wrapper exists.
- [x] `worlds/pricing_arena/training.py`: new architecture training/evaluation/exploitability hot path exists and is parity-tested against the legacy path.
- [x] `minds/q_learning.py`: `QLearningMind` wrapper exists, subclasses `Agent`, preserves existing two-index `QAgent` behavior by default, and supports explicit higher-dimensional discrete state shapes for non-pricing tabular worlds.
- [x] `minds/random_mind.py`: uniform random baseline exists and has unit tests.
- [x] `minds/heuristic_mind.py`: simple pricing heuristic baseline exists and has unit tests.
- [x] `minds/deep_rl/torch_dqn_mind.py`: PyTorch structured-observation DQN mind exists, subclasses `Agent`, uses a target network, replay, Huber TD loss, and DeepMind-style centered RMSProp, and is unit/integration-tested.
- [x] `minds/deep_rl/torch_ppo_mind.py`: PyTorch structured-observation discrete PPO mind exists, subclasses `Agent`, uses Categorical actions, GAE, clipped surrogate objective, value loss, entropy bonus, and minibatch epochs, and is unit/integration-tested.
- [x] `minds/deep_rl/dqn_mind.py` and `minds/deep_rl/ppo_mind.py`: public compatibility modules now alias the PyTorch implementations.
- [x] `minds/deep_rl/simple_dqn_mind.py` and `minds/deep_rl/simple_ppo_mind.py`: former NumPy implementations are retained as explicit lightweight baselines registered as `simple_dqn` and `simple_ppo`.
- [x] `minds/deep_rl/features.py`, `minds/deep_rl/numpy_nn.py`, and `minds/deep_rl/torch_optim.py`: shared feature encoding, legacy NumPy primitives, and DeepMind-style RMSProp support exist and are tested through Phase 3.
- [x] `minds/marl/independent_learners.py`: independent learner coordinator exists, and `independent_dqn` is registered. Current implementation uses `IndependentDQNLearners` with NumPy `SeedSequence` child streams, per-agent replay/exploration RNGs, and scoped PyTorch network initialization so the condition is no longer a DQN alias.
- [x] `minds/marl/centralized_critic.py`: decentralized-actor/centralized-critic scaffold exists and is integration-tested on Pricing Arena.
- [x] `institutions/*.py`: six institution classes exist, subclass `Institution`, have focused transform tests, and are used by `PricingArenaWorld.step()`.
- [x] `build_combined_table.py`: merges multiseed and exploitability aggregate outputs into one institution-level comparison table, including price-normalized and profit-normalized collusion columns.
- [x] `run_phase3_validation.py`: best-response and pricing-smoke validation runner exists and writes CSV/manifest outputs.
- [x] `run_phase3_full.py`: managed Phase 3 full-run orchestrator exists, dry-run output matched the requested 8 commands, and the live batch was launched through it.
- [x] `validate_phase3_full.py`: Phase 3 full-output validator exists and passes on the completed full batch.
- [x] `worlds/resource_island/DESIGN.md`: Resource Island design lock exists with inventory-aware tabular observations, actions, rewards, boundaries, collision resolution, starvation, posted-offer trading, trade-radius semantics, institutions, benchmark, metrics, and validation rules.
- [x] `worlds/resource_island/env.py`: `ResourceIslandWorld` subclasses `World` and implements bounded-grid movement, simultaneous collision resolution, inventory-aware local tabular observations, inventory, energy, gathering, starvation/death, resource regeneration, posted-offer trading, configurable trade radius, property/trade diagnostic counters, metrics, and render state.
- [x] `worlds/resource_island/resources.py`: Resource Island resource initialization, local visibility counting, and stochastic regeneration helpers exist and are tested through the world.
- [x] `worlds/resource_island/trading.py`: Resource Island adjacency and one-for-one complementary trade matching helpers exist and are tested through the world.
- [x] `worlds/resource_island/features.py`: fixed-width structured Resource Island observation encoder exists for PyTorch DQN/PPO/MARL minds and is unit-tested for grid-size-independent shape and finite values.
- [x] `worlds/resource_island/benchmarks.py`: efficient-gather upper bound and greedy full-information gather planner exist and have unit tests.
- [x] `worlds/resource_island/training.py`: Resource Island training loop supports Q-learning, DQN, PPO, independent-DQN, and centralized-critic with tabular or fixed structured observations as appropriate.
- [x] `run_resource_island_smoke.py`: Resource Island smoke runner writes seed summaries, aggregate summaries, diagnostic counters, and manifest for `--mind q_learning`, `dqn`, `ppo`, `independent_dqn`, and `centralized_critic`.
- [x] `run_resource_island_phase3_validation.py`: Resource Island neural/MARL validation runner exists and writes single-agent sanity, qualitative comparison, and manifest outputs.
- [x] `build_resource_island_mind_comparison.py`: Resource Island mind x institution comparison builder exists and merges aggregate outputs without requiring pandas.
- [x] `institutions/resource_island.py`: property rights, redistribution, trade price controls, and reputation system institutions exist, subclass `Institution`, and have focused tests.
- [x] `worlds/auction_house/env.py`: Auction House mechanics skeleton exists, subclasses `World`, and implements single-item sealed-bid allocation, deterministic tie-breaking, first-price payments, second-price payments, revenue/welfare/efficiency metrics, and registry construction tests.
- [x] `worlds/auction_house/benchmarks.py`: Auction House benchmark placeholders exist and explicitly raise `NotImplementedError` until the theory-lock pass.
- [x] `scripts/pull_results.sh`: remote output puller exists and has been run successfully for old1 plus old2 through ofi1.
- [x] `outputs/phase3_full/mind_comparison.csv`: full Phase 3 comparison table exists across Q-learning, random, DQN, PPO, independent-DQN, and centralized-critic for all 6 Pricing Arena mechanisms. Existing independent-DQN rows were generated before the decorrelation fix and should be treated as stale alias rows until fixed reruns land.
- [x] Phase 3 price-cap divergence diagnosis: exploitability falls under price cap across distinct tested mind families, while collusion effects are metric-sensitive for higher-capability minds. The DQN condition sits effectively at the cap under price cap (`avg_price_mean=3.999038`, `price_dispersion_mean=0.259025`) and earns materially higher profit than uncapped (`profit_total_mean=291.703303` vs. `236.345482`), supporting a real cap/quantity-profit channel rather than only normalized-index noise. The matching independent-DQN row is not separate corroboration because it currently routes to the same per-agent DQN implementation.

## Phase 0 - Core Abstractions

- [x] Define `World` abstract base: `step()`, `reset()`, `get_metrics()`, `render_state()`
- [x] Define `Agent` abstract base: `act(obs)`, `update(reward, next_obs, done)`
- [x] Define `Institution` abstract base: `apply(state) -> modified state/rewards`
- [x] Build `metrics.py`: welfare, gini/inequality, collusion_index, exploitability, stability, specialization, robustness_under_shock
- [x] Build `registry.py` for config-driven world/mind/institution composition
- [x] Build `logger.py` for standardized run manifests (JSON) + CSV outputs
- [x] Refactor `arena_v0.py` into `worlds/pricing_arena/` conforming to `World`
- [x] Refactor tabular Q-learner into `minds/q_learning.py` conforming to `Agent`
- [x] Refactor existing 6 mechanisms into `institutions/*.py` conforming to `Institution`
- [x] Port existing test suite, confirm parity with pre-refactor behavior
- [x] Add `random_mind.py` and `heuristic_mind.py` as baselines

Phase 0 notes:

- `arena_v0.py` is retained as a compatibility module and single-run CLI. The replicated experiment hot path now runs through `worlds/pricing_arena/training.py`, `PricingArenaWorld`, `QLearningMind`, and `Institution.apply()`.
- `DuopolyMarket.step()` still exists for compatibility and parity tests, but `run_multiseed.py` and `run_exploitability.py` no longer import the legacy training functions directly.
- Existing tests remain top-level files rather than a `tests/` package; that is cosmetic. The suite has been extended with architecture/parity coverage and full discovery passes.
- `core/metrics.py` now contains the current extracted formulas needed by v0/v1 plus Resource Island's generic survival, specialization, stability, robustness-under-shock, and resource-sustainability helpers.

## Phase 1 - Environment 01: Pricing Arena, Full Rigor

- [x] Full multiseed run (n=20+) across all institutions
- [x] Full exploitability protocol (n=20 seeds, 3 adversary restarts)
- [x] Static benchmark comparison (Nash, joint-profit) as reference lines
- [x] Collusion-index methodology audited against Calvano et al. methodology
- [x] Ranking table: institution x (welfare, collusion, exploitability)
- [ ] Literature section: Calvano et al. 2020 AER, Calvano et al. 2021 imperfect monitoring, Asker/Fershtman/Pakes 2022, Banchio/Skrzypacz auction design
- [ ] Write proposal/thesis draft v1 grounded entirely in Environment 01 results

Notes:

- A welfare/collusion ranking table exists at `outputs/full_v0_multiseed/mechanism_rankings.csv`.
- Exploitability aggregate results exist at `outputs/v1_exploitability/summary_aggregate.csv`.
- A single combined institution x welfare/collusion/exploitability table exists at `outputs/combined_phase1/institution_summary.csv`.
- Random-mind baseline outputs exist at `outputs/random_v0_multiseed/` and `outputs/random_v1_exploitability/` and are included in the combined table as scale/reference rows.
- No written thesis/proposal interpretation has been added yet.
- The existing `collusion_index` is documented as a price-normalized proxy, not the Calvano et al. profit-normalized index. A separate `profit_collusion_index` helper implements the profit-normalized formula, and `outputs/phase3_full/mind_comparison.csv` now reports both `collusion_index_mean` and `profit_collusion_index_mean`.

## Environments Track - New World Playbook

This replaces the old separate Phase 2, Phase 4, and Phase 5 world sections. Resource Island, Auction House, Public Goods, and Labor Market are all instances of the same engineering recipe: design lock, `World` subclass, benchmark, existing-mind wiring, institutions, validation. Central Planner is tracked separately as a likely cross-world institution/parameter sweep, not as a default standalone world.

### Shared Playbook

- [ ] P.1 Design lock: write `worlds/<name>/DESIGN.md` before implementation, including observations, actions, rewards, static benchmark, arbitrary rule decisions, and tie-breaking/collision/boundary semantics where relevant.
- [ ] P.2 Core implementation: create `worlds/<name>/env.py` subclassing `core.world.World` with `reset()`, `step()`, and `get_metrics()`.
- [ ] P.2 Static benchmark: create `worlds/<name>/benchmarks.py` with the analytical, oracle, or bracketing benchmark used to interpret learned behavior.
- [ ] P.2 Metrics: reuse `core/metrics.py` where concepts transfer; add new metrics only when the new world defines them operationally.
- [ ] P.3 Mind wiring: run `minds/q_learning.py` unchanged in the new world; if the mind must change, treat it as an interface bug.
- [ ] P.3 Sanity case: verify the simplest single-agent or two-agent case behaves sensibly before adding institutions.
- [ ] P.4 Institutions: add environment-specific institutions as small `Institution` subclasses and reuse existing institution logic where the concept transfers.
- [ ] P.5 Mechanics tests: unit-test world-specific transition logic, allocation logic, matching logic, or movement logic.
- [ ] P.5 Smoke run: run a short n=2-3 seed smoke experiment before any full run.
- [ ] P.5 Full run: run n=20 multiseed once smoke outputs are clean.
- [ ] P.5 Cross-world abstraction test: the same `QLearningMind` class runs on the new world and at least one prior world without modification.
- [ ] P.6 Cross-mind pass: wire PyTorch DQN/PPO/MARL minds into new worlds after 2-3 worlds exist, rather than repeating deep-RL integration per world prematurely.

### Environment Queue

#### 1. Resource Island

- [x] `worlds/resource_island/DESIGN.md`: lock grid boundaries, collision resolution, inventory-aware local observability, gather caps, inventory/energy dynamics, starvation/inactivity rules, posted-offer trading rules, trade-radius semantics, and oracle/planner benchmark definition.
- [x] `worlds/resource_island/env.py`: 2D grid world with agent positions, movement, partial observability, inventory-aware tabular observations, inventory, energy, resource nodes, posted-offer trading, and diagnostics.
- [x] `worlds/resource_island/resources.py`: resource spawning and regeneration dynamics.
- [x] `worlds/resource_island/trading.py`: distance helpers and simple one-shot trade matching support; no multi-round bargaining in v0.
- [x] `worlds/resource_island/benchmarks.py`: efficient-allocation or planner/oracle baseline for comparison.
- [x] Observation discretization exists so tabular Q-learning is feasible.
- [x] `QLearningMind` runs on Resource Island through the same shared class and update rule, using an explicit 3D discrete state shape.
- [x] Sanity check: lone agent can survive by gathering before multi-agent institutions are added.
- [x] Institutions: property rights, redistribution, price controls on trades, reputation system.
- [x] Metrics: `specialization_index` and `survival_rate`.
- [x] Metrics: inequality over time and resource sustainability with locked operational definitions.
- [x] Cross-world test: same `QLearningMind` class runs on Pricing Arena and Resource Island; Resource Island now passes an explicit state shape instead of forcing the world into Pricing Arena's two-index table.
- [x] Smoke run: `outputs/resource_island_smoke/` exists with `steps=200`, `n_seeds=3`, all five Resource Island institutions, summary CSVs, and manifest.
- [x] Corrected-observation smoke run: `outputs/resource_island_obsfix_smoke/` exists with `steps=500`, `n_seeds=3`, all five Resource Island institutions, inventory-aware observations, trade/property diagnostic counters, summary CSVs, and manifest.
- [x] Trade-fix smoke run: `outputs/resource_island_tradefix_smoke/` exists with `steps=500`, `n_seeds=3`, all five Resource Island institutions, posted-offer trading, nonzero successful trades, trade/property diagnostic counters, summary CSVs, and manifest.
- [x] Full run: `outputs/resource_island_tradefix_full/` exists with `steps=40000`, `n_seeds=20`, all five Resource Island institutions, inventory-aware observations, posted-offer trading, trade/property diagnostic counters, summary CSVs, and manifest.
- [x] P.6 structured-observation audit: DQN, PPO, the `independent_dqn` alias, and centralized-critic are portable to Resource Island through explicit `obs_dim`/`action_dim`; centralized-critic actors now receive the same fixed `obs_dim` as the centralized critic.
- [x] P.6 fixed structured observations: Resource Island exposes a constant-width vector with normalized energy, inventory, position, local resource maps, nearby-agent context, and alive flag.
- [x] P.6 smoke pass: DQN, PPO, the `independent_dqn` alias, and centralized-critic run on all five Resource Island institutions for `steps=500`, `n_seeds=3`, with clean CSV outputs.
- [x] P.6 smoke comparison table: `outputs/resource_island_phase3_smoke/mind_comparison.csv` compares Q-learning, DQN, PPO, the `independent_dqn` alias, and centralized-critic on Resource Island.
- [x] P.6 full run: n=20 Resource Island replicated runs for DQN, PPO, and centralized-critic as distinct neural conditions. The existing `independent_dqn` full output was generated before the decorrelation fix and should be treated as stale until the fixed rerun lands.
- [ ] P.6 fixed independent-DQN full rerun: rerun Resource Island `independent_dqn` after the decorrelation fix and rebuild the Resource Island full comparison table.
- [ ] P.6 full comparison: final Resource Island full cross-mind table after replacing the stale independent-DQN row.

Resource Island full-run note:

- The old `outputs/resource_island_full/` run used the original two-bin observation `(energy_bin, local_resource_bin)`. That observation did not expose inventory imbalance, so Q-learning could not condition trade offers on having food surplus or wood surplus. Treat that full run as a diagnostic artifact, not final Resource Island evidence.
- The corrected-observation smoke exposed inventory imbalance but still produced zero successful trades. Follow-up diagnostics showed the causes were strict-contact rarity and an overly brittle complementary-offer protocol.
- The current corrected full run uses posted one-sided offers and all-island default trade radius. Trade attempts are nonzero for every institution and every seed; successful trades remain sparse (`trade_count_mean` around 0.29 in the final window for most institutions), and almost all blocked attempts are inventory blocks rather than institution blocks.
- `property_rights` creates claims in every seed but has zero observed violations in the corrected full run. `trade_price_controls` has zero institution blocks because v0 only offers 1-for-1 trades, so there is no unequal exchange to block. Treat those as mechanism-inactive diagnostics, not evidence that the concepts are ineffective in richer Resource Island variants.
- Follow-up property-rights opportunity replay over 20 seeds x 40,000 steps found claims in 13/20 seeds but only 21 total non-owner-visible claimed-cell steps, only 9 claimed-resource-visible steps, and zero non-owner gather attempts on claimed cells. Property rights were therefore almost never put under behavioral pressure in v0.
- Source audit confirms `trade_price_controls` is structurally unable to bind in v0 because every constructed trade has `food_units=1` and `wood_units=1`, so the exchange ratio is always exactly 1. Unequal exchange offers are required before this institution can be evaluated.
- P.6 smoke results are execution/integration evidence, not final statistical Resource Island cross-mind evidence. At `n_seeds=3`, Q-learning has higher short-horizon welfare and more successful trades than DQN/PPO/MARL on the `none` institution. The old DQN/independent-DQN smoke match exposed an implementation defect; the fixed independent-DQN implementation now diverges from DQN in unit and smoke gates.
- P.6 full results show a stronger version of the smoke pattern. On `none`, neural/MARL minds have higher survival and welfare than Q-learning (`survival_rate_mean`: Q-learning `0.9686`, DQN `0.9875`, PPO `0.9931`, centralized-critic `0.9955`; `welfare_mean`: Q-learning `1.0052`, DQN `1.2575`, PPO `1.3393`, centralized-critic `1.3006`), but they do not develop the intended trade economy. Q-learning has `trade_count_mean=0.2939`; DQN, PPO, independent-DQN, and centralized-critic all have `trade_count_mean=0.0` in the full comparison table.
- Trade-attempt diagnostics mean the Resource Island deep-RL trade result should be framed as exploration/coordination failure, not as proven strategic trade avoidance. DQN makes rare attempts on `none` (`trade_attempt_count_mean=0.2850`) and centralized-critic also makes rare attempts (`0.1760`), but all are inventory-blocked; PPO makes no trade attempts at all. The safe claim is that the neural/MARL minds find higher survival/welfare policies without learning successful trade under the current reward/observation/training horizon.
- The old DQN and independent-DQN rows are exactly identical in the Resource Island full comparison because they were generated before the decorrelation fix. The full Resource Island independent-DQN row must be rerun before final thesis claims.

#### 2. Auction House

- [ ] `worlds/auction_house/DESIGN.md`: lock valuation model, bid grid, information structure, tie-breaking, payment rules, and benchmark assumptions.
- [x] `worlds/auction_house/env.py`: mechanics skeleton with private values and discrete bids.
- [x] Auction mechanisms: first-price and second-price/Vickrey sealed-bid mechanics.
- [ ] Auction mechanisms: optional simple clock auction.
- [ ] `worlds/auction_house/benchmarks.py`: closed-form or computed truthful/strategic bidding benchmarks where theory supplies them.
- [ ] `QLearningMind` runs unchanged on Auction House.
- [x] Unit tests cover allocation, payment, and tie-breaking.
- [ ] Unit tests cover benchmark calculations after theory lock.
- [ ] Smoke run and then n=20 multiseed run once mechanics are clean.

#### 3. Public Goods / Commons

- [ ] `worlds/public_goods/DESIGN.md`: lock public pool dynamics, contribution/extraction actions, regeneration, observability, and bracketing benchmarks.
- [ ] `worlds/public_goods/env.py`: contribution/extraction game with public resource dynamics.
- [ ] `worlds/public_goods/benchmarks.py`: social optimum and free-rider equilibrium/bracketing benchmarks.
- [ ] Institutions: reputation systems, penalty schedules, contribution matching, information restrictions.
- [ ] Metrics: sustainability, extraction rate relative to regeneration, contribution rate, and tragedy-of-commons threshold detection.
- [ ] `QLearningMind` runs unchanged on Public Goods.
- [ ] Smoke run and then n=20 multiseed run once mechanics are clean.

#### 4. Labor Market

- [ ] `worlds/labor_market/DESIGN.md`: lock worker/employer types, preference/payoff generation, action spaces, information structure, and matching protocol.
- [ ] `worlds/labor_market/env.py`: asymmetric-agent two-sided matching world.
- [ ] Deferred-acceptance-style matching institution exists.
- [ ] `worlds/labor_market/benchmarks.py`: truthful-revelation/strategy-proof benchmark for the proposing side where applicable.
- [ ] Tests cover matching validity, stability checks, payoff accounting, and asymmetric agent handling.
- [ ] `QLearningMind` or an explicit tabular baseline runs without changing the shared `Agent` interface.
- [ ] Smoke run and then n=20 multiseed run once mechanics are clean.

#### 5. Central Planner / Tax Schedule

- [ ] Decide explicitly whether this is a standalone `World` or a cross-world parameterized `Institution`; current default is institution/sweep, not new world.
- [ ] `institutions/tax_schedule.py`: configurable tax/redistribution mechanism if treated as an institution.
- [ ] Sweep tax/redistribution parameters across existing worlds rather than creating a new world unless a real planner state/action model is specified.
- [ ] Metrics: welfare, inequality, manipulation incidence where defined, and robustness under distribution shift.

## Phase 3 - Deep RL and MARL Minds

- [x] `dqn_mind.py`: PyTorch structured-observation DQN mind for discrete worlds
- [x] `ppo_mind.py`: PyTorch structured-observation discrete PPO mind
- [x] Validate both on Pricing Arena first
- [x] `independent_learners.py` exists and the registered `independent_dqn` mind now routes through a decorrelated independent-learners DQN baseline.
- [x] `centralized_critic.py`
- [x] Re-run exploitability protocol with deep RL minds
- [x] Compare exploitability: tabular Q vs DQN vs PPO vs MARL

Phase 3 notes:

- PyTorch is installed and pinned as `torch==2.12.1+cpu`; GPU is not required for these structured MLP experiments.
- The current repo still does not vendor the external Atari DQN or missile-interception PPO source trees. The active Phase 3 implementation is now PyTorch-native and uses the relevant audited update-rule structure adapted to structured observations/discrete actions.
- `run_multiseed.py` supports `--mind dqn`, `--mind ppo`, `--mind independent_dqn`, `--mind centralized_critic`, plus explicit fallback baselines `simple_dqn` and `simple_ppo`. Independent-DQN now routes through a decorrelated independent-learners coordinator, but existing full Pricing Arena independent-DQN outputs predate that fix and must be rerun.
- `run_exploitability.py` supports the same values through `--incumbent-mind`; existing full Pricing Arena independent-DQN exploitability outputs also predate the decorrelation fix and must be rerun.
- Best-response validation is intentionally one-step (`gamma=0`) against a fixed opponent action in Pricing Arena. It checks that DQN/PPO can recover the computed profit-maximizing discrete action before multi-agent smoke results are trusted.
- Phase 3 PyTorch smoke runs cover all 6 current institutions with `n_seeds=2`; they validate execution and table production, not full statistical evidence.
- Full-scale Phase 3 n=20 replicated multiseed/exploitability runs are complete and validated in `outputs/phase3_full/validation_report.json`.
- Resource Island v0 tabular-Q validation exists, Resource Island Phase 3 torch cross-world wiring passes validation and all-institution smoke tests, and full Resource Island n=20 cross-mind torch runs are complete. Use `outputs/resource_island_phase3_full/distinct_mind_comparison.csv` for thesis-facing comparisons.
- Independent-DQN now diverges from DQN in tests and smoke gates. Existing full independent-DQN result tables remain stale until fixed reruns replace them.

## Phase 6 - LLM Agents and Hybrids

- [ ] `prompted_planner.py`
- [ ] `memory_buffer.py`
- [ ] Reduced-seed, higher-scrutiny LLM evaluation protocol
- [ ] `llm_rl_hybrid.py`
- [ ] Head-to-head: LLM mind vs Q-learning/DQN mind
- [ ] Investigate whether LLM agents collude via different mechanisms than Q-learners

## Phase 7 - Visualization Layer

- [ ] `trajectory_to_json.py`
- [ ] Vite + React + canvas app
- [ ] `AgentInspector`
- [ ] `MetricsDashboard`
- [ ] `ReplayPlayer`
- [ ] Public demo deploy

## Phase 8 - Cross-Cutting Research Questions

- [x] Initial Pricing Arena answer: price cap reduces exploitability for every tested mind class in the n=20 Phase 3 table.
- [x] Initial Pricing Arena answer: price cap's collusion effect is capability- and metric-sensitive; DQN sits near the cap with low price dispersion and retains higher profit-normalized collusion than PPO/random, so price-based and profit-based collusion tell different stories for stronger learners. The matching independent-DQN row is a DQN-alias row, not separate evidence.
- [x] Initial Resource Island answer: corrected v0 produces nonzero trade attempts and sparse successful trades, but property-rights and trade-price-control institutions remain mechanically under-exercised; redistribution slightly lowers welfare/survival and reputation is close to baseline under tabular Q.
- [x] Resource Island cross-mind full answer: stronger neural/MARL minds improve survival and welfare in Resource Island, but they do so without learning successful trade under the current reward/observation/training horizon. In the full n=20 table, Q-learning is the only mind with nonzero successful trades on `none` (`trade_count_mean=0.2939`), while DQN, PPO, independent-DQN, and centralized-critic all have `trade_count_mean=0.0`; DQN and centralized-critic make rare inventory-blocked attempts, while PPO makes no attempts.
- [x] Initial cross-world synthesis: capability effects are real in both worlds, but their form is world-specific. In Pricing Arena, stronger learners can exploit a price cap through a profit/quantity channel; in Resource Island, stronger learners improve survival/welfare while failing to activate the trade/property institutions, so the non-pricing result should be framed as a coordination/exploration limitation of the current world setup rather than a clean institution-effect comparison.
- [ ] Does institution design that reduces collusion also reduce exploitability across non-pricing worlds?
- [ ] Does agent capability make institutions more or less robust across non-pricing worlds?
- [ ] General mechanism robustness under learning capability curve
- [ ] Resource Island specialization/inequality vs Pricing Arena collusion tendency with stronger Resource Island institution activation
- [ ] Institutions robust across all worlds vs world-specific

## Phase 9 - Publication / Output Targets

- [ ] Undergrad thesis: Environment 01 full results + full architecture as roadmap
- [ ] Workshop paper: Pricing Arena + first validated Environments Track world cross-world generalization result
- [ ] arXiv preprint: full platform paper once 3+ worlds and 3+ mind classes are validated
- [ ] Open-source release of `agent-economies` repo with docs, reproducible configs
- [ ] Conference target once LLM-agent results exist
