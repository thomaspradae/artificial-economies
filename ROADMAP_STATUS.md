# Artificial Economies Roadmap Status

Status rule: an item is checked only if it is implemented in this repo, covered by tests or full-run output validation, and the validation has been observed. Existing behavior in `arena_v0.py` does not count as completion of a requested refactor item unless the requested target file/interface exists.

## Validation Evidence

- [x] Core v0 tests pass: `test_arena_v0.py` ran 7 tests, all passing.
- [x] Multiseed tests pass: `test_multiseed.py` ran 6 tests, all passing.
- [x] Exploitability tests pass: `test_exploitability.py` ran 6 tests, all passing.
- [x] Phase 0 architecture tests pass: `test_phase0_core.py` ran 18 focused tests and is included in full discovery, covering core interfaces, registry composition, pricing-world parity, mind wrappers, institution transforms, logging, and combined-table construction.
- [x] Phase 3 deep-RL/MARL tests pass: `test_phase3_deep_rl.py` ran 10 tests and is included in full discovery, covering feature encoding, PyTorch DQN, PyTorch PPO, explicit NumPy fallback baselines, decorrelated independent learners, centralized critic, Pricing Arena integration, and validation-run output.
- [x] Resource Island tests pass: `test_resource_island.py` ran 25 tests, all passing, covering metrics, benchmarks, movement/collision, gathering/starvation, posted-offer trading, trade-radius behavior, institutions, Q-learning integration, corrected inventory-aware observations, v1 contested layouts, unequal trades, property-pressure counters, specialization rewards, and smoke output.
- [x] Resource Island Phase 3 cross-mind tests pass: `test_resource_island_phase3.py` ran 7 tests, all passing, covering fixed-width structured observations, DQN, PPO, independent-DQN, centralized-critic, validation-run output, smoke-run mind selection, DQN-vs-independent-DQN divergence, and mind-comparison table construction.
- [x] Auction House mechanics/economics tests pass: `test_auction_house.py` ran 15 tests, all passing, covering allocation, first-price, second-price, clock-auction payment resolution, reserve/no-sale behavior, information-policy observation transforms, deterministic tie-breaking, registry construction, truthful second-price benchmarks, first-price bid-shading benchmarks, ex-post regret, Q-learning training smoke, fixed-width structured observations, neural/MARL mind wiring, and generic mind-comparison construction.
- [x] Public Goods / Commons tests pass: `test_public_goods.py` ran 13 tests, all passing, covering extraction rationing, contribution/extraction accounting, penalty binding, contribution matching, information restriction, tax schedule redistribution, benchmark brackets, registry construction, Q-learning training, smoke output, institution-effect classification, fixed-width structured observations, and neural/MARL mind wiring.
- [x] Labor Market tests pass: `test_labor_market.py` ran 14 tests, all passing, covering deferred acceptance, blocking-pair detection, report-top preference construction, truthful matching stability, payoff accounting, canonical fixed preference profiles, strategy-proof worker report enumeration, registry construction, Q-learning training, smoke output, fixed-width structured worker observations, and neural/MARL mind wiring for the asymmetric worker-only learner setup.
- [x] Full unittest discovery passes: 142 tests, all passing.
- [x] Python compile check passes for the legacy v0 module, experiment scripts, new `core/`, `worlds/`, `minds/`, `institutions/`, Resource Island modules, Auction House modules/training runner, Public Goods modules/training runner, Labor Market modules/training runner, deep-RL/MARL modules, validation runners, combined-table builder, and all test files.
- [x] Phase 0 hot-path parity smoke check passes: old-path vs new-architecture multiseed CSVs are byte-identical for fixed seeds.
- [x] Phase 0 hot-path parity smoke check passes: old-path vs new-architecture exploitability CSVs are byte-identical for fixed seeds.
- [x] Phase 3 guard parity passes: Q-learning multiseed/exploitability CSVs remain byte-identical to the saved Phase 0 new-path parity outputs after adding PyTorch deep-RL/MARL support.
- [x] PyTorch installed and pinned: `.venv` imports `torch==2.12.1+cpu`, CUDA unavailable/unused, and a trivial forward/backward smoke test is covered in `test_phase3_deep_rl.py`.
- [x] Full multiseed output exists at `outputs/full_v0_multiseed/` with `steps=40000`, `n_seeds=20`, all 6 mechanisms, Student-t 95% CI, CSV summaries, manifest, and plots.
- [x] Full exploitability output exists at `outputs/v1_exploitability/` with `incumbent_steps=40000`, `adversary_steps=20000`, `evaluation_steps=5000`, `n_seeds=20`, `adversary_restarts=3`, all 6 mechanisms, CSV summaries, manifest, and plots.
- [x] Full random-mind multiseed baseline exists at `outputs/random_v0_multiseed/` with `steps=40000`, `n_seeds=20`, all 6 mechanisms, CSV summaries, manifest, and plots.
- [x] Full random-mind exploitability baseline exists at `outputs/random_v1_exploitability/` with `incumbent_steps=40000`, `adversary_steps=20000`, `evaluation_steps=5000`, `n_seeds=20`, `adversary_restarts=3`, all 6 mechanisms, CSV summaries, manifest, and plots.
- [x] Combined Phase 1 table exists at `outputs/combined_phase1/institution_summary.csv` with Q-learning and random-mind rows for all 6 mechanisms.
- [x] Phase 3 PyTorch best-response validation exists at `outputs/phase3_validation_torch/best_response_validation.csv`; DQN and PPO learn the computed one-step best response at the validation seed, and independent-DQN has separate decorrelation tests.
- [x] Phase 3 PyTorch Pricing Arena smoke validation exists at `outputs/phase3_validation_torch/pricing_smoke_validation.csv`; DQN, PPO, independent-DQN, and centralized-critic all emit finite metrics.
- [x] Phase 3 torch-vs-NumPy qualitative comparison exists at `outputs/phase3_validation_torch/torch_vs_numpy_qualitative.csv`; DQN and PPO families both show price-cap collusion suppression directionally on matched smoke runs.
- [x] Phase 3 PyTorch all-institution smoke outputs exist at `outputs/phase3_smoke_torch/` for DQN, PPO, independent-DQN, and centralized-critic multiseed/exploitability runs, plus `mind_comparison.csv`.
- [x] Phase 3 full-run outputs exist for DQN, PPO, independent-DQN, and centralized-critic: all four have n=20 multiseed outputs and n=20 exploitability outputs with 3 adversary restarts. Fixed independent-DQN outputs have been pulled and validated locally at `outputs/independent_dqn_v0_multiseed_fixed/` and `outputs/independent_dqn_v1_exploitability_fixed/`.
- [x] Phase 3 full validation passes: `outputs/phase3_full/validation_report.json` reports `status=pass`, and the rebuilt `outputs/phase3_full/mind_comparison.csv` has 36 rows across Q-learning, random, DQN, PPO, fixed independent-DQN, and centralized-critic.
- [x] Phase 3 comparison table includes price, price-dispersion, total-profit, quantity, historical price-normalized `collusion_index_mean`, and literature-comparable `profit_collusion_index_mean` columns in `outputs/phase3_full/mind_comparison.csv`.
- [x] Thesis-facing Phase 3 comparison table is now the main `outputs/phase3_full/mind_comparison.csv`; the earlier `distinct_mind_comparison.csv` workaround is obsolete because fixed independent-DQN is no longer an alias row.
- [x] Resource Island smoke output exists at `outputs/resource_island_smoke/` with `steps=200`, `n_seeds=3`, five institutions, Q-learning agents, CSV summaries, and manifest.
- [x] Resource Island pre-observation-fix full output exists at `outputs/resource_island_full/` with `steps=40000`, `n_seeds=20`, five institutions, Q-learning agents, CSV summaries, and manifest; this is now treated as a diagnostic artifact, not final evidence for the corrected Resource Island implementation.
- [x] Resource Island corrected-observation smoke output exists at `outputs/resource_island_obsfix_smoke/` with `steps=500`, `n_seeds=3`, five institutions, inventory-aware tabular observations, trade/property diagnostic counters, CSV summaries, and manifest.
- [x] Resource Island full-run trigger diagnosis observed: `trade_count` is 0.0 for every institution and every seed in `outputs/resource_island_full/summary_by_seed.csv`; source audit found inventory-imbalance observations and 3D Q-table wiring are now present, but the saved full output predates the current `property_claims`/trade-diagnostic schema and should be rerun before drawing Resource Island institution conclusions.
- [x] Resource Island contact diagnosis observed: corrected-observation smoke had nonzero trade actions but zero successful trades because strict adjacency made contact rare and the complementary-offer protocol blocked on missing inventory.
- [x] Resource Island trade-fix smoke output exists at `outputs/resource_island_tradefix_smoke/` with nonzero successful trades, posted-offer trading, all-island default trade radius, and trade block diagnostics.
- [x] Resource Island corrected full output exists at `outputs/resource_island_tradefix_full/` with `steps=40000`, `n_seeds=20`, five institutions, inventory-aware observations, posted-offer trading, CSV summaries, and manifest.
- [x] Resource Island corrected full validation observed: `summary_by_seed.csv` has 100 rows, `summary_aggregate.csv` has 5 rows, every institution has 20 seeds, and no blank/NaN/inf values were found.
- [x] Resource Island Phase 3 validation output exists at `outputs/resource_island_phase3_validation/` with DQN/PPO single-agent sanity rows, Q-learning-vs-DQN qualitative comparison rows, and manifest.
- [x] Resource Island Phase 3 all-institution smoke outputs exist for DQN, PPO, independent-DQN, and centralized-critic at `outputs/resource_island_*_smoke/`; each has 15 by-seed rows, 5 aggregate rows, and no blank/NaN/inf values. Fixed smoke gates now confirm DQN and independent-DQN diverge.
- [x] Resource Island Phase 3 smoke comparison table exists at `outputs/resource_island_phase3_smoke/mind_comparison.csv` with 25 rows across Q-learning, DQN, PPO, independent-DQN, and centralized-critic for all five Resource Island institutions.
- [x] Resource Island Phase 3 full-run outputs exist for DQN, PPO, independent-DQN, and centralized-critic: all four have `steps=40000`, `n_seeds=20`, five institutions, CSV summaries, and manifests.
- [x] Resource Island Phase 3 full validation observed: each neural/MARL full run has 100 by-seed rows, 5 aggregate rows, exactly 20 seeds per institution, and no blank/NaN/inf values.
- [x] Resource Island Phase 3 full comparison table exists at `outputs/resource_island_phase3_full/mind_comparison.csv` with 25 rows across Q-learning, DQN, PPO, fixed independent-DQN, and centralized-critic for all five Resource Island institutions.
- [x] Thesis-facing Resource Island comparison table is now the main `outputs/resource_island_phase3_full/mind_comparison.csv`; the earlier distinct-mind workaround is obsolete after the fixed independent-DQN rerun.
- [x] Independent-DQN decorrelation fix validated: `test_phase3_deep_rl.py` covers decorrelated independent-DQN Q-values from one base seed, `test_resource_island_phase3.py` confirms Resource Island DQN and independent-DQN rollouts diverge, and `outputs/resource_island_independent_dqn_smoke_fixed_gate/summary_aggregate.csv` differs from the matching DQN smoke gate.
- [x] Auction House smoke output exists at `outputs/auction_house_smoke/` with `steps=1000`, `n_seeds=2`, second-price, first-price, and second-price-with-reserve scenarios, Q-learning bidders, summary CSVs, learned bid curves, and manifest.
- [x] Auction House validation output exists at `outputs/auction_house_validation/` with `steps=10000`, `n_seeds=3`, all three sealed-bid scenarios, finite benchmark-comparison metrics, learned bid curves, and manifest.
- [x] Auction House full output exists at `outputs/auction_house_full/` with `steps=40000`, `n_seeds=20`, second-price, first-price, and second-price-with-reserve scenarios, 60 by-seed rows, 3 aggregate rows, 1320 bid-curve rows, and no blank/NaN/inf values.
- [x] Auction House variant smoke output exists at `outputs/auction_house_variant_smoke/` with `steps=500`, `n_seeds=2`, clock, second-price public-signal, and second-price noisy-signal scenarios, 6 by-seed rows, 3 aggregate rows, 132 bid-curve rows, and no blank/NaN/inf values.
- [x] Auction House P.6 all-mind smoke outputs exist at `outputs/auction_house_phase3_smoke/` for DQN, PPO, fixed independent-DQN, and centralized-critic over six auction scenarios; each neural/MARL mind has 12 by-seed rows, 6 aggregate rows, and no blank/NaN/inf values, and `mind_comparison.csv` includes Q-learning plus all four ladder minds.
- [x] Auction House P.6 full outputs exist at `outputs/auction_house_phase3_full/` for DQN, PPO, fixed independent-DQN, and centralized-critic over six auction scenarios; each neural/MARL mind has 120 by-seed rows, 6 aggregate rows, a manifest, and no blank/NaN/inf values. The rebuilt `mind_comparison.csv` has 27 rows: Q-learning sealed-bid rows plus neural/MARL rows for sealed-bid, clock, public-signal, and noisy-signal variants.
- [x] Resource Island v1 smoke output exists at `outputs/resource_island_v1_pressure_smoke/` with `steps=500`, `n_seeds=3`, contested resources, complementary specialization preferences, 2-for-1 unequal trade offers, pressure-start inventories/positions, Q-learning agents, summary CSVs, and manifest.
- [x] Resource Island v1 validation output exists at `outputs/resource_island_v1_validation/` with `steps=5000`, `n_seeds=5`, contested resources, complementary specialization preferences, 2-for-1 unequal trade offers, pressure-start inventories/positions, Q-learning agents, summary CSVs, and manifest.
- [x] Resource Island v1 full output exists at `outputs/resource_island_v1_full/` with `steps=40000`, `n_seeds=20`, four institution variants, 80 by-seed rows, 4 aggregate rows, contested resources, unequal trades, property-pressure diagnostics, and no blank/NaN/inf values.
- [x] Public Goods smoke output exists at `outputs/public_goods_smoke/` with `steps=1000`, `n_seeds=3`, six institution variants, Q-learning agents, free-rider/social benchmark references, CSV summaries, and manifest.
- [x] Public Goods full output exists at `outputs/public_goods_full/` with `steps=40000`, `n_seeds=20`, six institution variants, 120 by-seed rows, 6 aggregate rows, and no blank/NaN/inf values.
- [x] Public Goods institution-effect validation exists at `outputs/public_goods_full/institution_effect_validation.json`, classifying which institutions change state metrics versus mostly reward/accounting metrics.
- [x] Public Goods P.6 all-mind smoke outputs exist at `outputs/public_goods_phase3_smoke/` for DQN, PPO, fixed independent-DQN, and centralized-critic over six institution variants; each neural/MARL mind has 12 by-seed rows, 6 aggregate rows, and no blank/NaN/inf values, and `mind_comparison.csv` includes Q-learning plus all four ladder minds.
- [x] Public Goods P.6 full outputs exist at `outputs/public_goods_phase3_full/` for DQN, PPO, fixed independent-DQN, and centralized-critic over six institution variants; each neural/MARL mind has 120 by-seed rows, 6 aggregate rows, a manifest, and no blank/NaN/inf values. The rebuilt `mind_comparison.csv` has 30 rows across Q-learning plus all four ladder minds.
- [x] Labor Market smoke output exists at `outputs/labor_market_smoke/` with `steps=1000`, `n_seeds=3`, deferred-acceptance matching, Q-learning workers, truthful matching benchmark references, CSV summaries, and manifest.
- [x] Labor Market full output exists at `outputs/labor_market_full/` with `steps=40000`, `n_seeds=20`, deferred-acceptance matching, 20 by-seed rows, 1 aggregate row, and no blank/NaN/inf values.
- [x] Labor Market fixed benchmark case report exists at `outputs/labor_market_benchmark_cases.json`, covering stable truthful matching, forced unstable matching, welfare accounting, and worker-proposing DA no-profitable-worker-report-deviation checks.
- [x] Labor Market P.6 all-mind smoke outputs exist at `outputs/labor_market_phase3_smoke/` for DQN, PPO, fixed independent-DQN, and centralized-critic over deferred acceptance; each neural/MARL mind has 2 by-seed rows, 1 aggregate row, and no blank/NaN/inf values, and `mind_comparison.csv` includes Q-learning plus all four ladder minds.
- [x] Labor Market P.6 full outputs exist at `outputs/labor_market_phase3_full/` for DQN, PPO, fixed independent-DQN, and centralized-critic over deferred acceptance; each neural/MARL mind has 20 by-seed rows, 1 aggregate row, a manifest, and no blank/NaN/inf values. The rebuilt `mind_comparison.csv` has 5 rows across Q-learning plus all four ladder minds.
- [x] Tax-schedule sweep output exists at `outputs/tax_schedule_sweep/` with `steps=40000`, `n_seeds=20`, Public Goods and Resource Island, four flat tax rates, 160 by-seed rows, 8 aggregate rows, and no blank/NaN/inf values.
- [x] Literature theory-scout pipeline exists at `tools/theory_scout/` with OpenAlex, Semantic Scholar, arXiv, Unpaywall/PDF, text-extraction, strict paper-card, ranking, and gap-table modules; `test_theory_scout.py` passes and `literature/` contains an OpenAlex-backed first cache, strict paper cards, `novelty_gap_table.csv`, and `theory_obligations.md`.
- [x] Preliminary cross-world synthesis outputs exist at `outputs/cross_world_synthesis_smoke/` with all five worlds represented, a capability-tier synthesis table, a monotonicity report, and a shared capability-ladder figure. Auction House, Public Goods, and Labor Market rows are smoke-scale for neural/MARL minds and are not final P.6 evidence.
- [x] Canonical five-world synthesis outputs exist at `outputs/cross_world_synthesis/` after importing the newer GitHub comparison builders. The generated artifacts include `synthesis_table.csv`, `baseline_capability_table.csv`, `monotonicity_report.json`, `coverage_report.json`, `protocol_comparability_report.json`, and `capability_ladder.png`.
- [x] `outputs/full_ladder_validation.json` reports `status=pass` for the currently available full ladder artifacts across Pricing Arena, Resource Island, Auction House, Public Goods, and Labor Market.
- [x] Publication-inference outputs exist at `outputs/publication_inference/`, including `publication_inference_all.csv`, `thesis_confirmatory_results.csv`, and `publication_inference_summary.md`; the summary applies Benjamini-Hochberg correction by world/effect/claim/metric family and excludes Resource Island cross-mind rows while the protocol-comparability audit fails.

## Running Work

- [x] No requested Auction House, Public Goods, or Labor Market P.6 full queues remain running. Their full ladder outputs have been pulled locally, validated for row counts and finite values, and rebuilt with the newer GitHub comparison builders.
- [ ] Resource Island apples-to-apples v1 ladder rerun is still active. PPO is complete on ofi1, DQN is complete on old1, centralized-critic is complete on old2, and old1 is currently running `independent_dqn` at `outputs/resource_island_v1_independent_dqn_full`; the ofi1 finalizer is still running and waiting to consolidate/validate/rebuild synthesis once that final output lands.

## Separate Atari DQN Audit Note

- [x] Stage 5A forward/model exchange pass observed after adding the DeepMind-faithful Atari network with `conv1 padding=1`; the historical local PyTorch Atari `QNetwork` remains documented as non-faithful because it used padding `0`.
- [x] Stage 5B Bellman-target pass observed: frozen Stage 5B batches match DeepMind `NeuralQLearner:getQUpdate` for selected `Q(s,a)`, target-network next-Q values, terminal masks, gamma, and final Bellman targets.
- [x] Stage 5C clipped TD-error / loss-semantics pass observed: raw TD errors, `clip_delta=1`, sparse selected-action output-gradient tensors, threshold controls around `±1`, and no-batch-mean DeepMind reduction semantics match the released learner path.
- [ ] Stage 5D network-backward gradient comparison is next; do not treat parameter gradients, RMSProp, or schedule as validated until their own audit gates pass.
- [ ] Atari preprocessing resize remains unresolved: Torch7 `image.scaleBilinear` still does not byte-match the current Python clone, so no exact `audit/pytorch/deepmind_preprocess.py` gate has passed yet.

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
- [x] `docs/P6_LADDER_PLAYBOOK.md`: reusable P.6 ladder wiring playbook exists, defining audit, structured observation, smoke, full-run, validation, comparison, and interpretation steps for all worlds.
- [x] `worlds/mind_ladder.py`: shared neural/MARL ladder helper exists for DQN, PPO, fixed independent-DQN, and centralized-critic construction, action dispatch, finite feature validation, and joint/single-agent updates across worlds.
- [x] `institutions/*.py`: six institution classes exist, subclass `Institution`, have focused transform tests, and are used by `PricingArenaWorld.step()`.
- [x] `build_combined_table.py`: merges multiseed and exploitability aggregate outputs into one institution-level comparison table, including price-normalized and profit-normalized collusion columns.
- [x] `run_phase3_validation.py`: best-response and pricing-smoke validation runner exists and writes CSV/manifest outputs.
- [x] `run_phase3_full.py`: managed Phase 3 full-run orchestrator exists, dry-run output matched the requested 8 commands, and the live batch was launched through it.
- [x] `validate_phase3_full.py`: Phase 3 full-output validator exists and passes on the completed full batch.
- [x] `worlds/resource_island/DESIGN.md`: Resource Island design lock exists with inventory-aware tabular observations, actions, rewards, boundaries, collision resolution, starvation, posted-offer trading, trade-radius semantics, institutions, benchmark, metrics, and validation rules.
- [x] `worlds/resource_island/env.py`: `ResourceIslandWorld` subclasses `World` and implements bounded-grid movement, simultaneous collision resolution, inventory-aware local tabular observations, inventory, energy, gathering, starvation/death, resource regeneration, posted-offer trading, configurable trade radius, unequal trade units, specialization preferences, pressure-start inventories, property/trade diagnostic counters, v1 property-opportunity counters, metrics, and render state.
- [x] `worlds/resource_island/resources.py`: Resource Island resource initialization, deterministic contested/split layouts, local visibility counting, and stochastic regeneration helpers exist and are tested through the world.
- [x] `worlds/resource_island/trading.py`: Resource Island adjacency and one-for-one complementary trade matching helpers exist and are tested through the world.
- [x] `worlds/resource_island/features.py`: fixed-width structured Resource Island observation encoder exists for PyTorch DQN/PPO/MARL minds and is unit-tested for grid-size-independent shape and finite values.
- [x] `worlds/resource_island/benchmarks.py`: efficient-gather upper bound and greedy full-information gather planner exist and have unit tests.
- [x] `worlds/resource_island/training.py`: Resource Island training loop supports Q-learning, DQN, PPO, independent-DQN, and centralized-critic with tabular or fixed structured observations as appropriate, including v1 activation counters in summaries.
- [x] `run_resource_island_smoke.py`: Resource Island smoke runner writes seed summaries, aggregate summaries, diagnostic counters, and manifest for `--mind q_learning`, `dqn`, `ppo`, `independent_dqn`, and `centralized_critic`; it supports v1 contested/split layouts, activation-pressure presets, complementary specialization preferences, unequal trade units, and trade-acquisition rewards.
- [x] `run_resource_island_phase3_validation.py`: Resource Island neural/MARL validation runner exists and writes single-agent sanity, qualitative comparison, and manifest outputs.
- [x] `build_resource_island_mind_comparison.py`: Resource Island mind x institution comparison builder exists and merges aggregate outputs without requiring pandas.
- [x] `build_cross_world_synthesis.py`: newer GitHub cross-world synthesis builder exists and writes `synthesis_table.csv`, `baseline_capability_table.csv`, `monotonicity_report.json`, `coverage_report.json`, `protocol_comparability_report.json`, and `capability_ladder.png` from canonical world comparison tables.
- [x] `build_world_mind_comparison.py`: newer GitHub comparison builder exists with automatic output discovery, seed-level uncertainty, paired institution effects, paired mind-vs-Q effects, historical Pricing Arena exploitability merging, and schema-driven metric handling.
- [x] `validate_full_ladders.py`, `world_metric_schemas.py`, and `build_publication_inference.py`: full-ladder validation, world metric schemas, and publication-inference helpers have been imported from the GitHub snapshot and focused tests pass under `.venv`.
- [x] `institutions/resource_island.py`: property rights, redistribution, trade price controls, and reputation system institutions exist, subclass `Institution`, and have focused tests.
- [x] `worlds/auction_house/DESIGN.md`: Auction House design lock exists with private values, action space, tie-breaking, first-price, second-price, clock-auction mechanics, information/noise variants, reserve/no-sale logic, benchmark assumptions, metrics, and validation rules.
- [x] `worlds/auction_house/env.py`: Auction House world exists, subclasses `World`, and implements single-item allocation, deterministic tie-breaking, reserve/no-sale logic, first-price payments, second-price payments, simple clock-auction clearing, institution-mediated private valuation-bin observations, revenue, bidder surplus, welfare, efficiency, regret, overbidding, underbidding, and bid-shading diagnostics.
- [x] `worlds/auction_house/features.py`: fixed-width structured Auction House bidder observation encoder exists and is unit-tested for shape and finite values.
- [x] `worlds/auction_house/benchmarks.py`: Auction House benchmark helpers implement truthful second-price/clock bidding, first-price uniform-IPV bid shading, deterministic allocation/payment helpers, ex-post grid regret, and exact expected outcomes over discrete valuation grids.
- [x] `worlds/auction_house/training.py`: Auction House training loop supports Q-learning, DQN, PPO, fixed independent-DQN, and centralized-critic with private or institution-modified valuation-bin observations, structured neural features, repeated independent value draws, final-window summaries, benchmark references, and learned tabular bid-curve extraction.
- [x] `run_auction_house_smoke.py`: Auction House runner writes seed summaries, aggregate summaries, learned bid curves, and manifest for second-price, first-price, reserve-price, clock, public-signal, and noisy-signal scenarios; it accepts `--mind q_learning`, `dqn`, `ppo`, `independent_dqn`, and `centralized_critic`.
- [x] `worlds/public_goods/DESIGN.md`: Public Goods / Commons design lock exists with pool dynamics, contribution/extraction action semantics, deterministic regeneration, institution hooks, bracketing benchmarks, metrics, and validation rules.
- [x] `worlds/public_goods/env.py`: Public Goods world exists, subclasses `World`, and implements public-pool stock, contribution costs, extraction rewards, proportional rationing, deterministic regeneration, collapse diagnostics, tabular observations, institution hooks, and render state.
- [x] `worlds/public_goods/features.py`: fixed-width structured Public Goods observation encoder exists and is unit-tested for shape and finite values.
- [x] `worlds/public_goods/benchmarks.py`: free-rider, social-optimum, extraction-rationing, and fixed-policy bracketing helpers exist and have unit tests.
- [x] `worlds/public_goods/training.py`: Public Goods training loop supports Q-learning, DQN, PPO, fixed independent-DQN, and centralized-critic with final-window summaries and benchmark references.
- [x] `run_public_goods_smoke.py`: Public Goods runner writes seed summaries, aggregate summaries, and manifest for no institution, penalty schedule, contribution matching, reputation, information restriction, and tax schedule variants; it accepts `--mind q_learning`, `dqn`, `ppo`, `independent_dqn`, and `centralized_critic`.
- [x] `worlds/labor_market/DESIGN.md`: Labor Market design lock exists with worker/employer roles, report-top action space, deferred-acceptance matching protocol, rewards, metrics, and validation rules.
- [x] `worlds/labor_market/env.py`: Labor Market world exists, subclasses `World`, and implements learning workers, fixed employer preferences, reported preference construction, deferred-acceptance matching, payoff accounting, blocking-pair/stability diagnostics, manipulation-gain diagnostics, and render state.
- [x] `worlds/labor_market/features.py`: fixed-width structured Labor Market worker observation encoder exists and is unit-tested for shape and finite values.
- [x] `worlds/labor_market/benchmarks.py`: preference-order, deferred-acceptance, blocking-pair, reported-preference, and truthful-matching benchmark helpers exist and have unit tests.
- [x] `worlds/labor_market/training.py`: Labor Market training loop supports Q-learning, DQN, PPO, fixed independent-DQN, and centralized-critic over learning workers while employers remain fixed preference holders.
- [x] `run_labor_market_smoke.py`: Labor Market runner writes seed summaries, aggregate summaries, and manifest for deferred-acceptance learning workers; it accepts `--mind q_learning`, `dqn`, `ppo`, `independent_dqn`, and `centralized_critic`.
- [x] `institutions/public_goods.py`: public-goods penalty schedule, contribution matching, reputation, and information-restriction institutions exist, subclass `Institution`, and have focused tests.
- [x] `institutions/auction_house.py`: Auction House information/noise policy institution exists, subclasses `Institution`, and is used by Auction House observation hooks.
- [x] `institutions/labor_market.py`: deferred-acceptance institution exists, subclasses `Institution`, and is used by `LaborMarketWorld`.
- [x] `institutions/tax_schedule.py`: generic progressive tax-and-redistribution institution exists, subclasses `Institution`, and is smoke-tested through Public Goods.
- [x] `run_tax_schedule_sweep.py`: flat tax/redistribution sweep runner exists for Public Goods and Resource Island, writes by-seed summaries, aggregate summaries, and manifest outputs.
- [x] `validate_public_goods_effects.py`: Public Goods full-run validator exists and classifies institution effects as state-changing, reward/accounting-only, or near-baseline.
- [x] `run_labor_market_benchmark_cases.py`: Labor Market fixed-case benchmark report writer exists for stable, unstable, welfare-accounting, and strategy-proofness cases.
- [x] `tools/theory_scout/`: reproducible literature obligation pipeline exists, using metadata APIs before any PDF/text work and producing cached records, ranked CSVs, strict paper-card templates, a novelty/gap table, and world-level theory obligations.
- [x] `scripts/pull_results.sh`: remote output puller exists and has been run successfully for old1 plus old2 through ofi1.
- [x] `outputs/phase3_full/mind_comparison.csv`: rebuilt full Phase 3 comparison table exists across Q-learning, random, DQN, PPO, fixed independent-DQN, and centralized-critic for all 6 Pricing Arena mechanisms.
- [x] Phase 3 price-cap divergence diagnosis: exploitability falls under price cap across tested mind families, while collusion effects are metric-sensitive for higher-capability minds. DQN sits effectively at the cap under price cap (`avg_price_mean=3.999038`, `price_dispersion_mean=0.259025`) and earns materially higher profit than uncapped (`profit_total_mean=291.703303` vs. `236.345482`), supporting a real cap/quantity-profit channel rather than only normalized-index noise. Fixed independent-DQN remains a distinct corroborating condition (`avg_price_mean=3.790038`, `profit_total_mean=272.954450`, `exploitability_mean=11.681067` under price cap).

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
- The literature section remains unchecked because prose is not finished. The theory-scout machinery now exists and generated first-pass obligations, but the strict paper cards still need manual extraction from the actual papers before citation claims are considered final.

## Environments Track - New World Playbook

This replaces the old separate Phase 2, Phase 4, and Phase 5 world sections. Resource Island, Auction House, Public Goods, and Labor Market are all instances of the same engineering recipe: design lock, `World` subclass, benchmark, existing-mind wiring, institutions, validation. Central Planner is tracked separately as a likely cross-world institution/parameter sweep, not as a default standalone world.

### Shared Playbook

- [ ] P.1 Design lock: write `worlds/<name>/DESIGN.md` before implementation, including observations, actions, rewards, static benchmark, arbitrary rule decisions, and tie-breaking/collision/boundary semantics where relevant.
- [ ] P.1 Economic validity lock: before a world can be called research-ready, write down why the incentives, action space, information structure, and institution hooks represent a meaningful economic mechanism rather than only a runnable toy.
- [ ] P.1 Literature/theory anchor: identify the canonical benchmark or theory result the world is meant to test against, or explicitly justify why only an oracle/bracketing benchmark is available.
- [ ] P.2 Core implementation: create `worlds/<name>/env.py` subclassing `core.world.World` with `reset()`, `step()`, and `get_metrics()`.
- [ ] P.2 Static benchmark: create `worlds/<name>/benchmarks.py` with the analytical, oracle, or bracketing benchmark used to interpret learned behavior.
- [ ] P.2 Metrics: reuse `core/metrics.py` where concepts transfer; add new metrics only when the new world defines them operationally.
- [ ] P.3 Mind wiring: run `minds/q_learning.py` unchanged in the new world; if the mind must change, treat it as an interface bug.
- [ ] P.3 Sanity case: verify the simplest single-agent or two-agent case behaves sensibly before adding institutions.
- [ ] P.4 Institutions: add environment-specific institutions as small `Institution` subclasses and reuse existing institution logic where the concept transfers.
- [ ] P.4 Institution activation diagnostics: every institution must either bind in observed runs or have a documented, measured reason why it did not bind. Do not treat inactive institutions as evidence of economic ineffectiveness.
- [ ] P.5 Mechanics tests: unit-test world-specific transition logic, allocation logic, matching logic, or movement logic.
- [ ] P.5 Benchmark tests: unit-test analytical benchmarks against hand-computed cases before using them as reference lines.
- [ ] P.5 Smoke run: run a short n=2-3 seed smoke experiment before any full run.
- [ ] P.5 Full run: run n=20 multiseed once smoke outputs are clean.
- [ ] P.5 Economic sanity report: after each full run, inspect whether learned behavior uses the intended strategic channel; if not, classify the result as a world-design diagnostic rather than a mechanism result.
- [ ] P.5 Cross-world abstraction test: the same `QLearningMind` class runs on the new world and at least one prior world without modification.
- [ ] P.6 Cross-mind pass: wire PyTorch DQN/PPO/MARL minds into new worlds after 2-3 worlds exist, rather than repeating deep-RL integration per world prematurely.
- [x] P.6 reusable ladder playbook/helper: `docs/P6_LADDER_PLAYBOOK.md`, `worlds/mind_ladder.py`, and `build_world_mind_comparison.py` now define the common pattern used after Pricing Arena and Resource Island.

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
- [x] P.6 structured-observation audit: DQN, PPO, independent-DQN, and centralized-critic are portable to Resource Island through explicit `obs_dim`/`action_dim`; centralized-critic actors now receive the same fixed `obs_dim` as the centralized critic.
- [x] P.6 fixed structured observations: Resource Island exposes a constant-width vector with normalized energy, inventory, position, local resource maps, nearby-agent context, and alive flag.
- [x] P.6 smoke pass: DQN, PPO, independent-DQN, and centralized-critic run on all five Resource Island institutions for `steps=500`, `n_seeds=3`, with clean CSV outputs.
- [x] P.6 smoke comparison table: `outputs/resource_island_phase3_smoke/mind_comparison.csv` compares Q-learning, DQN, PPO, independent-DQN, and centralized-critic on Resource Island.
- [x] P.6 full run: n=20 Resource Island replicated runs for DQN, PPO, fixed independent-DQN, and centralized-critic as distinct neural/MARL conditions.
- [x] P.6 fixed independent-DQN full rerun: `outputs/resource_island_independent_dqn_full_fixed/` exists and has been merged into `outputs/resource_island_phase3_full/mind_comparison.csv`.
- [x] P.6 full comparison: final Resource Island full cross-mind table has 25 rows across Q-learning, DQN, PPO, fixed independent-DQN, and centralized-critic.

Resource Island full-run note:

- The old `outputs/resource_island_full/` run used the original two-bin observation `(energy_bin, local_resource_bin)`. That observation did not expose inventory imbalance, so Q-learning could not condition trade offers on having food surplus or wood surplus. Treat that full run as a diagnostic artifact, not final Resource Island evidence.
- The corrected-observation smoke exposed inventory imbalance but still produced zero successful trades. Follow-up diagnostics showed the causes were strict-contact rarity and an overly brittle complementary-offer protocol.
- The current corrected full run uses posted one-sided offers and all-island default trade radius. Trade attempts are nonzero for every institution and every seed; successful trades remain sparse (`trade_count_mean` around 0.29 in the final window for most institutions), and almost all blocked attempts are inventory blocks rather than institution blocks.
- `property_rights` creates claims in every seed but has zero observed violations in the corrected full run. `trade_price_controls` has zero institution blocks because v0 only offers 1-for-1 trades, so there is no unequal exchange to block. Treat those as mechanism-inactive diagnostics, not evidence that the concepts are ineffective in richer Resource Island variants.
- Follow-up property-rights opportunity replay over 20 seeds x 40,000 steps found claims in 13/20 seeds but only 21 total non-owner-visible claimed-cell steps, only 9 claimed-resource-visible steps, and zero non-owner gather attempts on claimed cells. Property rights were therefore almost never put under behavioral pressure in v0.
- Source audit confirms `trade_price_controls` is structurally unable to bind in v0 because every constructed trade has `food_units=1` and `wood_units=1`, so the exchange ratio is always exactly 1. Unequal exchange offers are required before this institution can be evaluated.
- P.6 smoke results are execution/integration evidence, not final statistical Resource Island cross-mind evidence. At `n_seeds=3`, Q-learning has higher short-horizon welfare and more successful trades than DQN/PPO/MARL on the `none` institution. The old DQN/independent-DQN smoke match exposed an implementation defect; the fixed independent-DQN implementation now diverges from DQN in unit and smoke gates.
- P.6 full results show a stronger version of the smoke pattern. On `none`, neural/MARL minds have higher survival and welfare than Q-learning (`survival_rate_mean`: Q-learning `0.9686`, DQN `0.9875`, PPO `0.9931`, fixed independent-DQN `0.9937`, centralized-critic `0.9955`; `welfare_mean`: Q-learning `1.0052`, DQN `1.2575`, PPO `1.3393`, fixed independent-DQN `1.2372`, centralized-critic `1.3006`), but they do not develop the intended trade economy. Q-learning has `trade_count_mean=0.2939`; DQN, PPO, fixed independent-DQN, and centralized-critic all have `trade_count_mean=0.0` in the full comparison table.
- Trade-attempt diagnostics mean the Resource Island deep-RL trade result should be framed as exploration/coordination failure, not as proven strategic trade avoidance. DQN makes rare attempts on `none` (`trade_attempt_count_mean=0.2850`), fixed independent-DQN also makes rare attempts (`0.3020`), and centralized-critic makes rare attempts (`0.1760`), but all are inventory-blocked; PPO makes no trade attempts at all. The safe claim is that the neural/MARL minds find higher survival/welfare policies without learning successful trade under the current reward/observation/training horizon.

Resource Island v1 economic-hardening queue:

- [x] Add contested-resource layouts so property rights face real pressure: clustered scarce resource cells, repeated access incentives, and measured non-owner opportunities to gather from claimed cells.
- [x] Add unequal exchange offers or price vectors so `trade_price_controls` can actually bind; v0 one-for-one trades cannot test price-control economics.
- [x] Add specialization pressure by making agents heterogeneous in resource needs, gathering productivity, starting inventory, or local resource access.
- [ ] Run strict-local trade-radius ablations versus whole-island matching to separate spatial-friction effects from centralized-market effects.
- [x] Add institution activation thresholds to validation: trade attempts, successful trades, property opportunities, property violations, and institution-block counts must be reported before interpreting welfare differences.
- [x] V1 pressure smoke validates activation: `outputs/resource_island_v1_pressure_smoke/` shows successful trade under `none` (`trade_count_mean=3.45`), binding price controls under `trade_price_controls` (`trade_institution_blocked_count_mean=6.16`), property-right opportunities under `property_rights` (`property_opportunities_mean=10.32`, `property_resource_opportunities_mean=4.81`), and stronger reputation-mediated trade (`trade_count_mean=6.2367`, `welfare_mean=1.2565`).
- [x] V1 medium validation strengthens the activation result: `outputs/resource_island_v1_validation/` has 20 by-seed rows, 4 aggregate rows, no blank/NaN/inf values, successful trade under `none` (`trade_count_mean=6.0340`), binding price controls (`trade_institution_blocked_count_mean=5.7786`, `welfare_mean=0.8751`), active property-right opportunity exposure (`property_opportunities_mean=17.7336`, `property_resource_opportunities_mean=11.7134`), and reputation-mediated trade/welfare gains (`trade_count_mean=6.8672`, `welfare_mean=1.3130`).
- [x] Re-run n=20 after the v1 activation diagnostics show that the relevant institution channels are exercised: `outputs/resource_island_v1_full/` has 80 by-seed rows, 4 aggregate rows, no blank/NaN/inf values, successful trade under `none` (`trade_count_mean=5.3252`), active property-right pressure (`property_opportunities_mean=27.8249`), binding price controls (`trade_institution_blocked_count_mean=4.7862`, `trade_count_mean=0.0`), and reputation-mediated welfare gains (`welfare_mean=1.3477`).

#### 2. Auction House

- [x] `worlds/auction_house/DESIGN.md`: lock valuation model, bid grid, information structure, tie-breaking, payment rules, episode structure, bidder payoffs, reserve-price handling, and benchmark assumptions.
- [x] Auction House economic validity pass: verify the world is not just allocation code, but a clean auction-design testbed with private values, strategic bidding incentives, allocative efficiency metrics, bidder surplus, seller revenue, and benchmark deviations.
- [x] `worlds/auction_house/env.py`: mechanics implementation with private values, discrete bids, bidder-private valuation-bin observations, reserve/no-sale logic, and economic diagnostics.
- [x] Auction mechanisms: first-price and second-price/Vickrey sealed-bid mechanics.
- [x] Auction mechanisms: optional simple clock auction.
- [x] `worlds/auction_house/benchmarks.py`: closed-form or computed truthful/strategic bidding benchmarks where theory supplies them.
- [x] Second-price benchmark: truthful bidding is weakly dominant under private independent values; test allocation efficiency, expected revenue, bidder surplus, and regret from non-truthful bids on hand-computed cases.
- [x] First-price benchmark: implement a bid-shading reference for symmetric independent private values, plus a discrete-grid best-response/regret check when closed form is too coarse for the chosen grid.
- [x] Optional reserve-price benchmark: compute seller revenue and allocative-efficiency tradeoff under simple reserve prices before adding learning agents.
- [x] Metrics: revenue, allocative efficiency, bidder surplus, total welfare, regret to truthful bidding for second-price, regret/bid-shading distance for first-price, overbidding rate, underbidding rate, and allocation error.
- [x] Institution variants: first-price, second-price, and reserve-price sealed-bid auctions.
- [x] Institution variants: information disclosure/noise variant and optional clock auction smoke-tested after sealed-bid full runs became interpretable.
- [x] `QLearningMind` runs unchanged on Auction House.
- [x] Unit tests cover allocation, payment, and tie-breaking.
- [x] Unit tests cover benchmark calculations after theory lock.
- [x] Smoke run: confirm learned behavior moves in the expected direction before full runs, especially truthful bidding in second-price and shaded bidding in first-price.
- [x] Validation run: `outputs/auction_house_validation/` compares learned Q-learning behavior to truthful second-price, first-price shading, and reserve-price benchmarks.
- [x] Full run: `outputs/auction_house_full/` exists with `steps=40000`, `n_seeds=20`, all three sealed-bid scenarios, summary CSVs, learned bid curves, and manifest.
- [x] Economic sanity report: compare learned bidding curves against benchmarks before claiming any auction-design result.
- [x] P.6 structured-observation audit: Auction House exposes fixed-width bidder features and supports DQN, PPO, fixed independent-DQN, and centralized-critic through the shared ladder helper.
- [x] P.6 all-mind smoke pass: `outputs/auction_house_phase3_smoke/` exists with DQN, PPO, fixed independent-DQN, and centralized-critic rows for all six auction scenarios; `mind_comparison.csv` includes Q-learning plus the four ladder minds.
- [x] P.6 full run: n=20 Auction House replicated runs for DQN, PPO, fixed independent-DQN, and centralized-critic.
- [x] P.6 full comparison and economic interpretation: compare neural/MARL bidding curves, regret, allocative efficiency, and revenue against truthful and shaded-bid benchmarks.

Auction House validation note:

- The `steps=10000`, `n_seeds=3` validation run is economically coherent but not final evidence. Second-price and reserve-price learning move toward the benchmark direction, with lower regret than the short smoke (`second_price ex_post_regret_mean_mean=0.3010`, `second_price_reserve=0.1753`), but allocative efficiency remains below the truthful benchmark (`0.7470` and `0.8137` vs. `1.0`). First-price learning produces clear bid shading/underbidding (`underbid_rate_mean=0.7737`, `first_price_shading_distance_mean_mean=1.1397`) but still has high ex-post regret (`1.2978`). Treat this as a validated learning environment that needs longer/tuned full runs before making auction-design claims.
- The `steps=40000`, `n_seeds=20` full run is now validated. Second-price remains below the truthful efficiency benchmark (`allocative_efficiency_mean=0.7336`), first-price shows stronger revenue and underbidding (`revenue_mean=3.4529`, `underbid_rate_mean=0.8141`), and second-price with reserve raises seller revenue (`revenue_mean=3.9384`) while lowering realized welfare relative to the no-reserve second-price case.
- The variant smoke validates execution, not final auction-design evidence. Clock, public-signal, and noisy-signal scenarios all emit finite metrics. In the short run, public-signal second-price has lower allocative efficiency (`0.49`) than clock (`0.59`) or noisy-signal (`0.625`), so these variants need longer/tuned runs before interpretation.
- The P.6 full ladder is now interpretable. DQN and independent-DQN raise seller revenue in second-price/reserve settings but slightly reduce allocative efficiency relative to Q-learning; PPO has the lowest regret in second-price and reserve cases among the neural minds. Centralized-critic performs poorly in no-reserve second-price (`revenue_mean=0.0626`, `allocative_efficiency_mean=0.5218`, `ex_post_regret_mean_mean=1.4363`) and first-price (`revenue_mean=1.2798`, `allocative_efficiency_mean=0.5153`), but is less pathological under the reserve-price scenario. Economically, the auction ladder shows that higher-capability minds do not monotonically recover theory benchmarks; architecture matters, and centralized training can destabilize bidding in this small asymmetric-value game.

#### 3. Public Goods / Commons

- [x] `worlds/public_goods/DESIGN.md`: lock public pool dynamics, contribution/extraction actions, deterministic regeneration, observability, institution hooks, and bracketing benchmarks.
- [x] `worlds/public_goods/env.py`: contribution/extraction game with public resource dynamics, proportional extraction rationing, deterministic regeneration, collapse diagnostics, and institution hooks.
- [x] `worlds/public_goods/benchmarks.py`: social-optimum and free-rider bracketing benchmarks plus fixed-policy simulation helpers.
- [x] Institutions: reputation systems, penalty schedules, contribution matching, information restrictions.
- [x] Metrics: sustainability, extraction rate relative to regeneration, contribution rate, and tragedy-of-commons threshold detection.
- [x] `QLearningMind` runs unchanged on Public Goods through a two-index tabular observation.
- [x] Smoke run: `outputs/public_goods_smoke/` exists with `steps=1000`, `n_seeds=3`, six institution variants, summary CSVs, and manifest.
- [x] Full run: `outputs/public_goods_full/` exists with `steps=40000`, `n_seeds=20`, six institution variants, 120 by-seed rows, 6 aggregate rows, summary CSVs, and manifest.
- [x] Deeper institution-effect validation: `outputs/public_goods_full/institution_effect_validation.json` distinguishes state-changing institutions from reward/accounting-only effects.
- [x] P.6 structured-observation audit: Public Goods exposes fixed-width pool/recent-activity features and supports DQN, PPO, fixed independent-DQN, and centralized-critic through the shared ladder helper.
- [x] P.6 all-mind smoke pass: `outputs/public_goods_phase3_smoke/` exists with DQN, PPO, fixed independent-DQN, and centralized-critic rows for all six institution variants; `mind_comparison.csv` includes Q-learning plus the four ladder minds.
- [x] P.6 full run: n=20 Public Goods replicated runs for DQN, PPO, fixed independent-DQN, and centralized-critic.
- [x] P.6 full comparison and economic interpretation: test whether capability changes commons collapse severity, contribution discovery, and institution robustness.

Public Goods smoke note:

- The v0 smoke is mechanically clean and exposes the intended commons pressure: learned agents mostly extract while contribution remains low, leaving final-window sustainability near `0.087` under the baseline. Contribution matching slightly increases sustainability/welfare, penalty schedules reduce welfare under the current learned extraction pattern, reputation increases rewards through contributor bonuses, information restriction is currently close to baseline, and the tax schedule collects positive revenue while preserving aggregate welfare through redistribution. Treat this as smoke evidence, not a final institutional ranking.
- The full n=20 run confirms the commons pressure under baseline (`sustainability_mean=0.0892`, `contribution_total_mean=0.1418`). Contribution matching improves welfare and sustainability (`welfare_mean=2.0808`, `sustainability_mean=0.1048`), reputation is a strong reward-shaping institution (`welfare_mean=9.5750`), information restriction is close to baseline, and the tax schedule raises revenue while preserving baseline-scale welfare.
- The effect validator classifies `tax_schedule` as reward/accounting-only at the tested rates, while penalty, contribution matching, reputation, and information restriction change state metrics relative to baseline under the configured threshold. This should be read as a diagnostic classification, not a claim that every state change is welfare-improving.
- The P.6 full ladder shows a clear commons-learning pattern. PPO and centralized-critic almost never contribute under baseline (`contribution_total_mean` near zero), collapse rates approach one, and sustainability sits near the minimum (`0.0800`). DQN and fixed independent-DQN contribute more but still underperform Q-learning on baseline sustainability. Contribution matching works best for fixed independent-DQN (`welfare_mean=2.2689`, `sustainability_mean=0.1210`, `contribution_total_mean=0.4128`), while reputation mostly increases rewards rather than sustainability for neural minds. Economically, stronger learners do not automatically solve the commons; some become more efficient free riders unless the institution creates a discoverable contribution incentive.

#### 4. Labor Market

- [x] `worlds/labor_market/DESIGN.md`: lock worker/employer types, preference/payoff generation, report-top action space, information structure, and matching protocol.
- [x] `worlds/labor_market/env.py`: asymmetric-agent two-sided matching world with learning workers and fixed employer preferences.
- [x] Deferred-acceptance-style matching institution exists.
- [x] `worlds/labor_market/benchmarks.py`: truthful-revelation/deferred-acceptance benchmark and blocking-pair stability checks exist.
- [x] Tests cover matching validity, stability checks, payoff accounting, and asymmetric agent handling.
- [x] `QLearningMind` runs without changing the shared `Agent` interface.
- [x] Smoke run: `outputs/labor_market_smoke/` exists with `steps=1000`, `n_seeds=3`, summary CSVs, and manifest.
- [x] Full run: `outputs/labor_market_full/` exists with `steps=40000`, `n_seeds=20`, deferred-acceptance matching, 20 by-seed rows, 1 aggregate row, summary CSVs, and manifest.
- [x] Stronger benchmark cases: fixed stable, forced unstable, welfare-accounting, and worker-proposing strategy-proofness cases exist in code and `outputs/labor_market_benchmark_cases.json`.
- [x] P.6 structured-observation audit: Labor Market exposes fixed-width worker preference/match-history features and supports DQN, PPO, fixed independent-DQN, and centralized-critic over the learning-worker side.
- [x] P.6 all-mind smoke pass: `outputs/labor_market_phase3_smoke/` exists with DQN, PPO, fixed independent-DQN, and centralized-critic rows for deferred acceptance; `mind_comparison.csv` includes Q-learning plus the four ladder minds.
- [x] P.6 full run: n=20 Labor Market replicated runs for DQN, PPO, fixed independent-DQN, and centralized-critic.
- [x] P.6 full comparison and economic interpretation: test whether capability changes truthful-report rates, stability, and welfare under worker-proposing deferred acceptance.

Labor Market smoke note:

- The v0 smoke validates mechanics, not a final matching-market result. Match rate is `1.0`, while stability and truthful-report rates vary strongly by seed (`stability_mean=0.65`, `truthful_report_rate_mean=0.551`). This is useful: the report-top action space creates a real manipulation/stability channel, but the current random-preference setup needs more seeds and possibly fixed benchmark cases before drawing economic conclusions.
- The full n=20 run is more stable and interpretable than the smoke: match rate remains `1.0`, `stability_mean=0.9508`, `truthful_report_rate_mean=0.7859`, and `total_welfare_mean=3.6393`. The report-top action space still creates manipulation/truthfulness variation, but deferred acceptance produces mostly stable matchings under these settings.
- Fixed benchmark cases clarify the economics: worker-proposing deferred acceptance should not have profitable worker-side report deviations, so future manipulation tests should either target the non-proposing side, another matching mechanism, or explicit information/commitment changes rather than pretending worker misreports under DA should be profitable.
- The P.6 full ladder validates the asymmetric-worker setup and gives a useful mechanism-design result. DQN slightly improves stability over Q-learning (`0.9750` vs. `0.9508`) while lowering truthful-report rates (`0.7039` vs. `0.7859`). PPO lowers truthfulness further (`0.5751`) without improving welfare. Fixed independent-DQN has the best non-centralized welfare (`3.6584`) with moderate stability (`0.9547`). Centralized-critic has the highest welfare (`3.8206`) but much worse stability (`0.7492`) and truthfulness (`0.4331`). Economically, deferred acceptance keeps match rates at one, but stronger learners trade off truthfulness/stability against welfare in this report-top formulation.

#### 5. Central Planner / Tax Schedule

- [x] Decide explicitly whether this is a standalone `World` or a cross-world parameterized `Institution`; current default is institution/sweep, not new world.
- [x] `institutions/tax_schedule.py`: configurable tax/redistribution mechanism if treated as an institution.
- [x] Sweep tax/redistribution parameters across existing worlds rather than creating a new world unless a real planner state/action model is specified.
- [x] Initial metrics: Public Goods smoke reports tax revenue, welfare, inequality, sustainability, contribution, and extraction under the `tax_schedule` institution.
- [x] Full sweep metrics: welfare, inequality, sustainability/survival, and tax revenue are reported for Public Goods and Resource Island at flat tax rates `0.0`, `0.1`, `0.25`, and `0.4`.

Central Planner / Tax Schedule note:

- The first implementation intentionally treats the central planner as a parameterized institution, not a standalone world. A standalone planner world should only be added if the planner receives its own state/action model. The completed `outputs/tax_schedule_sweep/` run shows tax revenue increasing mechanically with tax rates while aggregate welfare stays roughly stable in Public Goods and Resource Island under the current reward definitions.

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
- `run_multiseed.py` supports `--mind dqn`, `--mind ppo`, `--mind independent_dqn`, `--mind centralized_critic`, plus explicit fallback baselines `simple_dqn` and `simple_ppo`. Independent-DQN now routes through a decorrelated independent-learners coordinator, and fixed full Pricing Arena independent-DQN outputs have replaced the stale rows in `outputs/phase3_full/mind_comparison.csv`.
- `run_exploitability.py` supports the same values through `--incumbent-mind`; fixed full Pricing Arena independent-DQN exploitability outputs have been pulled and merged into the main comparison table.
- Best-response validation is intentionally one-step (`gamma=0`) against a fixed opponent action in Pricing Arena. It checks that DQN/PPO can recover the computed profit-maximizing discrete action before multi-agent smoke results are trusted.
- Phase 3 PyTorch smoke runs cover all 6 current institutions with `n_seeds=2`; they validate execution and table production, not full statistical evidence.
- Full-scale Phase 3 n=20 replicated multiseed/exploitability runs are complete and validated in `outputs/phase3_full/validation_report.json`.
- Resource Island v0 tabular-Q validation exists, Resource Island Phase 3 torch cross-world wiring passes validation and all-institution smoke tests, and full Resource Island n=20 cross-mind torch runs are complete. Use `outputs/resource_island_phase3_full/mind_comparison.csv` for thesis-facing comparisons.
- Independent-DQN now diverges from DQN in tests, smoke gates, Pricing Arena full outputs, and Resource Island full outputs.

## Phase 6 - LLM Agents and Hybrids

- [ ] `prompted_planner.py`
- [ ] `memory_buffer.py`
- [ ] Reduced-seed, higher-scrutiny LLM evaluation protocol
- [ ] `llm_rl_hybrid.py`
- [ ] Head-to-head: LLM mind vs Q-learning/DQN mind
- [ ] Investigate whether LLM agents collude via different mechanisms than Q-learners

## Phase 6.5 - Theory Engine / Paper Scout

- [x] `tools/theory_scout/`: reproducible literature metadata pipeline exists with OpenAlex, Semantic Scholar, arXiv, optional PDF resolution/download, text extraction, paper-card generation, ranking, and novelty-gap table construction.
- [x] `literature/queries.yaml`: world-specific literature queries exist for Pricing Arena, Resource Island, Auction House, Public Goods, and Labor Market.
- [x] `literature/secrets.env.example`: ignored secrets-file template exists for OpenAlex and Semantic Scholar credentials.
- [x] Local `literature/secrets.env` is configured outside git and test calls against OpenAlex and Semantic Scholar succeeded.
- [x] First cached outputs exist: `literature/papers_raw.jsonl`, `literature/papers_ranked.csv`, `literature/novelty_gap_table.csv`, strict paper-card templates, and `literature/theory_obligations.md`.
- [x] Local LLM host benchmark exists at `literature/local_llm_benchmark_ofi1.md`: `ofi1` has a user-local Ollama `0.32.1` install, `llama3.2:3b` is the recommended bulk JSON extractor, and `qwen3:8b` is reserved for slower audit passes.
- [x] `tools/theory_scout/fill_paper_cards.py` and `ollama_client.py`: local-LLM paper-card filler exists, uses Ollama `/api/chat` JSON mode with `think=false`, falls back from extracted text to abstracts, records source basis/model/speed metadata, and fails closed on invalid JSON.
- [x] `scripts/run_theory_llm_fill_ofi1.sh`: SSH-tunnel helper exists for using `ofi1`'s Ollama server from this local checkout.
- [x] `tools/theory_scout/audit_obligations.py`: deterministic obligation checker exists and writes `literature/obligation_audit.csv`, `literature/obligation_audit.md`, and `literature/theory_gap_report.csv`.
- [x] Theory-engine validation observed: `test_theory_scout.py` runs 14 tests, compile check passes for `tools/theory_scout`, one real `ofi1` LLM card-fill smoke succeeded, and `audit-obligations` reports `pass=10`, `partial=2`, `missing=0`.
- [ ] Run the full Semantic Scholar-enriched scout with the configured secrets and refresh the cached literature outputs after rate limits clear.
- [ ] Bulk-fill the main paper cards from paper text or abstracts: Pricing RL collusion, auctions/deep-RL auction design, public goods/MARL, matching/strategy-proofness, common-pool resources/property rights, and LLM economic agents.
- [ ] Validate PDF downloading/text extraction at scale and prefer full-paper text over abstracts for high-value cards.
- [ ] Add stronger citation/evidence guards: each filled paper-card claim should cite a metadata id, DOI/arXiv id when available, and a short source excerpt tied to the exact extracted claim.

Theory Engine note:

- The current system now finds, structures, fills, and audits theory obligations, but it is still not a finished deep-reading system. Most paper cards remain templates, PDF/text extraction has not been validated at scale, and filled cards require human review before citation. The important change is that related work now produces concrete code/result obligations instead of generic summaries.

## Phase 7 - Visualization Layer

- [ ] `trajectory_to_json.py`
- [ ] Vite + React + canvas app
- [ ] `AgentInspector`
- [ ] `MetricsDashboard`
- [ ] `ReplayPlayer`
- [ ] Public demo deploy

## Phase 8 - Cross-Cutting Research Questions

- [x] Initial Pricing Arena answer: price cap reduces exploitability for every tested mind class in the n=20 Phase 3 table.
- [x] Initial Pricing Arena answer: price cap's collusion effect is capability- and metric-sensitive; DQN sits near the cap with low price dispersion and retains higher profit-normalized collusion than PPO/random, so price-based and profit-based collusion tell different stories for stronger learners. Fixed independent-DQN is now a distinct corroborating condition: it remains near the cap under price-cap regulation, with lower profit than plain DQN but higher profit-normalized collusion than PPO.
- [x] Initial Resource Island answer: corrected v0 produces nonzero trade attempts and sparse successful trades, but property-rights and trade-price-control institutions remain mechanically under-exercised; redistribution slightly lowers welfare/survival and reputation is close to baseline under tabular Q.
- [x] Resource Island cross-mind full answer: stronger neural/MARL minds improve survival and welfare in Resource Island, but they do so without learning successful trade under the current reward/observation/training horizon. In the full n=20 table, Q-learning is the only mind with nonzero successful trades on `none` (`trade_count_mean=0.2939`), while DQN, PPO, independent-DQN, and centralized-critic all have `trade_count_mean=0.0`; DQN and centralized-critic make rare inventory-blocked attempts, while PPO makes no attempts.
- [x] Initial cross-world synthesis: capability effects are real in both worlds, but their form is world-specific. In Pricing Arena, stronger learners can exploit a price cap through a profit/quantity channel; in Resource Island, stronger learners improve survival/welfare while failing to activate the trade/property institutions, so the non-pricing result should be framed as a coordination/exploration limitation of the current world setup rather than a clean institution-effect comparison.
- [x] Full-data five-world synthesis scaffold: `outputs/cross_world_synthesis/` combines full ladder outputs for Pricing Arena, Auction House, Public Goods, and Labor Market, plus the best currently available Resource Island ladder rows. The protocol-comparability report correctly marks Resource Island cross-mind capability claims as invalid until the active apples-to-apples v1 neural/MARL rerun finishes, while Auction House, Public Goods, Labor Market, and Pricing Arena are marked comparable for cross-mind capability claims.
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
