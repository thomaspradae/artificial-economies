# Theory Obligation Audit

This is a deterministic coverage check. `pass` means required files/columns/terms were observed. `partial` means there is implementation evidence but the obligation still needs human review. `missing` means the evidence was not found.

Summary: pass=10, partial=99, missing=0.

## auction_house

### benchmark: pass

- Obligation: Compare learned bidding to truthful second-price, shaded first-price, and reserve benchmarks.
- Code evidence: worlds/auction_house/benchmarks.py
- Output evidence: outputs/auction_house_full/summary_aggregate.csv
- Missing/review: none

### metrics: pass

- Obligation: Report revenue, allocative efficiency, welfare/surplus, regret, and bid-shading diagnostics.
- Code evidence: worlds/auction_house/env.py; worlds/auction_house/training.py
- Output evidence: outputs/auction_house_phase3_full/mind_comparison.csv
- Missing/review: none

### paper_benchmark: partial

- Obligation: 1961_counterspeculation_auctions_and_competitive_sealed_tenders.md: Truthful bidding is weakly dominant in a private-value second-price auction; efficient allocation should follow truthful reports.
- Code evidence: worlds/auction_house
- Output evidence: 53 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 1961_counterspeculation_auctions_and_competitive_sealed_tenders.md: Revenue, allocative efficiency, bidder surplus, welfare, truthful-bid regret, overbidding/underbidding rates.
- Code evidence: worlds/auction_house
- Output evidence: 53 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 1961_counterspeculation_auctions_and_competitive_sealed_tenders.md: Second-price benchmarks must include truthful bidding, allocative efficiency, and regret/IC proxy, not only bidder reward.
- Code evidence: worlds/auction_house
- Output evidence: 53 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 1981_optimal_auction_design.md: Myerson-style revenue maximization and reserve-price tradeoff under independent private values.
- Code evidence: worlds/auction_house
- Output evidence: 53 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 1981_optimal_auction_design.md: Seller revenue, allocative efficiency, welfare, no-sale rate, bidder surplus, regret.
- Code evidence: worlds/auction_house
- Output evidence: 53 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 1981_optimal_auction_design.md: Reserve-price scenarios must report the revenue/efficiency/welfare tradeoff against no-reserve second price.
- Code evidence: worlds/auction_house
- Output evidence: 53 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2019_optimal_auctions_through_deep_learning.md: Revenue maximization subject to incentive compatibility/regret and individual-rationality constraints.
- Code evidence: worlds/auction_house
- Output evidence: 53 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2019_optimal_auctions_through_deep_learning.md: Revenue, regret/IC violation, allocative efficiency, bidder surplus, generalization.
- Code evidence: worlds/auction_house
- Output evidence: 53 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2019_optimal_auctions_through_deep_learning.md: Auction House must include regret or incentive-compatibility proxies, revenue, allocative efficiency, and truthful/strategic benchmarks.
- Code evidence: worlds/auction_house
- Output evidence: 53 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2020_learning_in_repeated_auctions.md: Repeated strategic learning can diverge from one-shot auction predictions.
- Code evidence: worlds/auction_house
- Output evidence: 53 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2020_learning_in_repeated_auctions.md: Learning dynamics, final-window revenue/efficiency/regret, bid shading, convergence behavior.
- Code evidence: worlds/auction_house
- Output evidence: 53 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2020_learning_in_repeated_auctions.md: Treat Auction House as a repeated-learning environment and report final-window summaries plus learned bid curves.
- Code evidence: worlds/auction_house
- Output evidence: 53 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2022_artificial_intelligence_and_auction_design.md: Auction format changes learned bidding, bid shading, revenue, and efficiency.
- Code evidence: worlds/auction_house
- Output evidence: 53 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2022_artificial_intelligence_and_auction_design.md: Revenue, underbidding/bid shading, efficiency, surplus, regret, learned bid curves.
- Code evidence: worlds/auction_house
- Output evidence: 53 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2022_artificial_intelligence_and_auction_design.md: Compare first-price bid shading and second-price truthfulness-like behavior using learned bid curves and regret.
- Code evidence: worlds/auction_house
- Output evidence: 53 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

## cross_world_methods

### paper_benchmark: partial

- Obligation: 1992_q_learning.md: Algorithmic benchmark: Bellman optimality control with a discrete state-action value table.
- Code evidence: minds; worlds/mind_ladder.py
- Output evidence: 70 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 1992_q_learning.md: Repo-side behavioral metrics by world plus validation that the same tabular update rule runs unchanged across worlds.
- Code evidence: minds; worlds/mind_ladder.py
- Output evidence: 70 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 1992_q_learning.md: Reproduce the method-level Q-learning obligation inside this repo: explicit discrete state/action encoding, Q-table shape, TD target, epsilon schedule, and cross-world smoke/full evidence from minds/q_learning.py; cross-world Q-learning smoke and full runs..
- Code evidence: minds; worlds/mind_ladder.py
- Output evidence: 70 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 1993_multi_agent_reinforcement_learning_independent_versus_cooperative_agents.md: MARL baseline benchmark: each agent learns its own policy while treating other learners as part of the environment.
- Code evidence: minds; worlds/mind_ladder.py
- Output evidence: 70 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 1993_multi_agent_reinforcement_learning_independent_versus_cooperative_agents.md: Behavioral divergence from single-agent/shared baselines, coordination outcomes, welfare/exploitability/stability, and tests for seed/replay independence.
- Code evidence: minds; worlds/mind_ladder.py
- Output evidence: 70 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 1993_multi_agent_reinforcement_learning_independent_versus_cooperative_agents.md: Reproduce independent learning as a genuine MARL condition: no shared network, no shared replay buffer, no shared exploration stream, and no aliasing with plain DQN. Evidence must come from minds/marl/independent_learners.py; decorrelation tests; fixed independent-DQN full outputs..
- Code evidence: minds; worlds/mind_ladder.py
- Output evidence: 70 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2015_human_level_control_through_deep_reinforcement_learning.md: Method benchmark: neural Q-learning with replay, a target network, and TD-error minimization.
- Code evidence: minds; worlds/mind_ladder.py
- Output evidence: 70 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2015_human_level_control_through_deep_reinforcement_learning.md: Repo-side: finite task metrics, smoke/full outputs, DQN-vs-tabular comparisons, and tests for replay/target-network behavior.
- Code evidence: minds; worlds/mind_ladder.py
- Output evidence: 70 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2015_human_level_control_through_deep_reinforcement_learning.md: Do not reproduce Atari for this thesis. Reproduce the DQN method obligation: neural Q-values, target network, replay buffer, agent-local randomness, TD/Huber loss, and cross-world validation from minds/deep_rl/torch_dqn_mind.py; Phase 3 and Resource Island DQN tests..
- Code evidence: minds; worlds/mind_ladder.py
- Output evidence: 70 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2017_multi_agent_actor_critic_for_mixed_cooperative_competitive_environments.md: MARL benchmark: decentralized actors with a critic that can condition on broader joint or centralized information during training.
- Code evidence: minds; worlds/mind_ladder.py
- Output evidence: 70 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2017_multi_agent_actor_critic_for_mixed_cooperative_competitive_environments.md: World-level welfare/stability/exploitability plus comparison against DQN, PPO, and independent-DQN; special attention to asymmetric-agent worlds.
- Code evidence: minds; worlds/mind_ladder.py
- Output evidence: 70 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2017_multi_agent_actor_critic_for_mixed_cooperative_competitive_environments.md: Reproduce the interface-level obligation: centralized critic must be structurally distinct from independent learners, handle fixed observation dimensions, and either support or explicitly reject asymmetric-agent worlds. Evidence: minds/marl/centralized_critic.py; cross-world P.6 smoke/full outputs..
- Code evidence: minds; worlds/mind_ladder.py
- Output evidence: 70 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2017_proximal_policy_optimization_algorithms.md: No economic-theory benchmark. The method benchmark is empirical comparison against other online policy-gradient methods on standard RL tasks.
- Code evidence: minds; worlds/mind_ladder.py
- Output evidence: 70 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2017_proximal_policy_optimization_algorithms.md: Paper-side: average episodic return, learning curves, sample complexity, simplicity, and wall-time. Repo-side: world metrics and finite-run validation outputs.
- Code evidence: minds; worlds/mind_ladder.py
- Output evidence: 70 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2017_proximal_policy_optimization_algorithms.md: Do not reproduce Atari/MuJoCo for this thesis. Reproduce the method-level obligation: categorical discrete policy, clipped surrogate updates, old log probabilities, advantage estimation, value loss, entropy bonus, and repeated minibatch epochs, then compare PPO against Q-learning/DQN/MARL inside the artificial-economies worlds.
- Code evidence: minds; worlds/mind_ladder.py
- Output evidence: 70 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

## labor_market

### benchmark: pass

- Obligation: Verify deferred-acceptance stability and proposing-side strategy-proofness cases.
- Code evidence: worlds/labor_market/benchmarks.py; run_labor_market_benchmark_cases.py
- Output evidence: outputs/labor_market_benchmark_cases.json
- Missing/review: none

### metrics: pass

- Obligation: Report match rate, stability, truthfulness, welfare, and manipulation diagnostics.
- Code evidence: worlds/labor_market/env.py; worlds/labor_market/training.py
- Output evidence: outputs/labor_market_phase3_full/mind_comparison.csv
- Missing/review: none

### paper_benchmark: partial

- Obligation: 1962_college_admissions_and_the_stability_of_marriage.md: Deferred acceptance produces stable matchings with no blocking pairs under truthful preferences.
- Code evidence: worlds/labor_market
- Output evidence: 35 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 1962_college_admissions_and_the_stability_of_marriage.md: Match rate, stability, blocking pairs, welfare, truthful-report rate, manipulation diagnostics.
- Code evidence: worlds/labor_market
- Output evidence: 35 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 1962_college_admissions_and_the_stability_of_marriage.md: Verify matching validity and blocking-pair stability before interpreting learned reports.
- Code evidence: worlds/labor_market
- Output evidence: 35 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 1982_the_economics_of_matching_stability_and_incentives.md: Deferred acceptance is strategy-proof for the proposing side under standard assumptions but not for both sides.
- Code evidence: worlds/labor_market
- Output evidence: 35 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 1982_the_economics_of_matching_stability_and_incentives.md: Truthful-report rate, manipulation gain, stability, welfare, side-specific profitable deviations.
- Code evidence: worlds/labor_market
- Output evidence: 35 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 1982_the_economics_of_matching_stability_and_incentives.md: Do not interpret worker-side misreport profits under worker-proposing DA as expected theory; test side-specific incentives.
- Code evidence: worlds/labor_market
- Output evidence: 35 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2006_changing_the_boston_school_choice_mechanism.md: Mechanism changes can reduce manipulation and improve stability relative to manipulable assignment rules.
- Code evidence: worlds/labor_market
- Output evidence: 35 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2006_changing_the_boston_school_choice_mechanism.md: Truthfulness, stability, welfare, manipulation gain, blocking pairs.
- Code evidence: worlds/labor_market
- Output evidence: 35 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2006_changing_the_boston_school_choice_mechanism.md: Use fixed benchmark cases to separate mechanism manipulation from noisy learned reports.
- Code evidence: worlds/labor_market
- Output evidence: 35 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2007_deferred_acceptance_algorithms_history_theory_practice_and_open_questions.md: DA properties depend on which side proposes and on preference/information assumptions.
- Code evidence: worlds/labor_market
- Output evidence: 35 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2007_deferred_acceptance_algorithms_history_theory_practice_and_open_questions.md: Stability, match rate, truthfulness, manipulation gain, welfare.
- Code evidence: worlds/labor_market
- Output evidence: 35 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2007_deferred_acceptance_algorithms_history_theory_practice_and_open_questions.md: State which side proposes and what strategy-proofness claim is valid.
- Code evidence: worlds/labor_market
- Output evidence: 35 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2022_learn_to_match_with_no_regret_reinforcement_learning_in_markov_matching_markets.md: Learning-market performance should report regret/stability together, not reward alone.
- Code evidence: worlds/labor_market
- Output evidence: 35 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2022_learn_to_match_with_no_regret_reinforcement_learning_in_markov_matching_markets.md: Regret, stability, welfare, match rate, manipulation, convergence.
- Code evidence: worlds/labor_market
- Output evidence: 35 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2022_learn_to_match_with_no_regret_reinforcement_learning_in_markov_matching_markets.md: If using deep RL in Labor Market, report regret/stability jointly and handle worker/employer asymmetry explicitly.
- Code evidence: worlds/labor_market
- Output evidence: 35 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

## pricing_arena

### benchmark: pass

- Obligation: Report Nash and joint-profit price benchmarks.
- Code evidence: worlds/pricing_arena/benchmarks.py
- Output evidence: outputs/full_v0_multiseed/summary_aggregate.csv
- Missing/review: none

### metrics: pass

- Obligation: Report price-normalized and profit-normalized collusion, exploitability, welfare, price, and profit.
- Code evidence: core/metrics.py; build_combined_table.py
- Output evidence: outputs/phase3_full/mind_comparison.csv
- Missing/review: none

### paper_benchmark: partial

- Obligation: 1988_the_theory_of_industrial_organization.md: Static Bertrand/Nash pricing and joint-profit/collusive price benchmarks.
- Code evidence: worlds/pricing_arena
- Output evidence: 4 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 1988_the_theory_of_industrial_organization.md: Average price, welfare/profit, Nash and joint-profit benchmark gaps, price/profit collusion indices.
- Code evidence: worlds/pricing_arena
- Output evidence: 4 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 1988_the_theory_of_industrial_organization.md: Reproduce the IO benchmark role: every Pricing Arena result needs Nash/Bertrand and joint-profit reference lines before learned prices are interpreted.
- Code evidence: worlds/pricing_arena
- Output evidence: 4 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2020_artificial_intelligence_algorithmic_pricing_and_collusion.md: Profit-normalized collusion relative to competitive and monopoly/joint-profit benchmarks.
- Code evidence: worlds/pricing_arena
- Output evidence: 4 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2020_artificial_intelligence_algorithmic_pricing_and_collusion.md: Profit-normalized collusion index, price proxy collusion index, average price, total profit, welfare, exploitability.
- Code evidence: worlds/pricing_arena
- Output evidence: 4 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2020_artificial_intelligence_algorithmic_pricing_and_collusion.md: Report profit-normalized collusion, not only a price proxy, and interpret price-cap results using both price and profit channels.
- Code evidence: worlds/pricing_arena
- Output evidence: 4 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2021_algorithmic_collusion_with_imperfect_monitoring.md: Collusion under noisy monitoring and punishment/price-war dynamics.
- Code evidence: worlds/pricing_arena
- Output evidence: 4 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2021_algorithmic_collusion_with_imperfect_monitoring.md: Collusion, price dispersion, profit, exploitability, robustness under shock.
- Code evidence: worlds/pricing_arena
- Output evidence: 4 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2021_algorithmic_collusion_with_imperfect_monitoring.md: Treat demand shocks as an information/monitoring change and compare whether collusion/exploitability survive under the shock.
- Code evidence: worlds/pricing_arena
- Output evidence: 4 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2021_autonomous_algorithmic_collusion_q_learning_under_sequential_pricing.md: Sequential versus simultaneous pricing changes strategic observability and punishment opportunities.
- Code evidence: worlds/pricing_arena
- Output evidence: 4 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2021_autonomous_algorithmic_collusion_q_learning_under_sequential_pricing.md: Average prices, profit, collusion index, learning dynamics.
- Code evidence: worlds/pricing_arena
- Output evidence: 4 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2021_autonomous_algorithmic_collusion_q_learning_under_sequential_pricing.md: State explicitly that Pricing Arena uses simultaneous action semantics unless a sequential variant is implemented.
- Code evidence: worlds/pricing_arena
- Output evidence: 4 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2022_artificial_intelligence_algorithm_design_and_pricing.md: Learning-protocol changes can shift pricing outcomes between competitive and supra-competitive regimes.
- Code evidence: worlds/pricing_arena
- Output evidence: 4 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2022_artificial_intelligence_algorithm_design_and_pricing.md: Collusion indices, exploitability, price/profit/welfare, price dispersion, and mind-by-institution effects.
- Code evidence: worlds/pricing_arena
- Output evidence: 4 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2022_artificial_intelligence_algorithm_design_and_pricing.md: Interpret capability differences as algorithm-design effects and show matched comparisons across minds and institutions.
- Code evidence: worlds/pricing_arena
- Output evidence: 4 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2024_algorithmic_collusion_in_dynamic_pricing_with_deep_reinforcement_learning.md: Neural agents may reproduce, attenuate, or alter tabular algorithmic collusion patterns.
- Code evidence: worlds/pricing_arena
- Output evidence: 4 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2024_algorithmic_collusion_in_dynamic_pricing_with_deep_reinforcement_learning.md: Price/profit collusion, average price, profit, welfare, exploitability, dispersion.
- Code evidence: worlds/pricing_arena
- Output evidence: 4 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2024_algorithmic_collusion_in_dynamic_pricing_with_deep_reinforcement_learning.md: Compare whether DQN/PPO/MARL reproduce or break tabular collusion patterns under each institution.
- Code evidence: worlds/pricing_arena
- Output evidence: 4 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

## public_goods

### benchmark: pass

- Obligation: Compare learned commons behavior to free-rider and social-optimum brackets.
- Code evidence: worlds/public_goods/benchmarks.py
- Output evidence: outputs/public_goods_full/summary_aggregate.csv
- Missing/review: none

### metrics: pass

- Obligation: Separate state-changing institutions from reward/accounting-only effects.
- Code evidence: validate_public_goods_effects.py; worlds/public_goods/training.py
- Output evidence: outputs/public_goods_full/summary_aggregate.csv; outputs/public_goods_full/institution_effect_validation.json
- Missing/review: none

### paper_benchmark: partial

- Obligation: 1954_the_pure_theory_of_public_expenditure.md: Private free-riding versus social-optimum contribution/provision brackets.
- Code evidence: worlds/public_goods
- Output evidence: 36 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 1954_the_pure_theory_of_public_expenditure.md: Contribution rate, extraction rate, welfare, sustainability, collapse indicators.
- Code evidence: worlds/public_goods
- Output evidence: 36 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 1954_the_pure_theory_of_public_expenditure.md: Report free-rider and social-optimum brackets and show whether learned agents underprovide contributions.
- Code evidence: worlds/public_goods
- Output evidence: 36 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 1965_the_logic_of_collective_action.md: Group benefit does not guarantee individual contribution; free-rider incentives dominate absent institutions.
- Code evidence: worlds/public_goods
- Output evidence: 36 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 1965_the_logic_of_collective_action.md: Contribution, extraction, sustainability, welfare, group outcome versus individual incentive.
- Code evidence: worlds/public_goods
- Output evidence: 36 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 1965_the_logic_of_collective_action.md: Explain why low contribution is expected under private incentives and do not treat it as a bug without checking incentives.
- Code evidence: worlds/public_goods
- Output evidence: 36 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2000_conditional_cooperation_and_voluntary_contributions_to_public_goods.md: Contribution can depend on beliefs/observations about others' contributions.
- Code evidence: worlds/public_goods
- Output evidence: 36 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2000_conditional_cooperation_and_voluntary_contributions_to_public_goods.md: Contribution response, sustainability, welfare, reputation bonuses, information-condition deltas.
- Code evidence: worlds/public_goods
- Output evidence: 36 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2000_conditional_cooperation_and_voluntary_contributions_to_public_goods.md: Information/reputation variants should be judged by contribution response and sustainability, not only reward bonuses.
- Code evidence: worlds/public_goods
- Output evidence: 36 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2000_cooperation_and_punishment_in_public_goods_experiments.md: Punishment can sustain cooperation but may impose welfare costs.
- Code evidence: worlds/public_goods
- Output evidence: 36 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2000_cooperation_and_punishment_in_public_goods_experiments.md: Contribution, sustainability, welfare, penalty incidence, state-change classification.
- Code evidence: worlds/public_goods
- Output evidence: 36 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2000_cooperation_and_punishment_in_public_goods_experiments.md: Penalty institutions must show whether contributions/sustainability change, not just whether rewards were reduced.
- Code evidence: worlds/public_goods
- Output evidence: 36 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2007_the_effect_of_rewards_and_sanctions_in_provision_of_public_goods.md: Rewards and sanctions can change provision but may be costly or crowd out intrinsic incentives.
- Code evidence: worlds/public_goods
- Output evidence: 36 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2007_the_effect_of_rewards_and_sanctions_in_provision_of_public_goods.md: Welfare, contribution, sustainability, reward payments, penalty costs, state-change classification.
- Code evidence: worlds/public_goods
- Output evidence: 36 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2007_the_effect_of_rewards_and_sanctions_in_provision_of_public_goods.md: Separate welfare-improving cooperation from costly reward accounting under contribution matching and sanctions.
- Code evidence: worlds/public_goods
- Output evidence: 36 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2011_cooperation_and_contagion_in_web_based_networked_public_goods_experiments.md: Network structure and observability affect cooperation contagion and contribution dynamics.
- Code evidence: worlds/public_goods
- Output evidence: 36 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2011_cooperation_and_contagion_in_web_based_networked_public_goods_experiments.md: Contribution, sustainability, welfare, information-condition deltas, variance across seeds.
- Code evidence: worlds/public_goods
- Output evidence: 36 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2011_cooperation_and_contagion_in_web_based_networked_public_goods_experiments.md: Treat information restrictions as strategic observability changes and report whether contribution/sustainability differs from baseline.
- Code evidence: worlds/public_goods
- Output evidence: 36 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

## resource_island

### activation: pass

- Obligation: Report trade/property activation before interpreting Resource Island institutions.
- Code evidence: worlds/resource_island/env.py; worlds/resource_island/training.py
- Output evidence: outputs/resource_island_v1_full/summary_aggregate.csv
- Missing/review: none

### benchmark: pass

- Obligation: Provide oracle/greedy gather benchmarks for scale and sanity.
- Code evidence: worlds/resource_island/benchmarks.py
- Output evidence: outputs/resource_island_v1_full/summary_aggregate.csv
- Missing/review: none

### paper_benchmark: partial

- Obligation: 1968_the_tragedy_of_the_commons.md: Open-access common resources face depletion pressure when private extraction benefits exceed shared costs.
- Code evidence: worlds/resource_island
- Output evidence: 77 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 1968_the_tragedy_of_the_commons.md: Resource sustainability, survival, welfare, extraction/gathering, trade, inequality, specialization.
- Code evidence: worlds/resource_island
- Output evidence: 77 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 1968_the_tragedy_of_the_commons.md: Resource Island must report depletion/sustainability and activation diagnostics, not only agent reward.
- Code evidence: worlds/resource_island
- Output evidence: 77 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 1990_governing_the_commons.md: Governance works through boundaries, monitoring, graduated sanctions, and local rule fit.
- Code evidence: worlds/resource_island
- Output evidence: 77 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 1990_governing_the_commons.md: Property opportunity counts, property violations, trade counts, welfare, sustainability, survival, inequality.
- Code evidence: worlds/resource_island
- Output evidence: 77 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 1990_governing_the_commons.md: Property/reputation mechanisms must have activation diagnostics before welfare differences are interpreted.
- Code evidence: worlds/resource_island
- Output evidence: 77 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2010_a_review_of_design_principles_for_community_based_natural_resource_management.md: Operational design principles: boundaries, monitoring, sanctioning, conflict resolution, and local fit.
- Code evidence: worlds/resource_island
- Output evidence: 77 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2010_a_review_of_design_principles_for_community_based_natural_resource_management.md: Activation thresholds, property opportunities/violations, trade blocks, sustainability, welfare, survival.
- Code evidence: worlds/resource_island
- Output evidence: 77 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2010_a_review_of_design_principles_for_community_based_natural_resource_management.md: Check monitoring, boundaries, sanctions, and local incentive conditions through explicit v1 diagnostics.
- Code evidence: worlds/resource_island
- Output evidence: 77 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2017_multi_agent_reinforcement_learning_in_sequential_social_dilemmas.md: Cooperation and defection emerge over time through spatial/temporal coordination, not one-shot payoffs only.
- Code evidence: worlds/resource_island
- Output evidence: 77 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2017_multi_agent_reinforcement_learning_in_sequential_social_dilemmas.md: Cooperation/trade/contribution, sustainability, welfare, survival, coordination failures, robustness.
- Code evidence: worlds/resource_island
- Output evidence: 77 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2017_multi_agent_reinforcement_learning_in_sequential_social_dilemmas.md: Frame failed trade/institution activation as coordination/exploration evidence and report attempts versus successes.
- Code evidence: worlds/resource_island
- Output evidence: 77 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2018_inequity_aversion_improves_cooperation_in_intertemporal_social_dilemmas.md: Social-preference shaping can improve cooperation in intertemporal dilemmas.
- Code evidence: worlds/resource_island
- Output evidence: 77 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2018_inequity_aversion_improves_cooperation_in_intertemporal_social_dilemmas.md: Cooperation/trade/contribution, welfare, inequality, sustainability, reward-accounting versus state changes.
- Code evidence: worlds/resource_island
- Output evidence: 77 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2018_inequity_aversion_improves_cooperation_in_intertemporal_social_dilemmas.md: When institutions change rewards, check whether they alter state behavior or only accounting.
- Code evidence: worlds/resource_island
- Output evidence: 77 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2021_scalable_evaluation_of_multi_agent_reinforcement_learning_with_melting_pot.md: Generalization and robustness require held-out tasks/populations, not only in-distribution performance.
- Code evidence: worlds/resource_island
- Output evidence: 77 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2021_scalable_evaluation_of_multi_agent_reinforcement_learning_with_melting_pot.md: Generalization, robustness, welfare, cooperation, exploitation, held-out scenario performance.
- Code evidence: worlds/resource_island
- Output evidence: 77 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2021_scalable_evaluation_of_multi_agent_reinforcement_learning_with_melting_pot.md: Do not claim general social robustness from in-distribution training alone; use cross-world synthesis and future held-out variants.
- Code evidence: worlds/resource_island
- Output evidence: 77 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results
