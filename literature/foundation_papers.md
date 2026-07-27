# Foundation Papers

This is the simplified theory list: the papers that should ground the code and results.
It is curated from the project theory obligations, then matched against the local scout cache.

Status meanings:

- `ready_for_llm`: extracted text is already available under `literature/text/`.
- `fetch_pdf_or_paste_text`: metadata exists, but the paper text is not extracted yet.
- `manual_search_needed`: this anchor was not found in the current metadata cache.

## auction_house

| Priority | Role | Paper | Cache | Text | Manual action | Why it matters | Code/result check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| must_read | truthful_second_price | William Vickrey (1961). Counterspeculation, Auctions, and Competitive Sealed Tenders | not_found_in_cache | literature/text/manual/auction_house__1961__counterspeculation_auctions_and_competitive_sealed_tenders.txt | ready_for_llm | Original second-price auction truthfulness anchor. | worlds/auction_house/benchmarks.py; ex_post_regret and efficiency metrics. |
| must_read | optimal_auction | Roger B. Myerson (1981). Optimal Auction Design | not_found_in_cache | literature/text/manual/auction_house__1981__optimal_auction_design.txt | ready_for_llm | Classical revenue-optimal auction theory and reserve-price obligation. | second_price_reserve scenario; revenue and allocative_efficiency columns. |
| must_read | learned_auction_design | Paul Dutting; Zhe Feng; Harikrishna Narasimhan; David C. Parkes; Sai Srivatsa Ravindranath (2019). Optimal auctions through deep learning | found_exact_or_close | literature/text/2019_optimal_auctions_through_deep_learning.txt | ready_for_llm | RegretNet line: auction-learning papers care about regret/IC, revenue, efficiency, and generalization. | Auction House regret, revenue, efficiency, over/underbidding diagnostics. |
| core | ai_auction_design | Martino Banchio; Andrzej Skrzypacz (2022). Artificial Intelligence and Auction Design | not_found_in_cache | literature/text/manual/auction_house__2022__artificial_intelligence_and_auction_design.txt | ready_for_llm | Closest Q-learning auction-design paper for first-price versus second-price behavior. | Auction House first_price and second_price learned bid curves. |
| core | repeated_auctions | Thomas Nedelec; Clement Calauzenes; Noureddine El Karoui; Vianney Perchet (2020). Learning in repeated auctions | found_exact_or_close | literature/text/2020_learning_in_repeated_auctions.txt | ready_for_llm | Learning dynamics in repeated auction environments. | Auction House repeated draws and final-window bid curves. |

## cross_world_methods

| Priority | Role | Paper | Cache | Text | Manual action | Why it matters | Code/result check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| must_read | tabular_rl | Christopher J. C. H. Watkins; Peter Dayan (1992). Q-learning | not_found_in_cache | literature/text/manual/cross_world_methods__1992__q_learning.txt | ready_for_llm | Canonical tabular temporal-difference control reference. | minds/q_learning.py; cross-world Q-learning smoke and full runs. |
| must_read | deep_rl | Volodymyr Mnih et al. (2015). Human-level control through deep reinforcement learning | not_found_in_cache | literature/text/manual/cross_world_methods__2015__human_level_control_through_deep_reinforcement_learning.txt | ready_for_llm | Canonical DQN paper: replay, target network, TD loss, and benchmark discipline. | minds/deep_rl/torch_dqn_mind.py; Phase 3 and Resource Island DQN tests. |
| must_read | policy_gradient | John Schulman; Filip Wolski; Prafulla Dhariwal; Alec Radford; Oleg Klimov (2017). Proximal Policy Optimization Algorithms | not_found_in_cache | literature/text/manual/cross_world_methods__2017__proximal_policy_optimization_algorithms.txt | ready_for_llm | Canonical PPO reference for clipped policy-gradient learning. | minds/deep_rl/torch_ppo_mind.py; Phase 3 PPO tests and full outputs. |
| must_read | independent_marl | Ming Tan (1993). Multi-Agent Reinforcement Learning: Independent versus Cooperative Agents | not_found_in_cache | literature/text/manual/cross_world_methods__1993__multi_agent_reinforcement_learning_independent_versus_cooperative_agents.txt | ready_for_llm | Canonical independent-learner baseline for MARL. | minds/marl/independent_learners.py; decorrelation tests; fixed independent-DQN full outputs. |
| must_read | centralized_critic | Ryan Lowe; Yi Wu; Aviv Tamar; Jean Harb; Pieter Abbeel; Igor Mordatch (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments | not_found_in_cache | literature/text/manual/cross_world_methods__2017__multi_agent_actor_critic_for_mixed_cooperative_competitive_environments.txt | ready_for_llm | Canonical centralized-training/decentralized-execution MARL anchor. | minds/marl/centralized_critic.py; cross-world P.6 smoke/full outputs. |

## labor_market

| Priority | Role | Paper | Cache | Text | Manual action | Why it matters | Code/result check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| must_read | stable_matching | David Gale; Lloyd S. Shapley (1962). College Admissions and the Stability of Marriage | not_found_in_cache | literature/text/manual/labor_market__1962__college_admissions_and_the_stability_of_marriage.txt | ready_for_llm | Foundational deferred-acceptance/stable-matching paper. | worlds/labor_market/benchmarks.py; labor_market_full stability metrics. |
| must_read | incentives | Alvin E. Roth (1982). The Economics of Matching: Stability and Incentives | not_found_in_cache | literature/text/manual/labor_market__1982__the_economics_of_matching_stability_and_incentives.txt | ready_for_llm | Classical incentive result: stable mechanisms can be strategy-proof for one side but not both. | labor_market_benchmark_cases.json; manipulation-gain diagnostics. |
| core | market_design_application | Atila Abdulkadiroglu; Parag A. Pathak; Alvin E. Roth; Tayfun Sonmez (2006). Changing the Boston School Choice Mechanism | found_exact_or_close | literature/text/2006_changing_the_boston_school_choice_mechanism.txt | ready_for_llm | Applied market-design case for replacing manipulable mechanisms with deferred acceptance. | Labor Market fixed benchmark cases and full-run truthfulness/stability. |
| core | matching_overview | Alvin E. Roth (2007). Deferred Acceptance Algorithms: History, Theory, Practice, and Open Questions | found_exact_or_close | literature/text/2007_deferred_acceptance_algorithms_history_theory_practice_and_open_questions.txt | ready_for_llm | Compact theory/practice bridge for DA mechanisms. | Labor Market DESIGN.md and benchmark cases. |
| supporting | learning_matching | Yifei Min; Tianhao Wang; Ruitu Xu; Zhaoran Wang; Michael I. Jordan; Zhuoran Yang (2022). Learn to Match with No Regret: Reinforcement Learning in Markov Matching Markets | not_found_in_cache | literature/text/manual/labor_market__2022__learn_to_match_with_no_regret_reinforcement_learning_in_markov_matching_markets.txt | ready_for_llm | Closest RL/matching-market anchor for future cross-mind labor-market work. | Future Labor Market P.6 full ladder and synthesis. |

## pricing_arena

| Priority | Role | Paper | Cache | Text | Manual action | Why it matters | Code/result check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| must_read | classical_io | Jean Tirole (1988). The Theory of Industrial Organization | found_exact_or_close | literature/text/manual/pricing_arena__1988__the_theory_of_industrial_organization.txt | ready_for_llm | Classical industrial-organization reference for Bertrand pricing and collusion benchmarks. | worlds/pricing_arena/benchmarks.py; pricing summary benchmark columns. |
| must_read | algorithmic_collusion | Emilio Calvano; Giacomo Calzolari; Vincenzo Denicolo; Sergio Pastorello (2020). Artificial Intelligence, Algorithmic Pricing, and Collusion | found_exact_or_close | literature/text/2020_artificial_intelligence_algorithmic_pricing_and_collusion.txt | ready_for_llm | Main algorithmic-pricing-collusion benchmark for Q-learning firms. | outputs/phase3_full/mind_comparison.csv includes profit_collusion_index_mean. |
| must_read | monitoring | Emilio Calvano; Giacomo Calzolari; Vincenzo Denicolo; Sergio Pastorello (2021). Algorithmic collusion with imperfect monitoring | not_found_in_cache | literature/text/manual/pricing_arena__2021__algorithmic_collusion_with_imperfect_monitoring.txt | ready_for_llm | Connects collusion to noisy monitoring and price-war punishment dynamics. | demand_shock mechanism; exploitability and collusion comparison under shocks. |
| core | algorithm_design | John Asker; Chaim Fershtman; Ariel Pakes (2022). Artificial Intelligence, Algorithm Design, and Pricing | found_exact_or_close | literature/text/manual/pricing_arena__2022__artificial_intelligence_algorithm_design_and_pricing.txt | ready_for_llm | Shows that learning-protocol details can move pricing outcomes from competitive to monopoly-like. | Phase 3 ladder: Q-learning, DQN, PPO, independent-DQN, centralized critic. |
| core | sequential_pricing | Timo Klein (2021). Autonomous algorithmic collusion: Q-learning under sequential pricing | found_exact_or_close | literature/text/manual/pricing_arena__2021__autonomous_algorithmic_collusion_q_learning_under_sequential_pricing.txt | ready_for_llm | Shows algorithmic collusion can arise under sequential pricing protocols. | Pricing Arena design and step semantics. |
| supporting | deep_rl_pricing | Shidi Deng; Maximilian Schiffer; Martin Bichler (2024). Algorithmic Collusion in Dynamic Pricing with Deep Reinforcement Learning | found_exact_or_close | literature/text/manual/pricing_arena__2024__algorithmic_collusion_in_dynamic_pricing_with_deep_reinforcement_learning.txt | ready_for_llm | Closest deep-RL pricing anchor for the neural side of the ladder. | Phase 3 mind comparison table and price-cap diagnosis. |

## public_goods

| Priority | Role | Paper | Cache | Text | Manual action | Why it matters | Code/result check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| must_read | public_goods_theory | Paul A. Samuelson (1954). The Pure Theory of Public Expenditure | not_found_in_cache | literature/text/manual/public_goods__1954__the_pure_theory_of_public_expenditure.txt | ready_for_llm | Classical public-goods benchmark: private and social incentives diverge. | worlds/public_goods/benchmarks.py; public_goods_full summaries. |
| must_read | collective_action | Mancur Olson (1965). The Logic of Collective Action | not_found_in_cache | literature/text/manual/public_goods__1965__the_logic_of_collective_action.txt | ready_for_llm | Foundational free-rider/collective-action framing. | baseline extraction/contribution/sustainability metrics. |
| must_read | punishment | Ernst Fehr; Simon Gachter (2000). Cooperation and Punishment in Public Goods Experiments | not_found_in_cache | literature/text/manual/public_goods__2000__cooperation_and_punishment_in_public_goods_experiments.txt | ready_for_llm | Canonical evidence that sanctions can sustain contributions. | validate_public_goods_effects.py; penalty_schedule state-change classification. |
| core | conditional_cooperation | Claudia Keser; Frans van Winden (2000). Conditional Cooperation and Voluntary Contributions to Public Goods | found_exact_or_close | literature/text/manual/public_goods__2000__conditional_cooperation_and_voluntary_contributions_to_public_goods.txt | ready_for_llm | Grounds contribution behavior in conditional cooperation rather than pure one-shot selfishness. | Public Goods reputation and information_restriction outputs. |
| core | rewards_sanctions | Martin Sefton; Robert Shupp; James M. Walker (2007). The Effect of Rewards and Sanctions in Provision of Public Goods | found_exact_or_close | literature/text/2007_the_effect_of_rewards_and_sanctions_in_provision_of_public_goods.txt | ready_for_llm | Directly maps to reward/sanction institution variants. | Public Goods institution-effect validator. |
| supporting | networked_public_goods | Siddharth Suri; Duncan J. Watts (2011). Cooperation and Contagion in Web-Based, Networked Public Goods Experiments | found_exact_or_close | literature/text/manual/public_goods__2011__cooperation_and_contagion_in_web_based_networked_public_goods_experiments.txt | ready_for_llm | Supports the idea that interaction/network information changes cooperation. | information_restriction institution and contribution/sustainability deltas. |

## resource_island

| Priority | Role | Paper | Cache | Text | Manual action | Why it matters | Code/result check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| must_read | commons | Garrett Hardin (1968). The Tragedy of the Commons | not_found_in_cache | literature/text/manual/resource_island__1968__the_tragedy_of_the_commons.txt | ready_for_llm | Basic common-resource overuse benchmark. | resource_sustainability, survival, and gather/trade diagnostics. |
| must_read | common_pool_governance | Elinor Ostrom (1990). Governing the Commons | not_found_in_cache | literature/text/manual/resource_island__1990__governing_the_commons.txt | ready_for_llm | Canonical common-pool institution design reference. | Resource Island v1 property opportunities, violations, and trade diagnostics. |
| core | design_principles | Michael Cox; Gwen Arnold; Sergio Villamayor-Tomas (2010). A Review of Design Principles for Community-based Natural Resource Management | found_exact_or_close | literature/text/2010_a_review_of_design_principles_for_community_based_natural_resource_management.txt | ready_for_llm | Operationalizes Ostrom-style design principles for natural-resource institutions. | Resource Island v1 contested layouts and activation thresholds. |
| core | sequential_social_dilemmas | Joel Z. Leibo; Vinicius Flores Zambaldi; Marc Lanctot; Janusz Marecki; Thore Graepel (2017). Multi-agent Reinforcement Learning in Sequential Social Dilemmas | not_found_in_cache | literature/text/manual/resource_island__2017__multi_agent_reinforcement_learning_in_sequential_social_dilemmas.txt | ready_for_llm | Closest MARL framing for spatial, temporally extended commons/public-goods dilemmas. | Resource Island P.6 full trade-attempt and success diagnostics. |
| supporting | social_preferences | Edward Hughes; Joel Z. Leibo; Matthew Phillips; Karl Tuyls; Edgar Duenas-Guzman; others (2018). Inequity aversion improves cooperation in intertemporal social dilemmas | not_found_in_cache | literature/text/manual/resource_island__2018__inequity_aversion_improves_cooperation_in_intertemporal_social_dilemmas.txt | ready_for_llm | Shows how reward/social-preference modifications can change cooperation in sequential dilemmas. | Resource Island and Public Goods welfare versus state-metric comparisons. |
| supporting | marl_benchmark | Joel Z. Leibo et al. (2021). Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot | not_found_in_cache | literature/text/manual/resource_island__2021__scalable_evaluation_of_multi_agent_reinforcement_learning_with_melting_pot.txt | ready_for_llm | Benchmark suite for social generalization, resource sharing, and social dilemmas. | Cross-world synthesis and future held-out institution/world variants. |

## Manual PDF Queue

Bring these PDFs or paste their full text first. They are the highest-value gaps.

| World | Priority | Paper | Action | URL | PDF URL | Scholar |
| --- | --- | --- | --- | --- | --- | --- |
