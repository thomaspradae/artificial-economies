# Theory Obligation Audit

This is a deterministic coverage check. `pass` means required files/columns/terms were observed. `partial` means there is implementation evidence but the obligation still needs human review. `missing` means the evidence was not found.

Summary: pass=7, partial=31, missing=0.

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

- Obligation: 1990_why_are_vickrey_auctions_rare.md: Vickrey Auctions
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 1994_auctions_vs_negotiations.md: auction
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 1994_auctions_vs_negotiations.md: expected revenue
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 1994_auctions_vs_negotiations.md: results for different auction mechanisms
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 1994_the_interdisciplinary_study_of_coordination.md: Myerson 1981
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 1996_analyzing_the_airwaves_auction.md: classical benchmark
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 1997_auctioning_conservation_contracts_a_theoretical_analysis_and_an_application.md: Myerson optimal auction reserve price revenue
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2003_bid_rotation_and_collusion_in_repeated_auctions.md: equilibrium
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2003_bid_rotation_and_collusion_in_repeated_auctions.md: payoff
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2003_bid_rotation_and_collusion_in_repeated_auctions.md: the classical benchmark, prior learning setup, metrics, and what this project must reproduce
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2010_why_customers_value_self_designed_products_the_importance_of_process_effort_and_enjoyment_.md: Vickrey 1961
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2017_market_mechanisms_and_funding_dynamics_in_equity_crowdfunding.md: Vickrey 1961
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2018_blockchain_disruption_and_smart_contracts.md: Myerson optimal auction reserve price revenue
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2019_optimal_auctions_through_deep_learning.md: dominant-strategy incentive compatibility (DSIC)
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2019_optimal_auctions_through_deep_learning.md: expected revenue and regret
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2019_optimal_auctions_through_deep_learning.md: none
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2021_a_permutation_equivariant_neural_network_architecture_for_auction_design.md: symmetric auctions
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2021_a_permutation_equivariant_neural_network_architecture_for_auction_design.md: regret minimization
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2021_a_permutation_equivariant_neural_network_architecture_for_auction_design.md: deep learning architectures for auction design
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2021_decentralized_edge_intelligence_a_dynamic_resource_allocation_framework_for_hierarchical_f.md: Myerson optimal auction reserve price revenue
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2022_optimal_er_auctions_through_attention.md: Myerson [47]
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2022_optimal_er_auctions_through_attention.md: Revenue and regret.
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2022_optimal_er_auctions_through_attention.md: The classical benchmark, prior learning setup, metrics, and what this project must reproduce.
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2023_a_scalable_neural_network_for_dsic_affine_maximizer_auction_design.md: DSIC affine maximizer auctions
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2023_a_scalable_neural_network_for_dsic_affine_maximizer_auction_design.md: revenue; scalability
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2023_a_scalable_neural_network_for_dsic_affine_maximizer_auction_design.md: the classical benchmark, prior learning setup, metrics, and what this project must reproduce
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2023_optimal_auctions_through_deep_learning_advances_in_differentiable_economics.md: Myerson (1981)
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2023_optimal_auctions_through_deep_learning_advances_in_differentiable_economics.md: Classical benchmark
- Code evidence: worlds/auction_house
- Output evidence: 28 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

## labor_market

### benchmark: partial

- Obligation: Verify deferred-acceptance stability and proposing-side strategy-proofness cases.
- Code evidence: worlds/labor_market/benchmarks.py; run_labor_market_benchmark_cases.py
- Output evidence: none
- Missing/review: missing output path: outputs/labor_market_benchmark_cases.json

### metrics: pass

- Obligation: Report match rate, stability, truthfulness, welfare, and manipulation diagnostics.
- Code evidence: worlds/labor_market/env.py; worlds/labor_market/training.py
- Output evidence: outputs/labor_market_phase3_full/mind_comparison.csv
- Missing/review: none

## pricing_arena

### benchmark: pass

- Obligation: Report Nash and joint-profit price benchmarks.
- Code evidence: worlds/pricing_arena/benchmarks.py
- Output evidence: outputs/full_v0_multiseed/summary_aggregate.csv
- Missing/review: none

### metrics: partial

- Obligation: Report price-normalized and profit-normalized collusion, exploitability, welfare, price, and profit.
- Code evidence: core/metrics.py; build_combined_table.py
- Output evidence: none
- Missing/review: missing output path: outputs/phase3_full/mind_comparison.csv; missing output column: collusion_index_mean; missing output column: profit_collusion_index_mean; missing output column: exploitability_mean; missing output column: welfare_mean; missing output column: avg_price_mean; missing output column: profit_total_mean

## public_goods

### benchmark: pass

- Obligation: Compare learned commons behavior to free-rider and social-optimum brackets.
- Code evidence: worlds/public_goods/benchmarks.py
- Output evidence: outputs/public_goods_full/summary_aggregate.csv
- Missing/review: none

### metrics: partial

- Obligation: Separate state-changing institutions from reward/accounting-only effects.
- Code evidence: validate_public_goods_effects.py; worlds/public_goods/training.py
- Output evidence: outputs/public_goods_full/summary_aggregate.csv
- Missing/review: missing output path: outputs/public_goods_full/institution_effect_validation.json

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
