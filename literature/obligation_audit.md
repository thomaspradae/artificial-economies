# Theory Obligation Audit

This is a deterministic coverage check. `pass` means required files/columns/terms were observed. `partial` means there is implementation evidence but the obligation still needs human review. `missing` means the evidence was not found.

Summary: pass=10, partial=5, missing=0.

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

- Obligation: 2022_optimal_er_auctions_through_attention.md: Myerson [47]
- Code evidence: worlds/auction_house
- Output evidence: 61 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_metrics: partial

- Obligation: 2022_optimal_er_auctions_through_attention.md: Revenue and regret.
- Code evidence: worlds/auction_house
- Output evidence: 61 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2022_optimal_er_auctions_through_attention.md: The classical benchmark, prior learning setup, metrics, and what this project must reproduce.
- Code evidence: worlds/auction_house
- Output evidence: 61 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_benchmark: partial

- Obligation: 2023_optimal_auctions_through_deep_learning_advances_in_differentiable_economics.md: Myerson (1981)
- Code evidence: worlds/auction_house
- Output evidence: 61 matching output paths
- Missing/review: human review required: compare filled card obligation to exact code/results

### paper_reproduce: partial

- Obligation: 2023_optimal_auctions_through_deep_learning_advances_in_differentiable_economics.md: Classical benchmark
- Code evidence: worlds/auction_house
- Output evidence: 61 matching output paths
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
