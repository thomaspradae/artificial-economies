# Theory Obligations

Generated from `literature/queries.yaml` and `literature/papers_raw.jsonl`.

## pricing_arena

- Closest paper: Artificial Intelligence, Algorithmic Pricing, and Collusion
- Classical prediction: One-shot Bertrand competition predicts competitive/Nash pricing; repeated interaction can support supracompetitive prices under suitable dynamic incentives.
- Known RL/MARL result: Q-learning and deep RL pricing agents can learn supracompetitive prices without explicit communication, but outcomes depend on algorithm, monitoring, and metric choice.
- Theory benchmark: static Nash prices, joint-profit price, profit-normalized Calvano-style collusion
- Prior metric: collusion index, price/profit, welfare/profit, sometimes convergence
- Our metric: price collusion, profit collusion, exploitability, welfare, price dispersion, quantity/profit
- Prior-work failure mode to check: price-based collusion can understate profit extraction when regulation changes quantity/profit channels.
- Code obligation: Reproduce Nash/joint-profit reference lines and report profit-normalized collusion next to the historical price proxy.
- Gap: institution robustness across learner capability and exploitability, not only collusion level

## resource_island

- Closest paper: A Review of Design Principles for Community-based Natural Resource Management
- Classical prediction: Commons institutions require monitoring, credible exclusion/sanctions, and repeated interaction; trade requires observable gains from exchange and enough contact or market access.
- Known RL/MARL result: Sequential social dilemmas and common-pool MARL often show cooperation failures or institution-sensitive cooperation depending on observability and incentives.
- Theory benchmark: oracle/greedy gather bounds plus institution activation thresholds
- Prior metric: cooperation, sustainability, inequality, returns, sanction/compliance rates
- Our metric: survival, welfare, trade, property pressure, sustainability, specialization, inequality
- Prior-work failure mode to check: institutions can appear ineffective simply because their trigger conditions never occur.
- Code obligation: Report trade attempts, successful trades, property opportunities, violations, and institution blocks before interpreting welfare differences.
- Gap: activation-validated institutions in a small reproducible spatial economy

## auction_house

- Closest paper: Optimal Auctions through Deep Learning: Advances in Differentiable Economics
- Classical prediction: Second-price auctions make truthful bidding weakly dominant; first-price auctions induce bid shading; reserves can raise revenue while reducing allocative efficiency.
- Known RL/MARL result: Learning auction papers evaluate revenue, regret/incentive compatibility, efficiency, and generalization rather than bidder reward alone.
- Theory benchmark: truthful Vickrey, shaded first-price, reserve/no-sale benchmark, ex-post regret
- Prior metric: revenue, regret, incentive compatibility, efficiency, bidder utility
- Our metric: revenue, efficiency, surplus, welfare, regret, over/underbidding, shading distance
- Prior-work failure mode to check: high bidder payoff or seller revenue can hide incentive-compatibility or allocative-efficiency failures.
- Code obligation: Compare learned bid curves to truthful and shaded benchmarks and report regret/exploitability-style misreport incentives.
- Gap: known auction theory recovered inside the same cross-world capability ladder

## public_goods

- Closest paper: THE EFFECT OF REWARDS AND SANCTIONS IN PROVISION OF PUBLIC GOODS
- Classical prediction: Private incentives underprovide contributions and overuse shared resources relative to the social optimum unless institutions alter incentives or information.
- Known RL/MARL result: MARL public-goods and commons studies often find cooperation sensitive to reward shaping, punishment, reputation, and observability.
- Theory benchmark: free-rider/social-optimum policy brackets and collapse diagnostics
- Prior metric: contribution, cooperation, punishment, welfare, sustainability
- Our metric: contribution, extraction, sustainability, welfare, inequality, tax revenue, collapse
- Prior-work failure mode to check: reward/accounting institutions can change measured welfare without changing the underlying pool state.
- Code obligation: Separate state-changing effects from reward/accounting effects and compare learned behavior to free-rider/social-optimum brackets.
- Gap: separating reward-accounting institutions from state-changing institutions

## labor_market

- Closest paper: Deferred Acceptance Algorithms: History, Theory, Practice, and Open Questions
- Classical prediction: Worker-proposing deferred acceptance produces stable matchings and is strategy-proof for the proposing side under standard assumptions.
- Known RL/MARL result: Learning in matching markets is less standardized; the key obligation is to preserve mechanism-theory predictions before claiming learned manipulation.
- Theory benchmark: truthful deferred acceptance, blocking-pair checks, no profitable proposing-side report deviation
- Prior metric: stability, match rate, welfare, regret/manipulation incentives
- Our metric: match rate, stability, truthfulness, welfare, manipulation gain, blocking pairs
- Prior-work failure mode to check: apparent manipulation by proposing-side agents may be a benchmark or mechanism-specification bug rather than an economic finding.
- Code obligation: Verify stable/truthful benchmark cases and target manipulation tests at a side/mechanism where profitable deviations are theoretically possible.
- Gap: learned reporting behavior in an asymmetric-agent world using the shared mind ladder
