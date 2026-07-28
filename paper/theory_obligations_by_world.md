# Theory Obligations By World

This is the writing bridge between the literature cards and the experiments. It is not a related-work dump. Each world must answer the same six questions before its results can become thesis prose:

1. What is the classical benchmark?
2. What is the known RL/MARL result?
3. What metric obligation follows from that literature?
4. What does the repo reproduce or validate?
5. What does the repo add?
6. What can the repo still not claim?

Canonical inputs:

- Foundation cards: `literature/foundation_paper_cards/`
- Theory gap report: `literature/theory_gap_report.csv`
- Obligation audit: `literature/obligation_audit.md`
- Card-quality audit: `literature/foundation_card_quality_audit.md`
- Novelty table: `literature/novelty_gap_table.csv`

## Pricing Arena

**Classical benchmark.** One-shot Bertrand-style pricing predicts competitive or Nash prices, while repeated interaction can support supracompetitive outcomes when dynamic incentives make punishment credible.

**Known RL/MARL result.** Q-learning and deep-RL pricing agents can learn supracompetitive pricing without explicit communication. Prior results are sensitive to learner class, monitoring, algorithm design, and metric choice.

**Metric obligation.** Do not report only average price. Pricing results must include static Nash and joint-profit references, profit-normalized collusion, price-normalized collusion as a proxy, welfare/profit, price dispersion, and exploitability.

**What our code reproduces.** The repo implements the repeated pricing game, tabular Q-learning, static Nash and joint-profit benchmarks, profit-normalized Calvano-style collusion, price-proxy collusion, multiseed evaluation, and frozen-policy exploitability.

**What our result adds.** The cross-mind table shows that price caps reduce exploitability across tested minds, but collusion conclusions are metric- and capability-sensitive. DQN-family agents can sit near a price ceiling with low dispersion while preserving elevated profit through the quantity/profit channel.

**What we still cannot claim.** The result is not a general theorem about price caps. It is finite-run evidence in a stylized duopoly with discrete prices and configured demand. It does not establish real-market antitrust outcomes, legal liability, or universal convergence to collusion.

## Resource Island

**Classical benchmark.** Common-pool-resource theory says governance requires boundaries, monitoring, enforceable sanctions or exclusion, and local fit. Trade also requires observable gains from exchange and enough contact or market access.

**Known RL/MARL result.** Sequential social-dilemma and commons MARL work often finds cooperation failures unless observation, incentives, punishment, reputation, or social preferences support cooperation.

**Metric obligation.** Resource Island cannot be judged only by welfare. It must report survival, sustainability, inequality, specialization, trade attempts, successful trades, trade blocks, property opportunities, property violations, and institution activation.

**What our code reproduces.** The repo implements a bounded spatial gather/trade economy with inventory, energy, property rights, redistribution, trade price controls, reputation, oracle/greedy gather benchmarks, Q-learning, and the neural/MARL mind ladder.

**What our result adds.** Resource Island is evaluated as a pressure-tested scarce-resource world with contested resources, unequal trades, specialization pressure, and activation diagnostics. This lets institution claims be conditioned on observed trade attempts, property opportunities, and institution-block counters rather than inferred from welfare alone.

**What we still cannot claim.** Resource Island is not a realistic model of property law, bargaining, or community governance. Whole-island trade weakens the spatial-trade interpretation. Strict-local trade-radius ablations remain needed before claiming spatial-friction effects.

## Auction House

**Classical benchmark.** Under independent private values, second-price auctions make truthful bidding weakly dominant and support efficient allocation under standard assumptions. First-price auctions induce bid shading. Reserve prices can increase seller revenue while reducing allocative efficiency and welfare.

**Known RL/MARL result.** Learning-in-auctions and learned-mechanism-design papers evaluate revenue, allocative efficiency, bidder surplus, regret or incentive-compatibility proxies, bid shading, and generalization. They do not treat bidder reward alone as enough.

**Metric obligation.** Auction House must report revenue, bidder surplus, total welfare, allocative efficiency, ex-post regret, overbidding, underbidding, shading distance, no-sale behavior, and learned bid curves.

**What our code reproduces.** The repo implements first-price, second-price, reserve-price, and simple clock-auction mechanics; deterministic tie-breaking; truthful second-price benchmarks; first-price shading references; reserve/no-sale checks; ex-post regret; and learned bid curves.

**What our result adds.** Auction House puts a theory-anchored mechanism-design environment into the same world/mind/institution platform as Pricing Arena and Resource Island. The validated Q-learning runs show interpretable first-price underbidding/shading and reserve-price revenue/efficiency tradeoffs.

**What we still cannot claim.** Current auction results do not prove learned bidders converge to equilibrium. Second-price learning remains below the truthful efficiency benchmark in the recorded full run. Neural/MARL cross-mind auction results need full-run interpretation before capability claims.

## Public Goods

**Classical benchmark.** Private incentives underprovide contributions and over-extract shared resources relative to the social optimum unless institutions change incentives, information, reputation, or punishment.

**Known RL/MARL result.** MARL public-goods and commons studies find cooperation is sensitive to reward shaping, punishment, matching, reputation, observability, and repeated interaction.

**Metric obligation.** Public Goods must separate reward-accounting changes from state changes. It must report contribution, extraction, sustainability, welfare, inequality, tax revenue, collapse diagnostics, and free-rider/social-optimum brackets.

**What our code reproduces.** The repo implements contribution/extraction decisions, public-pool dynamics, extraction rationing, deterministic regeneration, penalty schedules, contribution matching, reputation, information restriction, tax schedule, and benchmark brackets.

**What our result adds.** The full Q-learning run validates the commons pressure: baseline agents mostly extract with low contribution and low sustainability. Contribution matching improves welfare/sustainability; reputation strongly changes rewards; the tax schedule mostly redistributes/accountingly preserves welfare at tested rates.

**What we still cannot claim.** Not every measured improvement is a real sustainability improvement. Reward bonuses can raise welfare metrics without materially changing the common-pool state. The effect validator must be cited alongside any institution ranking.

## Labor Market

**Classical benchmark.** Worker-proposing deferred acceptance produces stable matchings and is strategy-proof for the proposing side under standard preferences and information assumptions.

**Known RL/MARL result.** Learning in matching markets is less standardized than pricing or auctions. The central obligation is to preserve mechanism-theory predictions before claiming learned manipulation or instability.

**Metric obligation.** Labor Market must report match rate, blocking-pair stability, total welfare, truthful-report rate, manipulation-gain diagnostics, and benchmark cases for truthful DA and forced unstable matchings.

**What our code reproduces.** The repo implements worker-side learning, fixed employer preferences, report-top actions, deferred-acceptance matching, blocking-pair checks, truthful matching benchmarks, welfare accounting, and strategy-proofness benchmark cases.

**What our result adds.** The full Q-learning run produces full matching and mostly stable outcomes while still leaving room for learned reporting variation. This gives the platform an asymmetric-agent world where not every economic actor is a learner.

**What we still cannot claim.** Worker-side misreport profitability should not be expected under worker-proposing DA, so manipulation claims must target the correct side or a different mechanism. Employer learning, interviews, priorities, information changes, and non-DA matching rules remain future extensions.

## Cross-World Writing Rule

Every results subsection should end with this sentence shape:

> This result is reportable because the world reproduces `[benchmark]`, measures `[metrics]`, and validates `[activation/evidence]`; the novel claim is `[addition]`, while `[boundary]` remains outside the claim.

If a world cannot fill that sentence, its output is diagnostic engineering evidence, not a thesis result yet.
