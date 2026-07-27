# Suggested Citation / Artifact Patch Queue

This is a review queue, not an automatic rewrite. Apply only the suggested fixes that are substantively correct after checking the relevant paper card or output.

## C0001 - Abstract - needs_artifact

This paper introduces a code-first experimental platform for studying whether economic mechanisms remain robust when the strategic agents inside them are adaptive learners rather than equilibrium solvers.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0002 - Abstract - needs_artifact

The platform implements a shared World/Agent/Institution interface, repeated multiseed evaluation, exploitability tests, and interchangeable mind classes ranging from random and tabular Q-learning agents to PyTorch DQN, PPO, decorrelated independent-DQN, and centralized-critic agents.

- Suggested citations: `sutton2018, mnih2015, schulman2017, tan1993`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0004 - Abstract - needs_artifact

In this world, price caps robustly reduce exploitability across mind classes, but their effect on collusion is metric- and architecture-dependent.

- Suggested citations: `calvano2020, calvano2021`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0005 - Abstract - needs_artifact

A new paired-seed audit shows that DQN's positive profit-under-cap result is not visible in 1000-step traces; it emerges with training budget, becomes mostly positive by 10,000 steps, and is positive in the 40,000-step full table.

- Suggested citations: `mnih2015`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0008 - Abstract - needs_citation

Three additional environments extend the suite: Auction House provides a benchmarked auction-design testbed, Public Goods exposes common-pool extraction and contribution incentives, and Labor Market tests learned reporting under deferred acceptance.

- Suggested citations: `samuelson1954, olson1965, hardin1968, ostrom1990, galeshapley1962, rothsotomayor1990`
- Artifact support: `outputs/auction_house_full/summary_aggregate.csv, outputs/auction_house_phase3_full/mind_comparison.csv, outputs/public_goods_full/summary_aggregate.csv, outputs/public_goods_phase3_full/mind_comparison.csv, outputs/labor_market_full/summary_aggregate.csv, outputs/labor_market_phase3_full/mind_comparison.csv, outputs/labor_market_benchmark_cases.json`
- Fix: Consider adding \citep{samuelson1954}, \citep{olson1965}, \citep{hardin1968}, \citep{ostrom1990}.

## C0009 - Abstract - needs_artifact

The current contribution is both empirical and methodological: it reports initial evidence that institutional robustness changes across learning architectures, and it documents the validation discipline needed to distinguish real economic findings from inactive-mechanism or short-horizon artifacts.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0010 - Introduction - needs_artifact

Economic mechanisms are usually evaluated against a model of strategic behavior: equilibrium prices, truthful bidding, efficient allocation, or some other theoretical benchmark.

- Suggested citations: `vickrey1961`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0014 - Introduction - needs_artifact

The goal is to create environments that are simple enough to audit but rich enough to expose failures of institutional robustness.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0015 - Introduction - needs_artifact

A mechanism is treated as robust only if it performs well across seeds, across learner classes, and under adversarial or exploitability-style checks.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0016 - Introduction - needs_artifact

The central research question is: When the agents inside an institution move from random behavior and tabular Q-learning to function approximation, policy gradients, independent learners, or centralized critics, do the same economic institutions still reduce collusion, exploitability, inequality, and inefficient strategic behavior?

- Suggested citations: `calvano2020, calvano2021, sutton2018, schulman2017, tan1993, lowe2017`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0021 - Introduction - needs_artifact

Price caps reduce exploitability for every tested mind class, but their collusion effect depends on the operational metric and on learner architecture.

- Suggested citations: `calvano2020, calvano2021`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0023 - Introduction - needs_artifact

Under a price cap, average price falls to 4.73 and exploitability falls to 40.32.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0024 - Introduction - needs_citation

For DQN-family learners, however, the cap can create a different pattern.

- Suggested citations: `mnih2015`
- Artifact support: `none`
- Fix: Consider adding \citep{mnih2015}.

## C0025 - Introduction - needs_artifact

A short-run mechanism trace shows profit falling under the cap, but a paired-seed budget audit shows the sign reversing as training continues.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0026 - Introduction - needs_artifact

The DQN price-cap profit delta is negative after 1000 steps, mixed after 5000 steps, mostly positive after 10,000 steps, and positive in the existing 40,000-step full table.

- Suggested citations: `mnih2015`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0027 - Introduction - needs_artifact

That is the kind of result this platform is meant to surface: a regulation can look strong under one behavioral lens, weak under another, and different again when the learning horizon changes.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0029 - Related Work - needs_artifact

The exploitability protocol follows from the same concern: a learned institution should not only be evaluated by its on-policy average outcome, but also by its brittleness against strategic deviation.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0036 - Theory Obligations - manual_verify

For each environment, the project records six claims before interpreting results: the classical benchmark, the known RL/MARL result, the metric obligation, what the repo reproduces, what the repo adds, and what the repo still cannot claim.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Manually verify this claim against foundation cards or add a precise citation/artifact.

## C0037 - Theory Obligations - needs_artifact

This prevents a runnable world from being mistaken for an economically meaningful world.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0044 - Theory Obligations - manual_verify

It is used to keep each environment tied to a benchmark concept, a metric obligation, and a claim boundary before any learned-agent table is interpreted.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Manually verify this claim against foundation cards or add a precise citation/artifact.

## C0055 - Auction House. - manual_verify

Auction House has the clearest benchmark obligations.

- Suggested citations: `none`
- Artifact support: `outputs/auction_house_full/summary_aggregate.csv, outputs/auction_house_phase3_full/mind_comparison.csv`
- Fix: Manually verify this claim against foundation cards or add a precise citation/artifact.

## C0059 - Auction House. - manual_verify

The current claim is therefore benchmark deviation under different learners, not convergence to auction equilibrium.

- Suggested citations: `none`
- Artifact support: `outputs/auction_house_full/summary_aggregate.csv, outputs/auction_house_phase3_full/mind_comparison.csv`
- Fix: Manually verify this claim against foundation cards or add a precise citation/artifact.

## C0060 - Public Goods. - needs_citation

The Public Goods world is built around a metric trap.

- Suggested citations: `samuelson1954, olson1965`
- Artifact support: `outputs/public_goods_full/summary_aggregate.csv, outputs/public_goods_phase3_full/mind_comparison.csv`
- Fix: Consider adding \citep{samuelson1954}, \citep{olson1965}.

## C0077 - Validation Discipline - needs_artifact

Second, where theory provides a known answer, the implementation must recover or bracket that answer before learned-policy output is interpreted.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0079 - Validation Discipline - needs_artifact

Third, the experiment must be replicated over seeds, with per-seed and aggregate CSV outputs.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0085 - Validation Discipline - needs_artifact

The second new validation layer is mechanism tracing.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0089 - Validation Discipline - needs_artifact

That disagreement triggered the dedicated paired-seed price-cap audit described below.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0091 - Learner Axis - needs_citation

PPO is not ``more capable'' than DQN in a general sense, and independent-DQN is not simply one step above either one.

- Suggested citations: `mnih2015, schulman2017, tan1993`
- Artifact support: `none`
- Fix: Consider adding \citep{mnih2015}, \citep{schulman2017}, \citep{tan1993}.

## C0092 - Learner Axis - manual_verify

The cleaner independent variable is an assumption-violation axis: each learner changes a specific assumption behind the benchmark story.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Manually verify this claim against foundation cards or add a precise citation/artifact.

## C0093 - Learner Axis - needs_artifact

Tabular Q-learning is the discrete value-learning control; DQN adds function approximation and replay; PPO replaces off-policy value learning with on-policy stochastic policy optimization; independent-DQN makes each agent learn separately in a nonstationary multi-agent environment; and the centralized-critic scaffold adds centralized training information while retaining decentralized actions.

- Suggested citations: `sutton2018, mnih2015, schulman2017, tan1993`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0102 - Learner Axis - needs_artifact

The corrected implementation now uses separate per-agent learners, decorrelated child random streams, separate replay and exploration randomness, and scoped PyTorch initialization.

- Suggested citations: `mnih2015`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0107 - World - needs_artifact

Firms earn profit from demand net of costs, and learning agents update from repeated play.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0108 - World - needs_artifact

The implemented mechanisms are no regulation, price cap, high-price tax, random audits, anti-collusion penalty, and demand shocks.

- Suggested citations: `calvano2020, calvano2021, zheng2020, zheng2021`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0109 - Metrics - needs_artifact

The main metrics are average price, total profit, consumer surplus, welfare, exploitability, and two collusion indexes.

- Suggested citations: `calvano2020, calvano2021`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0111 - Metrics - needs_artifact

The literature-comparable metric is a profit-normalized index: CI_profit = _observed - _Nash _monopoly - _Nash.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0113 - Main Result - needs_citation

Table reftab:pricing-cap shows a compact version of the mature Pricing Arena result for Q-learning and random baselines.

- Suggested citations: `sutton2018`
- Artifact support: `outputs/phase3_full/mind_comparison.csv`
- Fix: Consider adding \citep{sutton2018}.

## C0114 - Main Result - needs_artifact

Price caps reduce exploitability and raise welfare relative to the no-regulation case in these rows.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0116 - Main Result - needs_artifact

In the Phase 3 table, price caps reduce exploitability across tested minds, but collusion is metric-sensitive for stronger learners.

- Suggested citations: `calvano2020, calvano2021`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0117 - Main Result - needs_artifact

DQN-family agents can sit effectively at the cap, with low price dispersion, while maintaining high total profit.

- Suggested citations: `mnih2015`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0118 - Main Result - needs_artifact

The short-run mechanism trace initially contradicted this headline result: at 1000 steps, profit fell under the cap for every tested mind.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0119 - Main Result - needs_artifact

A dedicated paired-seed audit resolves the discrepancy.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0121 - Main Result - needs_artifact

For DQN, the cap-minus-none profit delta is negative at 1000 steps, mixed at 5000 steps, mostly positive at 10,000 steps, and positive in the existing 40,000-step full-run audit.

- Suggested citations: `mnih2015`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0124 - Main Result - needs_artifact

The 40,000-step row comes from the existing n=20 full-run table; the shorter rows are fresh local audits. This correction also tightens the economic interpretation.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0126 - Main Result - needs_artifact

At 1000 and 10,000 steps, welfare and quantity can rise while profit behavior is unstable across seeds.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0127 - Main Result - needs_artifact

In the 40,000-step full table, the DQN profit effect is statistically positive (\( =55.358\), 95\% CI \(=[4.859,105.857]\), positive seed share \(0.85\)), but the mean quantity delta is small.

- Suggested citations: `mnih2015`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0128 - Main Result - needs_citation

The safer claim is that the price cap becomes a focal constraint that some DQN runs learn to exploit or coordinate around after sufficient training; it is not a simple short-horizon quantity story.

- Suggested citations: `mnih2015`
- Artifact support: `none`
- Fix: Consider adding \citep{mnih2015}.

## C0134 - What v0 Taught - needs_artifact

In the initial full run, successful trade was sparse and some institutions did not bind.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0135 - What v0 Taught - needs_citation

Follow-up replay showed that property rights almost never faced behavioral pressure: even when claims existed, non-owner opportunities to gather from claimed resource cells were extremely rare.

- Suggested citations: `mnih2015, schulman2017`
- Artifact support: `none`
- Fix: Consider adding \citep{mnih2015}, \citep{schulman2017}.

## C0137 - What v0 Taught - needs_citation

This is an important negative result for the platform: a world can be runnable, tested, and statistically replicated while still failing to exercise the institution it is supposed to study.

- Suggested citations: `schulman2017`
- Artifact support: `none`
- Fix: Consider adding \citep{schulman2017}.

## C0139 - v1 Hardening - needs_artifact

The v1 validation run verified that the economic channels were active, and the subsequent n=20 full run preserves that activation.

- Suggested citations: `myerson1981, milgrom2004`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0140 - v1 Hardening - needs_artifact

Table reftab:resource-v1 shows the medium validation values used as the pre-full-run gate.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0143 - v1 Hardening - needs_artifact

Reputation increases successful trade and welfare in the validation setting.

- Suggested citations: `fehrgachter2000, isaac1994, ostrom1990`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0144 - v1 Hardening - needs_artifact

Property-right opportunities are present, although the welfare effect is small in this medium run.

- Suggested citations: `schulman2017`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0145 - v1 Hardening - needs_artifact

The full n=20 v1 run confirms the activation pattern: successful trade appears under no institution, price controls bind and suppress trade, property-right opportunities are measured, and reputation produces the strongest welfare gains in this pressure setup.

- Suggested citations: `schulman2017, fehrgachter2000, isaac1994, ostrom1990`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0146 - v1 Hardening - needs_artifact

The strict-local ablation confirms that this remains partly a spatial-friction story: reducing trade from all-island matching to radius-one local matching cuts baseline trade from 5.325 to 1.900 and reputation trade from 6.667 to 1.927, while property opportunities and price-control blocks remain positive.

- Suggested citations: `schulman2017, fehrgachter2000, isaac1994, ostrom1990`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0151 - Cross-Mind Resource Island - needs_citation

PPO and centralized-critic discover the trade channel under pressure, while DQN and fixed independent-DQN remain more conservative and trade much less.

- Suggested citations: `mnih2015, schulman2017, tan1993`
- Artifact support: `outputs/resource_island_v1_full/summary_aggregate.csv, outputs/resource_island_v1_phase3_full/mind_comparison.csv`
- Fix: Consider adding \citep{mnih2015}, \citep{schulman2017}, \citep{tan1993}.

## C0156 - World - needs_artifact

The world reports seller revenue, bidder surplus, total welfare, allocative efficiency, regret, overbidding, underbidding, and distance from benchmark bid functions.

- Suggested citations: `myerson1981, milgrom2004, dutting2019, banchio2022`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0161 - Validation - needs_citation

Second-price learning moves toward the truthful benchmark, first-price learning shows underbidding and bid shading, and reserve prices increase revenue while changing allocative efficiency.

- Suggested citations: `vickrey1961, myerson1981, milgrom2004`
- Artifact support: `none`
- Fix: Consider adding \citep{vickrey1961}, \citep{myerson1981}, \citep{milgrom2004}.

## C0162 - Validation - needs_artifact

Table reftab:auction-validation summarizes the current validation run.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0163 - Cross-Mind Auction Diagnostics - needs_artifact

The full P.6 auction learner-suite run confirms that the auction world should be read against benchmark deviations, not raw reward alone.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0164 - Cross-Mind Auction Diagnostics - needs_artifact

In second-price auctions, PPO has the lowest ex-post regret among the tested learning minds (\(0.187\)) and higher allocative efficiency than Q-learning (\(0.762\) versus \(0.734\)).

- Suggested citations: `sutton2018, schulman2017, vickrey1961, dutting2019, banchio2022`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0165 - Cross-Mind Auction Diagnostics - needs_citation

DQN and fixed independent-DQN earn higher seller revenue but with lower allocative efficiency.

- Suggested citations: `mnih2015, tan1993, myerson1981, milgrom2004`
- Artifact support: `none`
- Fix: Consider adding \citep{mnih2015}, \citep{tan1993}, \citep{myerson1981}, \citep{milgrom2004}.

## C0166 - Cross-Mind Auction Diagnostics - needs_citation

The centralized-critic scaffold performs poorly in this asymmetric bidding setting, with low revenue and high regret.

- Suggested citations: `myerson1981, milgrom2004, dutting2019, banchio2022`
- Artifact support: `none`
- Fix: Consider adding \citep{myerson1981}, \citep{milgrom2004}, \citep{dutting2019}, \citep{banchio2022}.

## C0174 - World - needs_artifact

Deterministic regeneration makes the sustainability channel easy to audit.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0176 - World - needs_artifact

The main metrics are welfare, sustainability, total contribution, extraction relative to regeneration, inequality, collapse diagnostics, and tax revenue where applicable.

- Suggested citations: `myerson1981, milgrom2004, zheng2020, zheng2021`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0177 - Q-Learning Full Run - needs_artifact

The n=20 tabular Q-learning full run confirms the intended commons pressure.

- Suggested citations: `sutton2018, hardin1968, ostrom1990`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0178 - Q-Learning Full Run - needs_artifact

In the baseline, agents mostly extract and contribute little: sustainability is 0.0892 and total contribution is 0.1418.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0179 - Q-Learning Full Run - needs_artifact

Contribution matching improves both welfare and sustainability in this setting, reaching welfare 2.0808 and sustainability 0.1048.

- Suggested citations: `fehrgachter2000, isaac1994, ostrom1990`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0180 - Q-Learning Full Run - needs_artifact

Reputation is a strong reward-shaping intervention, with welfare 9.5750, while the tax schedule mainly changes redistribution/accounting at the tested rates.

- Suggested citations: `fehrgachter2000, isaac1994, ostrom1990, zheng2020, zheng2021`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0182 - Q-Learning Full Run - needs_artifact

This is a diagnostic classification, not yet a final welfare ranking.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0183 - Cross-Mind Status - needs_citation

The full Public Goods learner suite shows that stronger function approximation or centralized training does not automatically solve the commons problem.

- Suggested citations: `samuelson1954, olson1965, hardin1968, ostrom1990`
- Artifact support: `outputs/public_goods_full/summary_aggregate.csv, outputs/public_goods_phase3_full/mind_comparison.csv`
- Fix: Consider adding \citep{samuelson1954}, \citep{olson1965}, \citep{hardin1968}, \citep{ostrom1990}.

## C0184 - Cross-Mind Status - needs_artifact

Under no institution, Q-learning reaches welfare \(1.815\), sustainability \(0.089\), and contribution \(0.142\).

- Suggested citations: `sutton2018`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0185 - Cross-Mind Status - needs_artifact

DQN and fixed independent-DQN are close but contribute less; PPO and centralized-critic contribute almost nothing and sit near the lower sustainability regime.

- Suggested citations: `mnih2015, schulman2017, tan1993`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0187 - Cross-Mind Status - needs_artifact

It improves Q-learning sustainability from \(0.089\) to \(0.105\) and fixed independent-DQN sustainability from \(0.085\) to \(0.121\), with a larger increase in contribution for independent-DQN.

- Suggested citations: `sutton2018, mnih2015, tan1993`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0188 - Cross-Mind Status - needs_citation

It does not rescue PPO in this configuration.

- Suggested citations: `schulman2017`
- Artifact support: `none`
- Fix: Consider adding \citep{schulman2017}.

## C0193 - World - needs_artifact

The world reports match rate, total welfare, truthful-report rate, stability, blocking-pair diagnostics, and manipulation-gain diagnostics.

- Suggested citations: `vickrey1961`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0194 - World - needs_citation

Unlike Pricing Arena, Resource Island, Public Goods, and Auction House, not every economic actor is a learning agent.

- Suggested citations: `samuelson1954, olson1965`
- Artifact support: `outputs/phase3_full/mind_comparison.csv, outputs/resource_island_v1_full/summary_aggregate.csv, outputs/resource_island_v1_phase3_full/mind_comparison.csv, outputs/auction_house_full/summary_aggregate.csv, outputs/auction_house_phase3_full/mind_comparison.csv, outputs/public_goods_full/summary_aggregate.csv, outputs/public_goods_phase3_full/mind_comparison.csv`
- Fix: Consider adding \citep{samuelson1954}, \citep{olson1965}.

## C0195 - World - needs_artifact

That asymmetry is the main interface test for the learning-architecture suite: DQN, PPO, independent-DQN, and centralized-critic operate only over the worker side while employers remain fixed preference holders.

- Suggested citations: `mnih2015, schulman2017, tan1993`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0196 - Q-Learning Full Run - needs_artifact

The n=20 tabular Q-learning full run is substantially more stable than the short smoke run.

- Suggested citations: `sutton2018`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0197 - Q-Learning Full Run - needs_artifact

Match rate remains 1.0, while stability\_mean=0.9508, truthful\_report\_rate\_mean=0.7859, and total\_welfare\_mean=3.6393.

- Suggested citations: `vickrey1961`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0198 - Q-Learning Full Run - needs_citation

The report-top action space creates truthfulness variation, but worker-proposing deferred acceptance still produces mostly stable matchings under the current random-preference setup.

- Suggested citations: `vickrey1961, galeshapley1962, rothsotomayor1990`
- Artifact support: `none`
- Fix: Consider adding \citep{vickrey1961}, \citep{galeshapley1962}, \citep{rothsotomayor1990}.

## C0199 - Q-Learning Full Run - manual_verify

Fixed benchmark cases clarify the interpretation.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Manually verify this claim against foundation cards or add a precise citation/artifact.

## C0201 - Q-Learning Full Run - needs_citation

Future manipulation tests therefore need to target the non-proposing side, change the mechanism, or add information and commitment frictions rather than treating worker misreports under standard deferred acceptance as the main expected failure mode.

- Suggested citations: `galeshapley1962, rothsotomayor1990`
- Artifact support: `none`
- Fix: Consider adding \citep{galeshapley1962}, \citep{rothsotomayor1990}.

## C0203 - Cross-Mind Status - needs_artifact

DQN slightly improves stability relative to Q-learning (\(0.975\) versus \(0.951\)) while reporting truthfully less often.

- Suggested citations: `sutton2018, mnih2015, vickrey1961`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0204 - Cross-Mind Status - needs_citation

PPO and fixed independent-DQN preserve high stability but also lower truthfulness.

- Suggested citations: `mnih2015, schulman2017, tan1993, vickrey1961, myerson1981, milgrom2004`
- Artifact support: `none`
- Fix: Consider adding \citep{mnih2015}, \citep{schulman2017}, \citep{tan1993}, \citep{vickrey1961}.

## C0205 - Cross-Mind Status - needs_artifact

The centralized-critic scaffold has the weakest matching behavior in this asymmetric setup, with stability \(0.749\) and truthful-report rate \(0.433\).

- Suggested citations: `vickrey1961`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0207 - Cross-Mind Status - manual_verify

The result is better framed as a learned-reporting diagnostic: the mechanism keeps matches mostly stable for most learners, while the centralized-critic architecture is brittle in this asymmetric environment.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Manually verify this claim against foundation cards or add a precise citation/artifact.

## C0210 - Cross-World Synthesis - needs_citation

Tabular value learning, function approximation, on-policy policy gradients, decorrelated independent learners, and centralized critics violate different assumptions of the classical benchmark story.

- Suggested citations: `sutton2018, schulman2017, tan1993, lowe2017`
- Artifact support: `none`
- Fix: Consider adding \citep{sutton2018}, \citep{schulman2017}, \citep{tan1993}, \citep{lowe2017}.

## C0216 - Cross-World Synthesis - needs_citation

Labor Market adds a matching-stability and truthfulness channel.

- Suggested citations: `vickrey1961`
- Artifact support: `outputs/labor_market_full/summary_aggregate.csv, outputs/labor_market_phase3_full/mind_comparison.csv, outputs/labor_market_benchmark_cases.json`
- Fix: Consider adding \citep{vickrey1961}.

## C0219 - Cross-World Synthesis - needs_artifact

The claim-audit suite adds a second pass over those outputs: paired seed effects, benchmark gaps, metric-conflict warnings, activation confirmations, and trace/full-run sign agreement where paired traces exist.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0220 - Cross-World Synthesis - needs_artifact

In its current run, the audit produces 1,025 paired or benchmark rows and 31 metric-conflict or activation rows.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0221 - Cross-World Synthesis - needs_artifact

This audit layer prevents the cross-world synthesis from becoming a table of averages without claim boundaries.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0230 - Limitations - needs_artifact

They are designed for auditability rather than realism.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0234 - Limitations - needs_citation

Public Goods is still a stylized common-pool model rather than an empirical commons.

- Suggested citations: `samuelson1954, olson1965, hardin1968, ostrom1990`
- Artifact support: `outputs/public_goods_full/summary_aggregate.csv, outputs/public_goods_phase3_full/mind_comparison.csv`
- Fix: Consider adding \citep{samuelson1954}, \citep{olson1965}, \citep{hardin1968}, \citep{ostrom1990}.

## C0235 - Limitations - needs_citation

Labor Market currently uses worker-side learning under deferred acceptance, so it tests one asymmetric matching channel rather than matching-market design generally.

- Suggested citations: `galeshapley1962, rothsotomayor1990`
- Artifact support: `outputs/labor_market_full/summary_aggregate.csv, outputs/labor_market_phase3_full/mind_comparison.csv, outputs/labor_market_benchmark_cases.json`
- Fix: Consider adding \citep{galeshapley1962}, \citep{rothsotomayor1990}.

## C0236 - Limitations - manual_verify

The most important limitation is interpretive.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Manually verify this claim against foundation cards or add a precise citation/artifact.

## C0240 - Limitations - needs_artifact

The independent-DQN alias issue is a concrete example: the table was statistically valid as an output file, but not valid as evidence for a distinct mind until the implementation was corrected and rerun.

- Suggested citations: `mnih2015, tan1993`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0242 - Limitations - needs_artifact

A mechanism trace can reveal the local channel but cannot replace a replicated training-budget audit.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0243 - Limitations - needs_artifact

The 1000-step trace correctly showed early profit losses under the cap; the 40,000-step table correctly showed a positive DQN cap-minus-none profit delta.

- Suggested citations: `mnih2015`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0245 - Reproducibility - needs_artifact

The repository records validation status in ROADMAP\_STATUS.md.

- Suggested citations: `none`
- Artifact support: `none`
- Fix: Add or name the output artifact/table supporting this numeric/result claim.

## C0251 - Conclusion - needs_citation

Resource Island, Auction House, Public Goods, and Labor Market extend the project beyond pricing games.

- Suggested citations: `samuelson1954, olson1965`
- Artifact support: `outputs/resource_island_v1_full/summary_aggregate.csv, outputs/resource_island_v1_phase3_full/mind_comparison.csv, outputs/auction_house_full/summary_aggregate.csv, outputs/auction_house_phase3_full/mind_comparison.csv, outputs/public_goods_full/summary_aggregate.csv, outputs/public_goods_phase3_full/mind_comparison.csv, outputs/labor_market_full/summary_aggregate.csv, outputs/labor_market_phase3_full/mind_comparison.csv, outputs/labor_market_benchmark_cases.json`
- Fix: Consider adding \citep{samuelson1954}, \citep{olson1965}.
