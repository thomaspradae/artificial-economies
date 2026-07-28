

## Pricing Arena Writing Note

### 1. What This World Is

Pricing Arena is a repeated oligopoly/duopoly pricing game. Agents choose prices from a discrete grid. Demand comes from a logit demand system. Profits are:

```text
profit_i = (price_i - cost) * quantity_i
```

The world records average price, price dispersion, total profit, total quantity, consumer surplus, welfare, price-normalized collusion index, profit-normalized collusion index, exploitability, victim loss, and welfare damage.

Code anchors:

- [worlds/pricing_arena/env.py](/home/t/Downloads/fogo/thesis/worlds/pricing_arena/env.py)
- [worlds/pricing_arena/benchmarks.py](/home/t/Downloads/fogo/thesis/worlds/pricing_arena/benchmarks.py)

### 2. Classical Economic Question

The classical question is:

> Can market institutions discipline pricing behavior?

In normal industrial-organization terms:

- Competition should push prices toward a Nash/static competitive level.
- Joint-profit pricing is the collusive/monopoly-like reference.
- A price cap should prevent high-price collusion.
- Taxes, audits, and anti-collusion penalties should make high-price coordination less attractive.

So the benchmark is not "PPO should get value 10." It is:

```text
Where do learned prices/profits sit relative to static Nash and joint-profit benchmarks?
```

Known-answer sanity checks pass for Pricing Arena:

- `n=2`: static Nash best-response check passes.
- `n=3`: static Nash best-response check passes.
- `n=4`: static Nash best-response check passes.
- `n=5`: static Nash best-response check passes.

That means the payoff/benchmark layer is not arbitrary; the code recovers the static economic reference where theory gives one.

Artifact:

- [known_answer_sanity_checks.md](/home/t/Downloads/fogo/thesis/outputs/known_answer_sanity_checks_current/known_answer_sanity_checks.md)

### 3. What Institutions Are Tested

Main mechanisms:

- `none`: baseline market.
- `price_cap`: clamps realized prices before demand.
- `tax_high_price`: subtracts a tax from rewards above a threshold.
- `random_audit`: stochastic punishment.
- `anti_collusion`: penalizes high and close prices.
- `demand_shock`: perturbs market size.

The important one so far is `price_cap`, because it gives the cleanest mechanism story.

Code anchors:

- [institutions/price_cap.py](/home/t/Downloads/fogo/thesis/institutions/price_cap.py)
- [institutions/tax_high_price.py](/home/t/Downloads/fogo/thesis/institutions/tax_high_price.py)
- [institutions/anti_collusion.py](/home/t/Downloads/fogo/thesis/institutions/anti_collusion.py)

### 4. Headline Results

From the full n=20 Phase 3 Pricing Arena table:

- [outputs/phase3_full/mind_comparison.csv](/home/t/Downloads/fogo/thesis/outputs/phase3_full/mind_comparison.csv)

Price cap versus no institution:

| Mind | Price Change | Profit Change | Welfare Change | Exploitability Change | Interpretation |
|---|---:|---:|---:|---:|---|
| Q-learning | `-0.759` | `-22.916` | `+19.567` | `-55.366` | Cap works conventionally: lower price, lower firm profit, lower exploitability. |
| Random | `-1.185` | `-16.028` | `+34.779` | `-31.775` | Mechanical cap effect, not strategic. |
| DQN | `+0.388` | `+55.358` | `+6.294` | `-11.619` | Main surprising result: exploitability falls, but profit rises. |
| PPO | `-0.042` | `-6.558` | `-1.198` | `-5.704` | Smaller, mostly muted response. |
| Independent-DQN | `-0.288` | `+3.093` | `+12.183` | `-14.871` | Distinct from DQN after fix; mild corroboration, not as strong. |
| Centralized critic | `-1.337` | `+10.347` | `+28.668` | `-98.274` | Exploitability falls sharply, profit still rises slightly. |

The cleanest result:

> Price caps reduce exploitability for every tested mind, but they do not uniformly reduce firm profit.

That is the "guardrail works on one axis, leaks on another" finding.

### 5. The DQN Quantity-Margin Finding

The surprising thing is DQN.

No institution:

```text
avg_price = 3.611
profit_total = 236.345
exploitability = 20.269
```

Price cap:

```text
avg_price = 3.999
profit_total = 291.703
exploitability = 8.650
```

So under price cap:

- exploitability falls,
- welfare slightly rises,
- but DQN profit rises a lot.

That means the cap did not simply "discipline" the learner. It changed the strategic landscape. DQN found a better profit region under the regulated market than under the unregulated one.

Important correction: this is not uniformly every seed sitting exactly at one cap price. The audit shows seed heterogeneity. Some DQN seeds are near the high capped region, others are closer to Nash-like pricing. The aggregate profit effect is still real, but the mechanism is learned and seed-sensitive.

Artifact:

- [pricing_quantity_margin_audit/diagnosis.md](/home/t/Downloads/fogo/thesis/outputs/pricing_quantity_margin_audit/diagnosis.md)

### 6. The Contradiction We Resolved

There was a serious contradiction:

- Short 1000-step mechanism traces showed DQN profit falling under the cap.
- Full n=20 results showed DQN profit rising under the cap.

The audit found this is mainly a training-budget effect.

DQN budget ladder:

| Steps | DQN Profit Delta, Cap - None |
|---:|---:|
| `1000` | `-11.711` |
| `5000` | `+1.730` |
| `10000` | `+35.448` |
| `40000` | `+55.358` |

Safe statement:

> DQN does not immediately exploit the price-cap environment. The profit-preserving/profit-improving behavior emerges with longer training.

That is much stronger than pretending the short trace and full run always agreed.

### 7. What Survives, What Breaks

Original thesis question:

> When institutions designed around classical economic assumptions are placed in front of learning agents, which guarantees survive, which break, and does the break depend on learning architecture?

For Pricing Arena:

Survives:

- Static benchmark layer is valid.
- Price caps reduce exploitability broadly.
- Q-learning behaves closer to the standard intuition: cap lowers price and firm profit.
- Consumer/welfare side often improves under caps.

Breaks or weakens:

- "Price cap reduces firm profit/collusion" is not architecture-invariant.
- DQN can produce higher profit under the cap.
- Profit-normalized collusion and price-normalized collusion can tell different stories.
- Short traces are not enough; long-training behavior can reverse the sign.

Depends on architecture: yes.

- Q-learning: cap disciplines profit.
- DQN: cap can increase profit after enough training.
- PPO: smaller response.
- Independent-DQN: distinct, weaker version of DQN effect.
- Centralized critic: exploitability falls sharply, profit does not fall.

### 8. What This World Contributes
    
Pricing Arena gives the cleanest "institutional guardrail failure" story:

> A rule can succeed on its explicit target, reducing exploitability/high-price behavior, while still creating a new strategic channel that some learning architectures exploit.

This is not "DQN is smarter." The better framing is:

> DQN introduces function approximation and longer-horizon learned policy behavior. Under the price cap, that approximate learner can find a different region of the policy space than tabular Q-learning.

### 9. What We Cannot Claim Yet

Do not overclaim:

- We cannot say price caps always increase DQN profit.
- We cannot say this is a universal real-market prediction.
- We cannot say the repeated game is fully solved theoretically.
- The n-firm boundary sweep is still running, so we do not yet know whether the result survives at `n=3/4/5`.

Current claim:

> In the validated n=2 Pricing Arena full runs, price caps robustly reduce exploitability, but their profit/collusion effects are learning-architecture dependent. DQN shows an emergent long-training reversal: the cap lowers exploitability while increasing firm profit relative to the unregulated DQN baseline.

### 10. Mechanism-by-Mechanism Interpretation

For Pricing Arena, each mechanism tests a different kind of institutional intervention. Only `price_cap` is strongly, consistently interpretable across minds right now. The others are useful, but several are weak, inactive, parameter-sensitive, or perverse depending on the learner.

#### Baseline: `none`

This is the uncontrolled pricing game.

| Mind | Avg Price | Profit | Welfare | Collusion Index | Exploitability |
|---|---:|---:|---:|---:|---:|
| Q-learning | `5.487` | `365.3` | `810.4` | `0.541` | `95.7` |
| DQN | `3.611` | `236.3` | `832.8` | `0.202` | `20.3` |
| PPO | `3.681` | `260.0` | `839.3` | `0.215` | `14.2` |
| Independent-DQN | `4.078` | `269.9` | `827.9` | `0.285` | `26.6` |
| Centralized critic | `5.925` | `288.6` | `791.6` | `0.618` | `147.5` |

Interpretation: Q-learning and centralized critic are the more collusive/high-price baselines. DQN/PPO are lower-price, higher-welfare baselines.

#### `price_cap`

Code: clamps realized prices before demand.

Question: Can a hard rule prevent high-price collusion/exploitability?

Result versus `none`:

| Mind | Profit Change | Welfare Change | Exploitability Change | Read |
|---|---:|---:|---:|---|
| Q-learning | `-22.9` | `+19.6` | `-55.4` | Works conventionally. |
| Random | `-16.0` | `+34.8` | `-31.8` | Mechanical consumer benefit. |
| DQN | `+55.4` | `+6.3` | `-11.6` | Main reversal: exploitability falls, profit rises. |
| PPO | `-6.6` | `-1.2` | `-5.7` | Mild discipline. |
| Independent-DQN | `+3.1` | `+12.2` | `-14.9` | Mild profit preservation. |
| Centralized critic | `+10.3` | `+28.7` | `-98.3` | Big exploitability reduction, profit does not fall. |

Finding:

> Price cap is a guardrail that survives on exploitability but breaks as a profit-disciplining guarantee for DQN/MARL-style learners.

This is the strongest Pricing Arena mechanism.

#### `tax_high_price`

Code: subtracts a quantity-weighted tax from rewards above a high-price threshold.

Question: Can soft punishment discourage high-price behavior?

Result versus `none`:

| Mind | Price Change | Profit Change | Exploitability Change | Read |
|---|---:|---:|---:|---|
| Q-learning | `-0.078` | `+2.2` | `-7.6` | Slight exploitability reduction, no real profit discipline. |
| DQN | `+0.006` | `+0.6` | `+1.2` | Basically no effect. |
| PPO | `-0.023` | `-3.6` | `-5.2` | Mild effect. |
| Independent-DQN | `+0.001` | `+0.2` | `+2.5` | Basically inert. |
| Centralized critic | `+0.088` | `+4.2` | `-1.2` | Weak/perverse on price and welfare. |

Finding:

> The high-price tax is mostly too weak or too avoidable under current parameters. It does not cleanly discipline learned pricing.

Classification: mostly inert / weak accounting intervention, not a headline result.

#### `random_audit`

Code: if average price is high, sometimes applies a penalty.

Question: Can stochastic enforcement deter collusion?

Result versus `none`:

| Mind | Price Change | Profit Change | Exploitability Change | Read |
|---|---:|---:|---:|---|
| Q-learning | `+0.114` | `+11.1` | `+32.0` | Perverse: worse exploitability. |
| DQN | `+0.035` | `+4.3` | `-0.5` | Nearly inert. |
| PPO | `-0.006` | `-1.0` | `-2.7` | Tiny improvement. |
| Independent-DQN | `+0.002` | `+0.2` | `+1.7` | Inert. |
| Centralized critic | `+0.125` | `+49.1` | `-21.7` | Profit rises a lot, exploitability falls some. |

Finding:

> Random audit does not robustly deter collusion. For Q-learning it makes exploitability worse; for centralized critic it reduces exploitability but increases profit.

Classification: potential perverse / noisy enforcement mechanism.

#### `anti_collusion`

Code: penalizes agents when prices are both high and close together.

Question: Can a direct anti-collusion rule punish coordinated high prices?

Result versus `none`:

| Mind | Price Change | Profit Change | Exploitability Change | Read |
|---|---:|---:|---:|---|
| Q-learning | `-0.048` | `-2.1` | `+31.9` | Price/profit barely lower, exploitability worse. |
| DQN | `+0.005` | `+0.5` | `0.0` | Inert. |
| PPO | `-0.067` | `-7.4` | `-5.5` | Mildly useful. |
| Independent-DQN | `+0.001` | `+0.2` | `0.0` | Inert. |
| Centralized critic | `-0.013` | `+0.1` | `+0.5` | Inert. |

Finding:

> The explicit anti-collusion penalty mostly fails to bind strongly enough or is avoided by behavior that still preserves bad outcomes.

Classification: mostly inert/brittle guardrail. This is useful because it says direct anti-collusion punishment is not automatically better than a blunt price cap.

#### `demand_shock`

Code: randomly scales market size with a lognormal demand shock.

Question: Does environmental volatility disrupt collusion or make institutions more robust?

Result versus `none`:

| Mind | Price Change | Profit Change | Welfare Change | Exploitability Change | Read |
|---|---:|---:|---:|---:|---|
| Q-learning | `+0.014` | `+4.0` | `+0.3` | `+11.4` | Slightly worse exploitability. |
| DQN | `-0.002` | `+1.3` | `+1.8` | `+1.3` | Almost neutral. |
| PPO | `-0.010` | `-0.3` | `+1.4` | `-6.0` | Mildly stabilizing. |
| Independent-DQN | `+0.001` | `+0.6` | `+1.2` | `+1.8` | Neutral. |
| Centralized critic | `-0.050` | `-9.2` | `-12.0` | `+2.7` | Hurts welfare/profit. |

Finding:

> Demand shocks do not consistently break collusion or improve robustness. They mainly test sensitivity to stochastic market size.

Classification: robustness check, not a strong institution result.

### 11. Implementation Versus Interpretation

All Pricing Arena mechanisms are implemented through the same institutional interface and covered by parity/unit/full-run validation. The weaker interpretation of `tax_high_price`, `random_audit`, `anti_collusion`, and `demand_shock` is not a claim that they are badly implemented.

The distinction is:

- Code validity: does the mechanism do what it says?
- Economic activation: does the mechanism bind often enough to change incentives?
- Result interpretability: does the mechanism produce a clear thesis claim across minds?

For all five mechanisms, code validity is broadly yes. For `price_cap`, economic activation and result interpretability are also strong. For the others, activation and interpretation are weaker, more parameter-sensitive, or more architecture-dependent.

Mechanism status:

| Mechanism | Implementation Status | Interpretation Status |
|---|---|---|
| `price_cap` | Strong | Strong, clean, cross-mind interpretable |
| `tax_high_price` | Implemented/tested | Economically weak under current parameters |
| `random_audit` | Implemented/tested | Mixed/perverse effects, not robust |
| `anti_collusion` | Implemented/tested | Mechanically valid but often too brittle/inert |
| `demand_shock` | Implemented/tested | Robustness perturbation, not really a policy mechanism |

Thesis-level takeaway:

> In learned pricing markets, institutional design is architecture-dependent. A blunt price cap robustly reduces exploitability, but other enforcement mechanisms are weak or perverse, and the cap itself can leave profit-preserving channels open for DQN-family learners. Directly targeting collusion is not enough; the learning architecture determines which strategic channel the agent finds.

## Resource Island Writing Note

Resource Island is Tier 3. Its central benchmark is not "economic theory says exact value = X." Its central validation question is:

> Does the world actually activate the economic channel the institution is supposed to govern?

This matters because property rights, trade controls, redistribution, and reputation cannot be interpreted if agents never contest resources or trade.

### 1. What This World Is

Resource Island is a spatial scarce-resource economy. Agents have a grid position, energy, food inventory, wood inventory, alive/dead state, local observations, and movement/gathering/trading actions.

Core action set:

```text
stay
move_up/down/left/right
gather
offer_food_for_wood
offer_wood_for_food
```

Code/design anchors:

- [worlds/resource_island/DESIGN.md](/home/t/Downloads/fogo/thesis/worlds/resource_island/DESIGN.md)
- [worlds/resource_island/env.py](/home/t/Downloads/fogo/thesis/worlds/resource_island/env.py)
- [institutions/resource_island.py](/home/t/Downloads/fogo/thesis/institutions/resource_island.py)

The world measures survival rate, welfare/reward, specialization, inequality over time, resource sustainability, trade attempts, successful trades, inventory-blocked trades, institution-blocked trades, property claims, property violations, and property opportunities.

The last group is the Tier 3 core: Resource Island needs activation diagnostics before institution effects can be interpreted.

### 2. Classical Economic Question

The question is:

> Do property rights, trade rules, redistribution, and reputation create productive exchange and survival in a scarce-resource economy?

Classical anchor:

- Property rights should reduce conflict over scarce resources.
- Trade should improve allocation when agents have heterogeneous needs or access.
- Specialization should make trade valuable.
- Reputation should support repeated exchange.
- Price controls may protect against exploitative exchange, but can also block mutually beneficial trade.
- Redistribution may reduce inequality but can distort effort/reward incentives.

Resource Island does not have a closed-form Nash benchmark. The benchmark is therefore conditional:

```text
Did the world create scarcity, contestation, heterogeneous needs, and trade opportunities?
Then, conditional on activation, did institutions change survival/trade/welfare?
```

### 3. Activation Requirement

The main thesis should not present Resource Island as a story about implementation versions. It should present Resource Island as a pressure-tested scarce-resource world.

The methodological rule is simple:

> Require activation diagnostics before interpreting institution effects.

The reported configuration includes contested resource layouts, specialization pressure, unequal trades, and diagnostic counters for property/trade activation. That is the configuration used for thesis-facing Resource Island interpretation.

### 4. Baseline

Baseline `none` result:

```text
survival_rate = 0.9439
welfare = 1.1065
trade_count = 5.3251
trade_attempt_count = 10.5507
specialization_index = 0.3895
inequality_over_time = 0.1558
resource_sustainability = 0.7638
```

Interpretation:

> The base Resource Island economy has meaningful exchange and scarcity. This is the correct configuration for interpreting Resource Island institutions.

### 5. Institution-by-Institution Results

#### `property_rights`

Mechanism: the first successful gather from a cell creates a claim. Later non-owner gather attempts from claimed cells can be blocked or penalized.

Result versus `none` under Q-learning:

```text
survival +0.0077
welfare +0.0474
trade_count +0.4861
property_claims = 1.8695
property_violations = 0.0468
property_opportunities = 27.8249
property_resource_opportunities = 14.4547
```

Interpretation:

> Property rights are activated. They create opportunity pressure and mildly improve survival/welfare/trade under Q-learning.

Caveat: violations remain low, so do not claim property rights broadly solve conflict. The safe claim is that the mechanism is measurable and mildly beneficial under this learner/configuration.

#### `trade_price_controls`

Mechanism: blocks trades whose exchange ratio exceeds a maximum allowed ratio.

Result versus `none` under Q-learning:

```text
survival -0.0249
welfare -0.1558
trade_count -5.3251
trade_count = 0.0000
trade_attempt_count = 9.2405
trade_institution_blocked_count = 4.7862
resource_sustainability -0.0300
```

Interpretation:

> Trade price controls bind. They block unequal trades and eliminate successful trade, reducing welfare and survival in this configuration.

This is economically coherent: a fairness/protection rule can shut down mutually useful exchange.

#### `reputation_system`

Mechanism: successful trades build reputation; reputation gives future reward bonuses.

Result versus `none` under Q-learning:

```text
survival -0.0138
welfare +0.2412
trade_count +1.3414
trade_attempt_count +0.5410
```

Interpretation:

> Reputation increases welfare and trade. Some welfare gain comes from reward bonuses, but trade also rises, so it is not purely accounting.

#### `redistribution`

Redistribution is implemented, but it is not part of the main activated Resource Island comparison currently emphasized in the thesis. The thesis-facing Resource Island result should focus on property rights, trade price controls, and reputation because those are the mechanisms with directly validated activation diagnostics in the reported comparison table.

### 6. Cross-Mind Results

Baseline `none`:

| Mind | Survival | Welfare | Trade | Trade Attempts | Specialization | Sustainability |
|---|---:|---:|---:|---:|---:|---:|
| Q-learning | `0.9439` | `1.1065` | `5.3251` | `10.5507` | `0.3895` | `0.7638` |
| DQN | `0.9102` | `1.0614` | `0.6398` | `1.0709` | `0.1790` | `0.9529` |
| PPO | `0.9150` | `1.2537` | `8.8919` | `9.8380` | `0.4052` | `0.8789` |
| Independent-DQN | `0.9064` | `1.0630` | `1.7300` | `2.1310` | `0.1524` | `0.9639` |
| Centralized critic | `0.9150` | `1.2484` | `8.6973` | `10.8525` | `0.4073` | `0.8920` |

Interpretation:

- PPO and centralized critic learn the trade economy.
- Q-learning also trades meaningfully.
- DQN and independent-DQN trade much less.
- DQN/independent-DQN preserve resources more, partly because they do less economic activity.

This is not the same pattern as Pricing Arena. In Pricing Arena, DQN finds a profit-preserving regulatory channel. In Resource Island, DQN-family learners underuse a productive trade channel at the current budget.

### 7. Architecture-Dependent Institution Activation

Property opportunities under `property_rights`:

```text
Q-learning = 27.8249
DQN = 8.2279
PPO = 27.1268
Independent-DQN = 6.5695
Centralized critic = 24.9577
```

Trade under `reputation_system`:

```text
Q-learning = 6.6665
DQN = 2.4212
PPO = 13.9870
Independent-DQN = 2.2882
Centralized critic = 7.7649
```

Welfare under `reputation_system`:

```text
Q-learning = 1.3477
DQN = 1.1686
PPO = 1.8244
Independent-DQN = 1.1541
Centralized critic = 1.4858
```

Interpretation:

> Reputation works best for minds that already learn or explore trade. It amplifies a trade channel only when the learning architecture discovers that channel.

Trade price controls eliminate successful trade for every mind:

```text
Q-learning = 0.0000
DQN = 0.0000
PPO = 0.0000
Independent-DQN = 0.0000
Centralized critic = 0.0000
```

But institution-block pressure differs by mind:

```text
Q-learning = 4.7862
DQN = 0.9085
PPO = 0.0406
Independent-DQN = 3.0438
Centralized critic = 0.4514
```

Interpretation:

> The formal rule survives: unequal trades are blocked. But the economic consequence depends on whether a mind tries to trade in the first place.

### 8. What Survives, What Breaks

Original thesis question:

> When institutions designed around classical economic assumptions are placed in front of learning agents, which guarantees survive, which break, and does the break depend on learning architecture?

For Resource Island:

Survives:

- Trade, property, and reputation can be made economically active if the world has contested resources, specialization pressure, and unequal exchange.
- Price controls successfully block unequal trades.
- Reputation increases welfare/trade when agents learn to use trade.
- Property rights create measurable claims and opportunity pressure.
- Q-learning, PPO, and centralized critic can sustain nontrivial trade economies.

Breaks or weakens:

- Non-activated configurations are not evidence about institutions.
- Property rights cannot be interpreted unless non-owners actually encounter claimed resources.
- Trade price controls can destroy trade rather than improve allocation.
- DQN/independent-DQN underuse the productive trade channel at the current training budget.
- High resource sustainability can be misleading: DQN-family learners preserve resources partly by not exploiting/trading much.

Depends on architecture: yes.

- Q-learning: trades meaningfully and responds to property/reputation.
- DQN: weak trade discovery; high sustainability but lower economic activation.
- PPO: strong trade/reputation behavior and high welfare.
- Independent-DQN: distinct from DQN, but still weak on trade.
- Centralized critic: strong trade behavior, close to PPO in baseline/reputation.

### 9. Pending Budget-Ladder Caveat

The DQN trade result should be stated as budget-conditioned until it receives the same audit discipline as Pricing Arena.

Current safe claim:

> At the validated 40k-step budget, PPO and centralized critic learn productive trade, while DQN and independent-DQN show low trade activation.

Stronger claim not yet justified:

> DQN-family learners structurally fail to discover trade in Resource Island.

Required audit before making the stronger claim:

- Resource Island DQN trade budget ladder at longer training budgets, e.g. `40k`, `80k`, `160k`, and possibly `320k` if feasible.

If DQN still does not trade at longer budgets, the result becomes a strong architecture finding. If it starts trading later, the finding becomes:

> DQN discovers trade more slowly than PPO/centralized critic.

### 10. Resource Island Thesis Takeaway

Clean current claim:

> In Resource Island, institutions affect real trade channels rather than only reward accounting. Reputation increases trade and welfare; trade price controls bind and suppress exchange; property rights create measurable access pressure. These effects are learning-architecture dependent: PPO and centralized critic learn high-trade policies, while DQN-family learners show low trade activation at the current training budget.

## Auction House Writing Note

Auction House is Tier 1. It is one of the most tightly benchmarked worlds in the project because auction theory supplies exact mechanism predictions: truthful bidding in second-price auctions, bid shading in first-price auctions, and reserve-price revenue/efficiency tradeoffs.

### 1. What This World Is

Auction House is a repeated single-item auction with independent private values. Each bidder privately observes its own value, submits a bid from a discrete grid, and receives surplus if it wins:

```text
reward_i = value_i - payment_i if bidder i wins, else 0
```

Code/design anchors:

- [worlds/auction_house/DESIGN.md](/home/t/Downloads/fogo/thesis/worlds/auction_house/DESIGN.md)
- [worlds/auction_house/env.py](/home/t/Downloads/fogo/thesis/worlds/auction_house/env.py)
- [worlds/auction_house/benchmarks.py](/home/t/Downloads/fogo/thesis/worlds/auction_house/benchmarks.py)
- [institutions/auction_house.py](/home/t/Downloads/fogo/thesis/institutions/auction_house.py)

The world records seller revenue, bidder surplus, realized welfare, maximum feasible welfare, allocative efficiency, welfare efficiency, ex-post regret, truthful-bid distance, first-price-shading distance, overbid rate, underbid rate, and no-sale rate.

### 2. Classical Economic Question

The guiding question is:

> Do learning bidders recover known auction-theory behavior under first-price, second-price, reserve, clock, and information variants?

Classical anchor:

- In second-price/Vickrey auctions with private values, truthful bidding is weakly dominant.
- In first-price auctions, bidders shade below value.
- Reserve prices can raise seller revenue while sacrificing allocative efficiency or bidder surplus.
- Clock auctions share a second-price-style dropout benchmark in the simplified ascending-clock abstraction used here.
- Public-signal and noisy-signal variants test whether altered information degrades allocation or bidding discipline.

This is a true theory-recovery world: when the learned agents fail, the failure is measured against a known benchmark rather than only against an oracle or diagnostic counter.

### 3. Known-Answer Sanity Checks

Known-answer checks pass for Auction House:

- `second_price`, `n=2`: truthful bidding has zero grid regret.
- `first_price`, `n=2`: the bid-shading benchmark bids below value.
- `second_price`, `n=3`: truthful bidding has zero grid regret.
- `first_price`, `n=3`: the bid-shading benchmark bids below value.

Artifact:

- [known_answer_sanity_checks.md](/home/t/Downloads/fogo/thesis/outputs/known_answer_sanity_checks_current/known_answer_sanity_checks.md)

This validates the economic instrument: the benchmark layer knows the textbook answers before learned bidders are interpreted.

### 4. What Institutions Are Tested

Auction House tests six auction scenarios:

- `second_price`: highest bidder wins and pays the second-highest bid.
- `first_price`: highest bidder wins and pays its own bid.
- `second_price_reserve`: second-price auction with a reserve/no-sale threshold.
- `clock`: simplified ascending-clock auction.
- `second_price_public_signal`: second-price payment rule with public-signal observation changes.
- `second_price_noisy_signal`: second-price payment rule with noisy private observations.

The first three are the core full Q-learning auction mechanisms. The last three extend the world to clock-auction and information-structure variants for the full neural/MARL learner suite.

Implementation status:

| Scenario | Code Status | Interpretation Status |
|---|---|---|
| `second_price` | Implemented, tested, full Q-learning and full P.6 outputs exist | Strong benchmark-deviation interpretation |
| `first_price` | Implemented, tested, full Q-learning and full P.6 outputs exist | Strong qualitative shading interpretation |
| `second_price_reserve` | Implemented, tested, full Q-learning and full P.6 outputs exist | Strong revenue/surplus tradeoff interpretation |
| `clock` | Implemented, tested, full P.6 outputs exist | Interpretable as a simplified benchmark-compatible clock variant |
| `second_price_public_signal` | Implemented, tested, full P.6 outputs exist | Information-structure stress test |
| `second_price_noisy_signal` | Implemented, tested, full P.6 outputs exist | Information-structure stress test |

All six scenarios are mechanically implemented and tested. The key distinction is that sealed-bid second-price, first-price, and reserve auctions have the cleanest economic benchmarks, while clock and information variants are currently best read as robustness extensions.

### 5. Q-Learning Full Results

Full Q-learning output:

- [outputs/auction_house_full/summary_aggregate.csv](/home/t/Downloads/fogo/thesis/outputs/auction_house_full/summary_aggregate.csv)

| Scenario | Revenue | Bidder Surplus | Welfare | Allocative Efficiency | Ex-Post Regret | Overbid | Underbid | No Sale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `second_price` | `3.100` | `3.141` | `6.241` | `0.734` | `0.270` | `0.414` | `0.423` | `0.000` |
| `first_price` | `3.453` | `2.865` | `6.318` | `0.769` | `1.361` | `0.066` | `0.814` | `0.000` |
| `second_price_reserve` | `3.938` | `2.122` | `6.060` | `0.796` | `0.150` | `0.359` | `0.441` | `0.172` |

Interpretation:

- Second-price does not produce perfect truthful/efficient learned bidding. The benchmark efficiency is `1.0`, but learned allocative efficiency is `0.734`.
- First-price produces clear bid shading/underbidding: underbid rate is `0.814`, and overbid rate is only `0.066`.
- Reserve raises seller revenue relative to plain second-price (`3.938` vs `3.100`) and lowers bidder surplus/welfare, with no-sale probability `0.172`.

### 6. Mechanism-by-Mechanism Interpretation

#### `second_price`

Mechanism: highest bidder wins and pays the second-highest bid.

Question: do learners recover truthful bidding and efficient allocation?

Result:

```text
revenue = 3.100
bidder_surplus = 3.141
welfare = 6.241
allocative_efficiency = 0.734
ex_post_regret = 0.270
truthful_bid_distance = 1.996
```

Finding:

> The mechanism has the dominant-strategy benchmark, but learned bidders do not fully recover it. The auction is theoretically strategy-proof, yet the learning dynamics leave substantial truthful-bid distance and imperfect allocative efficiency.

This is a benchmark-deviation result, not a mechanism-code failure.

#### `first_price`

Mechanism: highest bidder wins and pays its own bid.

Question: do learners shade bids below value?

Result:

```text
revenue = 3.453
bidder_surplus = 2.865
welfare = 6.318
allocative_efficiency = 0.769
underbid_rate = 0.814
overbid_rate = 0.066
first_price_shading_distance = 1.034
ex_post_regret = 1.361
```

Finding:

> Learned bidders recover the qualitative direction of first-price theory: they shade/underbid. But they do not converge tightly to the benchmark; regret remains high.

#### `second_price_reserve`

Mechanism: second-price auction with reserve/no-sale threshold.

Question: does a reserve raise revenue while trading off efficiency/surplus?

Result:

```text
revenue = 3.938
bidder_surplus = 2.122
welfare = 6.060
allocative_efficiency = 0.796
no_sale = 0.172
ex_post_regret = 0.150
```

Finding:

> Reserve pricing raises seller revenue but lowers bidder surplus and realized welfare relative to the no-reserve second-price auction. This matches the expected revenue/efficiency tradeoff direction.

#### `clock`

Mechanism: simple ascending-clock/dropout abstraction where the highest dropout wins and pays the second-highest dropout, subject to reserve.

Status:

- Full neural/MARL outputs exist.
- Q-learning variant smoke exists.
- Treat as validated and benchmark-compatible, but not as central as the sealed-bid mechanisms.

Cross-mind full highlights:

```text
DQN: revenue = 3.580, welfare = 6.157, efficiency = 0.704, regret = 0.255
PPO: revenue = 3.219, welfare = 6.402, efficiency = 0.769, regret = 0.186
Independent-DQN: revenue = 3.253, welfare = 6.036, efficiency = 0.689, regret = 0.275
Centralized critic: revenue = 0.060, welfare = 4.992, efficiency = 0.516, regret = 1.207
```

Finding:

> PPO is closest to the second-price-style benchmark in this clock abstraction; centralized critic collapses toward very low bids/revenue and poor efficiency.

#### `second_price_public_signal` and `second_price_noisy_signal`

Mechanism: information institution modifies bidder observations while preserving the one-bin observation shape. Public signal mixes own value with rival-value information; noisy signal perturbs observed value bins.

Status:

- Variant smoke and full neural/MARL outputs exist.
- These are information-structure stress tests, not separate payment-rule benchmarks.

Cross-mind full highlights:

Public signal:

```text
DQN efficiency = 0.614, regret = 0.333
PPO efficiency = 0.714, regret = 0.250
Independent-DQN efficiency = 0.636, regret = 0.375
Centralized critic efficiency = 0.504, regret = 1.487
```

Noisy signal:

```text
DQN efficiency = 0.703, regret = 0.270
PPO efficiency = 0.760, regret = 0.184
Independent-DQN efficiency = 0.705, regret = 0.295
Centralized critic efficiency = 0.525, regret = 1.512
```

Finding:

> Public/noisy information variants generally degrade or perturb allocation and regret, with PPO remaining closest to the benchmark among neural minds and centralized critic performing poorly in this asymmetric private-information setting.

### 7. Cross-Mind Results

Full cross-mind output:

- [outputs/auction_house_phase3_full/mind_comparison.csv](/home/t/Downloads/fogo/thesis/outputs/auction_house_phase3_full/mind_comparison.csv)

Second-price:

| Mind | Revenue | Surplus | Welfare | Efficiency | Regret | Truthful Distance |
|---|---:|---:|---:|---:|---:|---:|
| Q-learning | `3.100` | `3.141` | `6.241` | `0.734` | `0.270` | `1.996` |
| DQN | `3.593` | `2.522` | `6.115` | `0.705` | `0.260` | `2.392` |
| PPO | `3.233` | `3.156` | `6.389` | `0.762` | `0.187` | `1.780` |
| Independent-DQN | `3.630` | `2.494` | `6.124` | `0.707` | `0.280` | `2.294` |
| Centralized critic | `0.063` | `4.998` | `5.061` | `0.522` | `1.436` | `4.577` |

First-price:

| Mind | Revenue | Surplus | Welfare | Efficiency | Regret | Underbid Rate |
|---|---:|---:|---:|---:|---:|---:|
| Q-learning | `3.453` | `2.865` | `6.318` | `0.769` | `1.361` | `0.814` |
| DQN | `3.696` | `2.713` | `6.409` | `0.791` | `1.008` | `0.723` |
| PPO | `3.408` | `2.958` | `6.365` | `0.771` | `0.937` | `0.716` |
| Independent-DQN | `3.412` | `3.013` | `6.425` | `0.798` | `1.048` | `0.753` |
| Centralized critic | `1.280` | `3.736` | `5.016` | `0.515` | `1.974` | `0.832` |

Reserve:

| Mind | Revenue | Surplus | Welfare | Efficiency | Regret | No Sale |
|---|---:|---:|---:|---:|---:|---:|
| Q-learning | `3.938` | `2.122` | `6.060` | `0.796` | `0.150` | `0.172` |
| DQN | `4.384` | `1.788` | `6.172` | `0.755` | `0.226` | `0.104` |
| PPO | `3.804` | `2.157` | `5.961` | `0.780` | `0.107` | `0.213` |
| Independent-DQN | `4.419` | `1.749` | `6.168` | `0.718` | `0.276` | `0.074` |
| Centralized critic | `4.316` | `2.026` | `6.342` | `0.682` | `0.298` | `0.003` |

Interpretation:

- PPO is generally closest to the efficient/truthful benchmark in second-price and clock/noisy variants.
- DQN and independent-DQN often raise seller revenue but reduce bidder surplus and remain farther from truthful bidding.
- Centralized critic is poor in non-reserve second-price/clock/information settings, with very low revenue and high bidder surplus because it often underbids.
- In reserve settings, centralized critic no longer collapses revenue, because the reserve supports revenue mechanically.

### 8. What Survives, What Breaks

Original thesis question:

> When institutions designed around classical economic assumptions are placed in front of learning agents, which guarantees survive, which break, and does the break depend on learning architecture?

For Auction House:

Survives:

- The benchmark/mechanism layer is exact: truthful second-price zero-regret and first-price bid-shading checks pass.
- First-price learning recovers qualitative bid shading/underbidding.
- Reserve pricing raises seller revenue relative to no-reserve second-price in Q-learning full runs.
- The auction world produces interpretable incentive metrics: regret, truthful distance, revenue, surplus, efficiency.

Breaks or weakens:

- Learned bidders do not perfectly recover the dominant-strategy truthful behavior in second-price auctions.
- Allocative efficiency remains below the `1.0` truthful benchmark.
- First-price learners shade, but not tightly enough to eliminate regret.
- Centralized critic performs badly in several private-information auction settings.
- Public-signal/noisy-information variants can degrade efficiency and increase regret.

Depends on architecture: yes.

- Q-learning: economically coherent but imperfect; shows shading and reserve tradeoffs.
- DQN: often higher revenue, lower surplus, farther from truthful bidding.
- PPO: closest overall auction learner by regret/efficiency in many scenarios.
- Independent-DQN: distinct from DQN but broadly similar revenue-seeking/bid-distance pattern.
- Centralized critic: fails badly in non-reserve asymmetric-information settings, suggesting centralized value learning does not automatically solve private-information bidding.

### 9. Relation to the Main Thesis Question

Auction House answers the main question differently from Pricing Arena and Resource Island.

It does not primarily show that an institution becomes inactive or that a regulatory guardrail leaks through another strategic channel. It shows that exact mechanism-design guarantees are not automatically recovered by finite learned behavior. The second-price auction remains strategy-proof as a mechanism, but learned bidders do not perfectly learn truthful bidding. The first-price auction induces the correct qualitative direction, underbidding, but learned bidders still retain regret. The reserve mechanism produces the expected seller-revenue increase and surplus/welfare shift, but the exact magnitude depends on the learner.

Surviving guarantees:

- The code reproduces the exact auction-theory benchmark layer.
- First-price learning moves in the expected shading direction.
- Reserve pricing creates the expected revenue/surplus/no-sale tradeoff.

Broken or weakened guarantees:

- Dominant-strategy truthfulness in the mechanism does not imply learned truthful behavior.
- Allocative efficiency remains below the truthful benchmark under learned bidding.
- More complex training information does not guarantee better bidding: centralized critic performs poorly in several private-information auction settings.

Architecture dependence:

- PPO is the most benchmark-aligned learner in several auction scenarios.
- DQN and independent-DQN often increase revenue while moving farther from truthful bidding.
- Centralized critic is brittle in the asymmetric private-information settings, especially without a reserve.

### 10. What This World Contributes

Clean current claim:

> Auction House validates the platform against exact mechanism-design benchmarks, then shows that learned bidders do not automatically recover those benchmarks. First-price auctions induce the expected underbidding direction, reserve prices raise revenue while shifting surplus and no-sale behavior, and second-price auctions remain below perfect truthful/efficient allocation under learned policies. The failure is architecture-dependent: PPO is closest to the benchmark in several settings, while centralized critic performs poorly in private-information auctions.

### 11. What We Cannot Claim Yet

Do not overclaim:

- We cannot say the learning bidders converge to auction equilibrium.
- We cannot say centralized critic is generally bad for auctions; this is evidence about the current implementation and private-information setup.
- We cannot treat the clock and information variants as equally theory-tight as the sealed-bid second-price and first-price mechanisms.
- We cannot claim full learned mechanism design in the RegretNet sense; the auction rules are fixed, and the learners are bidders, not mechanism designers.

Current safe claim:

> In a benchmarked auction world with fixed auction rules, learned bidders recover some qualitative auction-theory directions but not the full theoretical guarantees. The gap is measurable with regret, truthful-bid distance, allocative efficiency, revenue, and surplus, and the size and direction of the gap depend on the learning architecture.

## Public Goods / Commons Writing Note

Public Goods is Tier 2. It has a strong classical direction but not a single scalar prediction for learned-agent behavior. Theory predicts free-riding and overuse when private incentives diverge from the group optimum. The code therefore owes bracketed comparison and state diagnostics: welfare alone is not enough.

### 1. What This World Is

Public Goods is a repeated common-pool resource and public-goods environment. Agents choose actions that contribute to the shared pool, extract from it, or remain inactive. Contributions are costly to the individual but support the public state; extraction gives private reward while depleting the pool. If requested extraction exceeds available stock, extraction is rationed. Deterministic regeneration makes the resource-state channel auditable.

Code/design anchors:

- [worlds/public_goods/DESIGN.md](/home/t/Downloads/fogo/thesis/worlds/public_goods/DESIGN.md)
- [worlds/public_goods/env.py](/home/t/Downloads/fogo/thesis/worlds/public_goods/env.py)
- [worlds/public_goods/benchmarks.py](/home/t/Downloads/fogo/thesis/worlds/public_goods/benchmarks.py)
- [institutions/public_goods.py](/home/t/Downloads/fogo/thesis/institutions/public_goods.py)
- [institutions/tax_schedule.py](/home/t/Downloads/fogo/thesis/institutions/tax_schedule.py)

The world records pool stock, sustainability, total contribution, total extraction, contribution and extraction rates, collapse rate, welfare, inequality, penalty totals, matched contribution, reputation bonuses, and tax revenue.

### 2. Classical Economic Question

The guiding question is:

> Do institutions prevent free-riding and commons collapse under different learning minds and group sizes?

Classical anchor:

- Public goods are vulnerable to underprovision because each individual contributor captures only part of the social benefit.
- Common-pool resources are vulnerable to over-extraction because each extractor imposes depletion costs on the group.
- Larger groups can worsen free-riding because each individual's contribution becomes less pivotal.
- Penalties, matching, reputation, monitoring, and taxes can change incentives, but only if they alter contribution/extraction behavior or resource state, not only reward accounting.

So the benchmark is not a single known value. It is:

```text
Does learned behavior move from the free-rider bracket toward the cooperative/social bracket, and does the resource state improve?
```

Known-answer sanity checks pass for Public Goods:

- `n=2`: free-rider/cooperative brackets pass.
- `n=4`: free-rider/cooperative brackets pass.
- `n=8`: free-rider/cooperative brackets pass.

Artifact:

- [known_answer_sanity_checks.md](/home/t/Downloads/fogo/thesis/outputs/known_answer_sanity_checks_current/known_answer_sanity_checks.md)

This validates the bracket layer before learned behavior is interpreted. It does not say a learned run should converge to the social optimum; it says the code can distinguish the free-rider and cooperative reference regimes.

### 3. What Institutions Are Tested

Public Goods tests six institution variants:

- `none`: baseline commons game.
- `public_goods_penalty`: penalizes extraction/free-riding behavior.
- `contribution_matching`: matches or rewards contribution to the shared pool.
- `public_goods_reputation`: adds reputation-based contributor rewards.
- `information_restriction`: coarsens or restricts observable public-state information.
- `tax_schedule`: taxes and redistributes rewards.

Implementation status:

| Institution | Code Status | Interpretation Status |
|---|---|---|
| `none` | Implemented, tested, full Q-learning and full P.6 outputs exist | Strong baseline commons-pressure interpretation |
| `public_goods_penalty` | Implemented, tested, full Q-learning and full P.6 outputs exist | Weak or slightly perverse under current parameters |
| `contribution_matching` | Implemented, tested, full Q-learning and full P.6 outputs exist | Clearest state-changing intervention |
| `public_goods_reputation` | Implemented, tested, full Q-learning and full P.6 outputs exist | Strong reward/welfare effect; state effect must be separated |
| `information_restriction` | Implemented, tested, full Q-learning and full P.6 outputs exist | Close to baseline under current parameters |
| `tax_schedule` | Implemented, tested, full Q-learning and full P.6 outputs exist | Mostly accounting/revenue at tested rate |

All six are mechanically implemented and tested. The important interpretive distinction is whether they move the underlying commons state, not only whether they move total reward.

### 4. Q-Learning Full Results

Full Q-learning output:

- [outputs/public_goods_full/summary_aggregate.csv](/home/t/Downloads/fogo/thesis/outputs/public_goods_full/summary_aggregate.csv)
- [outputs/public_goods_full/institution_effect_validation.json](/home/t/Downloads/fogo/thesis/outputs/public_goods_full/institution_effect_validation.json)

| Institution | Sustainability | Contribution | Extraction | Welfare | Collapse | Read |
|---|---:|---:|---:|---:|---:|---|
| `none` | `0.089` | `0.142` | `1.783` | `1.815` | `0.912` | Baseline free-riding/near-collapse pressure. |
| `public_goods_penalty` | `0.088` | `0.126` | `1.762` | `1.785` | `0.919` | Penalty binds but slightly worsens state/welfare here. |
| `contribution_matching` | `0.105` | `0.279` | `2.088` | `2.081` | `0.839` | Clearest state-improving Q-learning institution. |
| `public_goods_reputation` | `0.094` | `0.222` | `1.885` | `9.575` | `0.873` | Large reward/welfare effect plus modest state improvement. |
| `information_restriction` | `0.088` | `0.125` | `1.760` | `1.799` | `0.920` | Close to baseline or slightly worse. |
| `tax_schedule` | `0.089` | `0.134` | `1.773` | `1.808` | `0.916` | Mostly accounting/revenue, not state-changing. |

Interpretation:

> Baseline agents mostly extract and contribute little. Contribution matching is the cleanest Q-learning improvement because it changes both welfare and state variables. Reputation raises welfare strongly through bonuses, but its state improvement is much smaller than the welfare change. Tax schedule raises revenue but is classified as reward/accounting-only at the tested rate.

### 5. Mechanism-by-Mechanism Interpretation

#### `none`

Baseline condition: no institution changes contribution, extraction, information, or rewards.

Result:

```text
sustainability = 0.089
contribution_total = 0.142
extraction_total = 1.783
welfare = 1.815
collapse_rate = 0.912
```

Finding:

> The baseline exposes the intended commons pressure: extraction dominates contribution and collapse remains high.

#### `contribution_matching`

Mechanism: contribution to the shared pool is matched or rewarded.

Result relative to baseline:

```text
contribution_delta = +0.138
sustainability_delta = +0.016
collapse_delta = -0.073
welfare_delta = +0.266
```

Finding:

> Contribution matching is the clearest current Public Goods institution because it changes the underlying state, not just rewards. It raises contribution and sustainability while lowering collapse.

#### `public_goods_reputation`

Mechanism: contributor reputation creates reward bonuses for cooperative behavior.

Result:

```text
reputation_bonus_mean = 7.684
welfare = 9.575
contribution_total = 0.222
sustainability = 0.094
collapse_rate = 0.873
```

Finding:

> Reputation is partly state-changing and partly reward-shaping. It dramatically raises welfare through bonuses, but the state improvement is smaller than the welfare number suggests. This is the main Public Goods metric-decomposition warning.

#### `public_goods_penalty`

Mechanism: penalty schedule discourages extraction/free-riding behavior.

Result:

```text
penalty_total = 0.014
welfare_delta = -0.030
sustainability_delta = -0.001
collapse_delta = +0.007
```

Finding:

> The penalty binds, but at this configuration it does not improve the commons. It slightly lowers welfare and slightly worsens collapse, so it is a weak or perverse incentive setting under current parameters.

#### `information_restriction`

Mechanism: restricts or coarsens information available to agents.

Result:

```text
contribution_delta = -0.017
sustainability_delta = -0.001
welfare_delta = -0.016
collapse_delta = +0.009
```

Finding:

> Information restriction is close to baseline and slightly worse in the Q-learning full run. It does not currently produce a clean positive institutional result.

#### `tax_schedule`

Mechanism: taxes and redistributes rewards.

Result:

```text
tax_revenue_delta = +0.182
welfare_delta = -0.007
sustainability_delta = -0.001
classification = reward_or_accounting_only
```

Finding:

> Tax schedule changes accounting and revenue without materially changing the public-pool state at the tested rate. This is why welfare, tax revenue, contribution, extraction, and sustainability must be reported separately.

### 6. Cross-Mind Results

Full cross-mind output:

- [outputs/public_goods_phase3_full/mind_comparison.csv](/home/t/Downloads/fogo/thesis/outputs/public_goods_phase3_full/mind_comparison.csv)

Baseline `none`:

| Mind | Sustainability | Contribution | Extraction | Welfare | Collapse |
|---|---:|---:|---:|---:|---:|
| Q-learning | `0.089` | `0.142` | `1.783` | `1.815` | `0.912` |
| DQN | `0.085` | `0.071` | `1.692` | `1.748` | `0.954` |
| PPO | `0.080` | `0.000` | `1.600` | `1.680` | `1.000` |
| Independent-DQN | `0.085` | `0.072` | `1.693` | `1.749` | `0.952` |
| Centralized critic | `0.080` | `0.004` | `1.606` | `1.684` | `0.997` |

Contribution matching:

| Mind | Sustainability | Contribution | Welfare | Collapse | Read |
|---|---:|---:|---:|---:|---|
| Q-learning | `0.105` | `0.279` | `2.081` | `0.839` | Improves state and welfare. |
| DQN | `0.096` | `0.170` | `1.923` | `0.904` | Improves versus DQN baseline. |
| PPO | `0.080` | `0.000` | `1.680` | `1.000` | Almost no state response. |
| Independent-DQN | `0.121` | `0.413` | `2.269` | `0.805` | Largest contribution response. |
| Centralized critic | `0.081` | `0.007` | `1.690` | `0.995` | Barely responds. |

Reputation:

| Mind | Sustainability | Contribution | Welfare | Reputation Bonus | Read |
|---|---:|---:|---:|---:|---|
| Q-learning | `0.094` | `0.222` | `9.575` | `7.684` | Large reward effect, modest state effect. |
| DQN | `0.085` | `0.072` | `12.444` | `10.696` | Welfare rises without state improvement. |
| PPO | `0.080` | `0.000` | `5.183` | `3.503` | Welfare rises while contribution remains zero. |
| Independent-DQN | `0.085` | `0.073` | `10.973` | `9.224` | Welfare rises without material state improvement. |
| Centralized critic | `0.080` | `0.006` | `3.976` | `2.290` | Small state response. |

Interpretation:

- PPO and centralized critic are near-zero contribution learners in the baseline.
- DQN and independent-DQN contribute more than PPO, but still below Q-learning under no institution.
- Contribution matching works clearly for Q-learning, DQN, and especially independent-DQN.
- PPO and centralized critic mostly fail to use the contribution channel even when the institution exists.
- Reputation creates large welfare bonuses for all neural learners, but this is often an accounting/reward effect much larger than the state change.

### 7. Group-Size Sweep Status

The group-size sweep asks whether the free-riding mechanism scales with `n_agents`.

Question:

> Does contribution decline as group size rises, and do matching/reputation/tax institutions still work at larger `n`?

Current local status:

- The full Q-learning and full P.6 Public Goods outputs are validated locally.
- The expected full relaunch directory, `outputs/public_goods_group_size_sweep_full_relaunch/`, is not present locally at the time of this note.
- Local split directories exist for `n=2`, `n=4`, `n=8`, and `n=16`, but they currently contain no validated summary CSVs.
- Therefore group-size results should be treated as pending until completed outputs are pulled or regenerated and validated.

This matters because the group-size hypothesis is not optional for the commons claim. If contribution failure is a free-riding mechanism, it should change as individual pivotality changes with group size.

### 8. What Survives, What Breaks

Original thesis question:

> When institutions designed around classical economic assumptions are placed in front of learning agents, which guarantees survive, which break, and does the break depend on learning architecture?

For Public Goods:

Survives:

- The free-riding/commons pressure is visible under baseline.
- Contribution matching improves the resource state for several learners.
- The validator separates state-changing effects from reward/accounting effects.
- Public Goods runs Q-learning, DQN, PPO, independent-DQN, and centralized critic through the shared learner suite.

Breaks or weakens:

- Welfare can rise without equivalent improvement in contribution or sustainability.
- Tax schedule is mostly accounting-only at the tested rates.
- PPO and centralized critic can nearly stop contributing, even when the commons is collapsing.
- Penalty does not automatically improve the commons.

Depends on architecture: yes.

- Q-learning: contributes some and responds clearly to matching/reputation.
- DQN: lower baseline contribution, but responds to matching.
- PPO: near-zero contribution and weak response to matching.
- Independent-DQN: largest matching response in the current full table.
- Centralized critic: weak contribution and near-collapse under most institutions.

### 9. Relation to the Main Thesis Question

Public Goods shows that institutional success must be decomposed into reward and state channels. A mechanism can raise welfare while leaving the commons almost unchanged, which means the institution appears successful only if the wrong metric is read in isolation. Reputation is the cleanest example: it raises welfare sharply through bonuses, but sustainability and contribution move much less. Tax schedule is the stricter version of the same warning because it changes revenue/accounting while leaving the resource state close to baseline.

The architecture dependence is also substantive. Contribution matching changes behavior for Q-learning, DQN, and independent-DQN, but PPO and centralized critic remain close to zero contribution. That means the institution's effect is not only a property of the rule. It also depends on whether the learner discovers and reinforces the contribution channel.

### 10. What This World Contributes

Clean current claim:

> Public Goods shows a commons/free-riding failure that is not captured by welfare alone. Contribution matching can improve the underlying resource state, reputation can raise welfare much more than it improves state, and tax schedule can remain mostly accounting-like. The architecture effect is clear at fixed `n`: PPO and centralized critic often fail to contribute even when the mechanism exists, while independent-DQN responds strongly to contribution matching. The group-size sweep remains the boundary condition for whether this is a robust free-riding result rather than only a fixed-group-size pattern.

### 11. What We Cannot Claim Yet

Do not overclaim:

- We cannot say Public Goods institutions solve commons collapse.
- We cannot rank institutions by welfare alone.
- We cannot treat reputation's welfare gain as equivalent to sustainability improvement.
- We cannot make final group-size claims until the `n_agents` sweep outputs are completed and validated.
- We cannot infer real climate-governance results from this world; it is a controlled commons benchmark, not an external-policy model.

Current safe claim:

> In the validated fixed-size Public Goods runs, baseline agents mostly extract and contribute little. Contribution matching is the clearest state-improving institution, reputation produces a large welfare/reward effect with weaker state movement, and tax schedule is mostly accounting-like. These effects depend on the learning architecture, especially for PPO and centralized critic, which often fail to activate contribution at all.

## Labor Market Writing Note

Labor Market is Tier 1. Its benchmark is not a scalar payoff optimum but an exact property benchmark from matching theory: worker-proposing deferred acceptance gives stable matching and strategy-proofness for the proposing side under the canonical assumptions.

### 1. What This World Is

Labor Market is a two-sided matching world. Workers are learning agents. Employers have fixed preferences. Workers report a top-choice preference action, the world constructs reported preferences, and a deferred-acceptance-style institution produces a matching. Rewards come from matched utilities.

Code/design anchors:

- [worlds/labor_market/DESIGN.md](/home/t/Downloads/fogo/thesis/worlds/labor_market/DESIGN.md)
- [worlds/labor_market/env.py](/home/t/Downloads/fogo/thesis/worlds/labor_market/env.py)
- [worlds/labor_market/benchmarks.py](/home/t/Downloads/fogo/thesis/worlds/labor_market/benchmarks.py)
- [institutions/labor_market.py](/home/t/Downloads/fogo/thesis/institutions/labor_market.py)

The world records match rate, worker welfare, employer welfare, total welfare, blocking pairs, stability, truthful-report rate, manipulation-gain diagnostics, matched worker utility, and unmatched count.

### 2. Classical Economic Question

The question is:

> Do learning workers and matching institutions produce stable, truthful, welfare-improving matches?

Classical anchor:

- Deferred acceptance produces stable matchings.
- In worker-proposing DA, truthful reporting is strategy-proof for workers under the canonical complete-preference setting.
- A lower truthful-report rate is therefore not automatically evidence of profitable manipulation.
- The correct test is joint: stability, truthfulness, welfare, and manipulation diagnostics must be read together.

Known-answer sanity checks pass for Labor Market:

- Stable truthful matching case passes.
- Forced unstable matching case exposes blocking pairs.
- Welfare accounting case passes.
- Worker-proposing DA no-profitable-worker-report-deviation check passes.

Artifact:

- [outputs/labor_market_benchmark_cases.json](/home/t/Downloads/fogo/thesis/outputs/labor_market_benchmark_cases.json)
- [known_answer_sanity_checks.md](/home/t/Downloads/fogo/thesis/outputs/known_answer_sanity_checks_current/known_answer_sanity_checks.md)

### 3. Q-Learning Full Results

Full Q-learning output:

- [outputs/labor_market_full/summary_aggregate.csv](/home/t/Downloads/fogo/thesis/outputs/labor_market_full/summary_aggregate.csv)

Result:

```text
match_rate = 1.000
stability = 0.951
truthful_report_rate = 0.786
total_welfare = 3.639
blocking_pairs = 0.057
manipulation_gain = 0.003
benchmark_truthful_total_welfare = 4.807
```

Interpretation:

> The world mostly recovers the matching mechanism's stability property, but learned worker reports are not always truthful and welfare remains below the truthful benchmark. The low manipulation-gain diagnostic matters: lower truthfulness is not the same as profitable manipulation under worker-proposing DA.

### 4. Cross-Mind Results

Full cross-mind output:

- [outputs/labor_market_phase3_full/mind_comparison.csv](/home/t/Downloads/fogo/thesis/outputs/labor_market_phase3_full/mind_comparison.csv)

| Mind | Match Rate | Stability | Truthful Report | Welfare | Manipulation Gain |
|---|---:|---:|---:|---:|---:|
| Q-learning | `1.000` | `0.951` | `0.786` | `3.639` | `0.003` |
| DQN | `1.000` | `0.975` | `0.704` | `3.643` | `0.006` |
| PPO | `1.000` | `0.950` | `0.575` | `3.643` | `-0.012` |
| Independent-DQN | `1.000` | `0.955` | `0.672` | `3.658` | `0.003` |
| Centralized critic | `1.000` | `0.749` | `0.433` | `3.821` | `-0.009` |

Interpretation:

- Every learner keeps match rate at `1.0`.
- DQN has the highest stability (`0.975`) while truthfulness falls relative to Q-learning.
- PPO has similar stability to Q-learning but much lower truthfulness.
- Independent-DQN slightly raises welfare and keeps stability close to Q-learning.
- Centralized critic produces the clearest metric conflict: highest welfare, but much lower stability and truthfulness.

### 5. Mechanism Interpretation

#### `deferred_acceptance`

Mechanism: worker-proposing deferred acceptance using learned worker reports and fixed employer preferences.

Question:

> Does the canonical stable/strategy-proof mechanism remain stable and truthful when the reporting side is learned?

Finding:

> Stability mostly survives for Q-learning, DQN, PPO, and independent-DQN, but truthfulness weakens substantially for neural/MARL learners. Centralized critic is the sharp exception: welfare rises while stability and truthfulness fall hard.

This is not evidence that worker-proposing DA loses strategy-proofness in the theorem's own setting. The code's diagnostic says manipulation gains are near zero or negative for several minds. The better interpretation is behavioral: learners can produce non-truthful reports that do not look profitable in the benchmark sense, while still changing stability and welfare.

### 6. Metric-Lie Candidate

Labor Market currently contains one of the cleanest metric-conflict findings in the project.

Centralized critic:

```text
total_welfare = 3.821
stability = 0.749
truthful_report_rate = 0.433
```

Compared with Q-learning:

```text
total_welfare = 3.639
stability = 0.951
truthful_report_rate = 0.786
```

Finding:

> Centralized critic raises the headline welfare number while damaging the exact properties the institution exists to protect: stability and truthful reporting.

This is not "centralized critic is better." It is almost the opposite: the learner improves one aggregate while degrading the mechanism guarantees.

### 7. What Survives, What Breaks

Original thesis question:

> When institutions designed around classical economic assumptions are placed in front of learning agents, which guarantees survive, which break, and does the break depend on learning architecture?

For Labor Market:

Survives:

- Match rate is perfect across all learners.
- Stability remains high for Q-learning, DQN, PPO, and independent-DQN.
- The DA benchmark layer is exact and fixed benchmark cases pass.
- Worker-side profitable manipulation is not supported by the current diagnostic.

Breaks or weakens:

- Truthful reporting weakens for every neural/MARL learner relative to Q-learning.
- Welfare remains below the truthful benchmark.
- Centralized critic sharply lowers stability and truthfulness while increasing welfare.
- Lower truthfulness must not be casually described as profitable manipulation.

Depends on architecture: yes.

- Q-learning: stable, relatively truthful baseline.
- DQN: most stable, less truthful.
- PPO: stable but much less truthful.
- Independent-DQN: slightly higher welfare, moderate truthfulness loss.
- Centralized critic: highest welfare but major stability/truthfulness damage.

### 8. Labor Market Thesis Takeaway

Clean current claim:

> Labor Market shows that exact mechanism guarantees need the right behavioral reading. Deferred acceptance mostly preserves matching and stability under several learners, but learned reporting is not uniformly truthful. The strongest result is a metric conflict: centralized critic raises welfare while degrading stability and truthfulness. The safe interpretation is not that DA fails as a theorem, but that learned-agent implementations can optimize aggregate reward while eroding the institutional properties the theorem tells us to care about.

## Thesis Structure Implication

Do not narrow the thesis to only the clean-benchmark worlds. That would make the project smaller and would silently redefine the question as "institutions economists have already solved." The real question is broader:

> Which institutional guarantees survive when learning agents replace classical agents, and what kind of evidence is available when classical theory is exact, bracketed, or qualitative?

The tiering should be stated directly in the methods or environment-design section. It preempts the committee question: "What is the theory benchmark for each world?"

## Resource Island Placement

Resource Island should be framed as different in kind, not simply weaker. Its contribution is not a scalar theory-recovery benchmark. Its contribution is a pressure-tested institutional setting where the central requirement is to verify that the economic channel is active before interpreting institution effects.

## Draft Sentence

The environments are deliberately benchmarked at different levels of theoretical sharpness. Auctions, matching, and static pricing have canonical predictions that can be checked directly; public goods and taxation are evaluated against brackets and mechanism-specific diagnostics; Resource Island is a constructed diagnostic world where the central validation question is whether the institutional channel is exercised at all. This tiering is not a defect of the platform but a reflection of where economic theory itself supplies exact guarantees versus qualitative governance principles.
