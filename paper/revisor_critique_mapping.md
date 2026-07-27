# Revisor Critique Mapping

This memo maps the reviewer-style criticism to the current code, outputs, and paper claims. The goal is not to defend the existing framing. The goal is to identify which claims survive, which need reframing, and what analysis is missing before the thesis can honestly claim more than a benchmark grid.

## 1. The Core Hit

The criticism lands on one real weakness: `capability ladder` sounds like a scalar ordering from weak to strong agents. The current code still encodes this idea directly in `build_cross_world_synthesis.py` through `CAPABILITY_TIERS`:

```text
random < q_learning < dqn < ppo < independent_dqn < centralized_critic
```

That ordering is not theoretically defensible as a general intelligence or capability scale. PPO is not universally more capable than DQN; DQN is not simply smarter than tabular Q-learning; independent-DQN is not necessarily higher capability than PPO; centralized critic can be worse in asymmetric environments.

The current data confirms this. Baseline outcomes are non-monotonic in almost every world:

| World | Baseline Pattern |
|---|---|
| Pricing Arena | DQN/PPO reduce exploitability relative to Q-learning, but centralized-critic has worse exploitability than Q-learning. |
| Resource Island v1 | PPO and centralized-critic trade heavily; DQN and independent-DQN trade much less. |
| Auction House | PPO has the best second-price regret; centralized-critic performs badly. |
| Public Goods | Q-learning contributes more than PPO and centralized-critic under baseline. |
| Labor Market | Centralized-critic has higher welfare but much lower stability/truthfulness. |

So the paper should not claim a monotone capability ladder unless it defines capability empirically per metric. The better framing is a learner-assumption stress test.

## 2. What Assumption Breaks Actually Map To The Code

The clean theoretical version is not fully implemented as isolated ablations. Some axes are real; others are confounded.

| Comparison | What It Could Mean | Current Status |
|---|---|---|
| Q-learning vs DQN | Tabular representation vs function approximation plus replay/target-network bootstrapping. | Partly clean, but exploration schedule, optimizer, network capacity, and feature encoding also change. |
| DQN vs PPO | Off-policy value learning vs on-policy stochastic policy optimization. | Real algorithmic contrast, but many details change at once, so not causal by itself. |
| DQN vs independent-DQN | Intended to represent independent MARL. | Weak as a theoretical axis. Plain DQN already constructs one DQN per agent in several worlds; fixed independent-DQN mainly decorrelates seed streams and coordinator wiring. Useful implementation condition, not a major assumption break. |
| PPO vs centralized-critic | Decentralized policy learning vs centralized training signal. | Stronger axis. The centralized critic observes joint observations during training, while actors remain decentralized. This is the clearest MARL-assumption comparison currently in the code. |
| Single-agent theory vs multi-agent learning | Stationarity breaks because other agents learn. | Important, but not isolated by `independent_dqn`; it is already true for tabular Q-learning, DQN, PPO, and centralized-critic in the multi-agent worlds. |

Bottom line: the paper can say the implementations differ along representation, update rule, and training-information axes. It should not say each row cleanly removes exactly one theory assumption unless extra ablations are added.

## 3. What The Existing Synthesis Already Finds

The current synthesis machinery produces monotonicity reports and FDR-filtered paired effects, but it does not yet directly search for the reviewer-facing patterns: reversals, metric lies, and assumption-break transition points. Running that audit manually against the current outputs gives several real candidates.

### A. Reversal Candidate: Pricing Arena Price Cap

Under `price_cap`, exploitability falls but DQN profit-normalized collusion rises relative to DQN baseline:

```text
DQN price_cap exploitability effect: -11.6
DQN price_cap profit_collusion_index effect: +0.122
```

For Q-learning and random minds, price cap lowers profit-normalized collusion. For DQN, it raises it. This is the strongest existing thesis result because it is both surprising and mechanistically interpretable: the cap can look effective under exploitability or price-based metrics while preserving profit extraction through a quantity/profit channel.

Claim status: strong, thesis-facing.

### B. Metric-Lie Candidate: Public Goods Reputation

Public Goods reputation raises welfare mostly through reward accounting, while sustainability and contribution do not improve reliably for several minds:

```text
DQN reputation: welfare +10.7, sustainability ~0, contribution ~0
PPO reputation: welfare +3.5, sustainability ~0, contribution ~0
independent-DQN reputation: welfare +9.22, sustainability ~0, contribution ~0
```

This is a real `metric lie`: welfare says the institution worked, but the commons state variables do not support a strong sustainability/cooperation claim.

Claim status: strong if framed as metric decomposition, not as institutional success.

### C. Reversal / Brittleness Candidate: Auction House Centralized-Critic

Centralized-critic performs badly in Auction House:

```text
second_price: efficiency 0.522, regret 1.436
first_price: efficiency 0.515, regret 1.974
```

PPO, by contrast, has the lowest second-price regret:

```text
PPO second_price regret: 0.187
Q-learning second_price regret: 0.270
```

This is not `stronger is better`; it is architecture-environment mismatch. The centralized critic has a training-information advantage but performs worse in this asymmetric bidding setting.

Claim status: good diagnostic, but be careful. It shows brittleness of this implementation, not a theorem about centralized critics.

### D. Activation-Reversal Candidate: Resource Island v0 to v1

Resource Island v0 made neural/MARL minds look like high-survival non-traders. Resource Island v1 changes that:

```text
v1 none trade_count:
Q-learning: 5.325
DQN: 0.640
PPO: 8.892
independent-DQN: 1.730
centralized-critic: 8.697
```

This is valuable because it proves the old non-trade result was not a final economic result. It was a design/training activation failure. Once contested resources, specialization pressure, and unequal trades exist, PPO and centralized-critic activate the market channel strongly.

Claim status: strong as a world-design lesson; moderate as an institution result.

### E. Labor Market Metric Tradeoff

Labor Market match rate is always 1.0, so the interesting margins are stability and truthful reporting:

```text
Q-learning: stability 0.951, truthfulness 0.786
DQN: stability 0.975, truthfulness 0.704
PPO: stability 0.950, truthfulness 0.575
centralized-critic: stability 0.749, truthfulness 0.433
```

Centralized-critic has higher welfare but worse stability/truthfulness. That is a metric tradeoff, not a clean success.

Claim status: useful, but constrained by the deferred-acceptance benchmark. Worker-side misreports should not be expected to improve canonical worker-proposing DA.

## 4. What The Current Scripts Miss

`build_cross_world_synthesis.py` is useful but too blunt for the revised thesis. It still assumes `capability_tier`, then reports monotonicity over that artificial ordering.

Missing analyses:

1. Reversal detector:
   - Find institutions that are welfare/state-improving for one mind but harmful for another.
   - Use paired seed effects and confidence intervals/FDR rows.

2. Metric-decomposition audit:
   - Flag cases where welfare/reward improves but the underlying state channel does not.
   - Examples: Public Goods reputation, Auction reserve revenue vs welfare, Pricing price cap exploitability vs profit-collusion.

3. Assumption-axis report:
   - Replace scalar `capability_tier` with categorical mind properties:
     - representation: tabular vs neural
     - update family: value-based vs policy-gradient
     - data regime: off-policy replay vs on-policy rollout
     - training information: decentralized vs centralized critic
     - stochastic independence: correlated/simple seeds vs decorrelated independent streams
   - Report effects by axis, not as one ladder.

4. Claim-strength classifier:
   - Strong: mechanism active, benchmark exists, effect significant, metric decomposition supports the interpretation.
   - Medium: mechanism active and effect visible, but attribution confounded.
   - Weak: runnable result, but mechanism inactive or benchmark contact thin.

## 5. What The Paper Should Change

Keep:

- The platform contribution.
- The five-world validated output discipline.
- The Pricing Arena price-cap finding.
- The Resource Island v1 activation lesson.
- The Public Goods metric-decomposition warning.
- The Auction House benchmark-deviation framing.

Change:

- Replace `capability ladder` as the main theoretical phrase with `learner-assumption stress test` or `learning-architecture stress test`.
- Keep `capability ladder` only as a descriptive shorthand for the implementation suite, not as a claimed ordinal theory.
- Stop implying DQN -> PPO -> independent-DQN -> centralized-critic is a monotone capability scale.
- Treat centralized-critic as a centralized-training-information condition, not as the top of a smartness ladder.
- Treat independent-DQN as a decorrelated independent value-learning baseline, not as the sole point where stationarity breaks.

## 6. What Is Good Enough Right Now

The project is not just a 4x5 grid search anymore, but only if the paper foregrounds the sharp findings:

1. Pricing Arena has a real reversal/metric conflict: price cap reduces exploitability while DQN profit-collusion rises.
2. Public Goods has a real metric lie: welfare/reputation rewards can improve without sustainability/cooperation improving.
3. Resource Island proves activation diagnostics are necessary: v0 under-tested institutions; v1 changes the learned economy.
4. Auction House shows benchmark deviation is the right object: centralized critic can be worse despite being a more complex MARL scaffold.
5. Labor Market shows outcome metrics split: high matching/welfare can coexist with lower stability/truthfulness.

That is enough for a thesis spine if written honestly:

> Learning architecture changes which institutional guarantees survive, but not along a universal capability ordering. The stable object is not a smartness ladder; it is a set of assumption violations that expose reversals, metric lies, and activation failures across economic mechanisms.

## 7. Immediate Next Build

Do not start new worlds. Build the missing synthesis layer:

```text
tools/synthesis/
  mind_assumption_schema.py
  detect_reversals.py
  metric_decomposition_audit.py
  assumption_axis_report.py
  claim_strength_report.py
```

Outputs:

```text
outputs/revisor_synthesis/
  mind_assumption_matrix.csv
  reversal_candidates.csv
  metric_lie_candidates.csv
  assumption_axis_effects.csv
  claim_strength_report.md
```

This is the analysis layer that converts the current validated runs from `four algorithms by five worlds` into a thesis argument.

## 8. Group-Size Robustness Critique

The reviewer-style `does it change with n agents?` objection is also real. It matters differently by world.

### Public Goods

Public Goods is already parameterized by group size. `PublicGoodsConfig.n_agents` exists, `run_public_goods_smoke.py` exposes `--n-agents`, and `run_public_goods_group_size_sweep.py` now runs the actual `n_agents x mind x institution` sweep with a scaling-effects table.

This should be tested before making strong claims about commons behavior because Olson-style group-size logic predicts that free-riding worsens as group size grows. If PPO and centralized-critic contribute almost nothing at `n=4`, the key question is whether that remains true at larger groups or whether the effect is an artifact of the current small-group setup.

Minimum sweep:

```text
n_agents in {2, 4, 8, 16}
minds: q_learning, dqn, ppo, independent_dqn, centralized_critic
institutions: none, contribution_matching, public_goods_reputation, tax_schedule
metrics: contribution_total, contribution_rate, sustainability, collapse_rate, welfare
```

What this answers:

- Does free-riding intensify as individual pivotality falls?
- Does contribution matching remain state-changing at larger group sizes?
- Does the Public Goods `metric lie` persist, where welfare/reputation rewards improve without sustainability or contribution improving?
- Is the PPO/centralized-critic low-contribution pattern robust or just a small-n result?

Implementation status: sweep machinery is built and smoke-tested. The full `n_agents in {2, 4, 8, 16}` run is still pending.

### Pricing Arena

Pricing Arena was the one world that was not initially parameterized by firm count. That has now been fixed at the world/training level: `PricingArenaWorld` accepts any `len(config.quality) >= 2`, emits per-firm and aggregate metrics, the feature encoder handles N previous firm actions, static symmetric N-firm benchmark references exist, `anti_collusion` uses max-min price closeness rather than `p1/p2`, and `run_multiseed.py` exposes `--n-firms`.

This matters because the price-cap result is the strongest thesis finding, and classical IO predicts tacit collusion becomes harder as the number of firms rises. If the DQN price-cap profit channel only exists at duopoly scale, that is a boundary condition on the headline result.

Minimum sweep:

```text
n_firms in {2, 3, 4, 5}
mechanisms: none, price_cap
minds: q_learning, dqn, ppo
metrics: profit_collusion_index, exploitability, avg_price, price_dispersion, quantity_total, profit_total
```

What this answers:

- Does DQN-family cap-pinning survive more competitors?
- Does exploitability suppression under price caps remain robust as market concentration falls?
- Does collusion decay with n in the expected IO direction?

Implementation status: multiseed training is now parameterized, and `run_exploitability.py --n-firms --adversary-index` defines a one-deviator protocol: freeze N-1 incumbent policies, replace one target firm with a fresh tabular adversary, select the best restart, and report target/frozen-others metrics. This supports honest N-firm one-deviator exploitability sweeps. It still does not define coalition exploitability.

## 9. Updated Priority After The Group-Size Critique

1. Build the revisor synthesis layer: reversals, metric lies, assumption-axis effects, and claim strength.
2. Run the Public Goods full group-size sweep, because it is cheap and theory-loaded.
3. Run the Pricing Arena N-firm multiseed boundary sweep for `none` and `price_cap`.
4. Run the Pricing Arena one-deviator N-firm exploitability sweep for `none` and `price_cap`; do not interpret it as coalition exploitability.
5. Do not add a sixth world before these are resolved.
