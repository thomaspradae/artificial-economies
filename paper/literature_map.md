# Literature Map for Artificial Economies Paper

This map is organized by paper section and environment. It separates sources
that should anchor the current draft from sources that are useful context or
future extensions.

## 1. Core Framing: Artificial Economies and Learning Agents

Main positioning:

- Agent-based computational economics (ACE): this is the broad methodological
  home. The paper's pitch is not "we invented economic simulation"; it is "we
  add a tested capability ladder, institution activation diagnostics, and
  exploitability-style validation to small artificial economies."
- Experimental economics / learning in games: useful for arguing that
  reinforcement learning is a legitimate behavioral model when equilibrium
  assumptions are too strong.
- Cooperative AI / MARL: useful for the move from single-agent optimality to
  social metrics such as cooperation, inequality, welfare, and robustness.

Must cite:

- Sutton and Barto (2018), for reinforcement learning background.
- Mnih et al. (2015), for DQN.
- Schulman et al. (2017), for PPO.
- Tan (1993), for independent multi-agent Q-learning.
- Dafoe et al. (2020), for the broader Cooperative AI frame.
- Holland and Miller (1991), for artificial adaptive agents in economic theory.
- Tesfatsion (2002, 2006), for agent-based computational economics.
- Roth (2002), for the "economist as engineer" view of market/mechanism
  design as engineering.
- Erev and Roth (1998), Camerer and Ho (1999), and Fudenberg and Levine (1998)
  for learning models in games.

Useful context:

- Agent-based computational economics surveys and Tesfatsion/Page-style ACE
  references.
- Experimental economics work on reinforcement and belief learning.

Search combinations to keep using:

- "agent-based computational economics" + "reinforcement learning"
- "learning in games" + "reinforcement learning" + "economics"
- "cooperative AI" + "economic institutions"
- "multi-agent reinforcement learning" + "mechanism design"
- Fagiolo, Moneta, and Windrum (2007), for empirical validation problems in
  agent-based economics.

## 2. Pricing Arena / Algorithmic Collusion

Main positioning:

Pricing Arena is closest to the algorithmic-collusion literature. The key
comparison is not just "do Q-learners collude?" but whether institutions that
look effective for one learner class remain effective for stronger learners and
under exploitability tests.

Must cite:

- Calvano, Calzolari, Denicolo, and Pastorello (2020), "Artificial
  Intelligence, Algorithmic Pricing, and Collusion." This is the central
  predecessor for repeated pricing with learning algorithms.
- Calvano et al. (2021), "Algorithmic Collusion with Imperfect Monitoring."
  This motivates information and monitoring variants.
- Dorner (2021), critical review of algorithmic collusion. Useful for not
  overclaiming from small simulations.
- Deng, Schiffer, and Bichler (2024), "Algorithmic Collusion in Dynamic Pricing
  with Deep Reinforcement Learning." This is directly relevant to the
  capability ladder: they compare tabular Q-learning with DRL algorithms and
  find algorithm-dependent collusion.
- Wang and Ye (2026), "Strategic Information Disclosure in Algorithmic
  Pricing." This is newer than the current draft's evidence but useful for
  information-design/regulatory framing.
- Frick (2026), "Convergence to collusion in algorithmic pricing." Useful for
  convergence-time framing with modern DRL and continuous prices.
- Assad et al. (2024), gasoline-pricing algorithms, for empirical antitrust
  motivation if the final paper wants a real-market bridge.
- Klein (2021), autonomous algorithmic collusion Q-learning pricing, as another
  simulation-side anchor.

Paper-specific angle:

- Our result fits here: price caps reduce exploitability across minds, but
  DQN-family agents can sit near the cap and preserve high profit through the
  quantity margin. That should be framed as "metric- and capability-sensitive
  institutional robustness," not simply "price cap works."

Need literature pass:

- Asker/Fershtman/Pakes-style work on algorithmic pricing and regulation. The
  exact citation still needs verification before final draft.

Search combinations:

- "algorithmic collusion" + "price cap"
- "algorithmic pricing" + "regulation" + "Q-learning"
- "deep reinforcement learning" + "Bertrand competition" + "collusion"
- "algorithmic collusion" + "exploitability" or "deviation"
- "information disclosure" + "algorithmic pricing" + "Q-learning"
- Ezrachi and Stucke, for legal/antitrust framing of algorithmic tacit
  collusion.

## 3. Resource Island / Sequential Social Dilemmas and Common-Pool Resources

Main positioning:

Resource Island is not just a gridworld. It belongs near sequential social
dilemmas, common-pool resource games, institutional enforcement, and property /
trade governance. The important methodological lesson is that a spatial economy
must be checked for institution activation before interpreting welfare changes.

Must cite:

- Leibo et al. (2017), "Multi-agent Reinforcement Learning in Sequential Social
  Dilemmas." Closest technical neighbor: temporally extended social dilemmas
  where cooperation is a property of policies, not one-shot actions.
- Hughes et al. (2018), "Inequity aversion improves cooperation in
  intertemporal social dilemmas." Relevant to inequality and social preferences.
- Koster et al. (2020), "Silly rules improve..." Relevant to institutions,
  enforcement, and compliance in learned foraging worlds.
- Ostrom (1990), "Governing the Commons." Needed for property rights,
  monitoring, sanctions, and common-pool institution design.
- Public goods / commons literature for free-riding and resource sustainability.
- Perolat et al. (2017), "A multi-agent reinforcement learning model of
  common-pool resource appropriation." This is the closest direct neighbor to
  Resource Island: partially observed Markov games, common-pool resources,
  exclusion, sustainability, and inequality.
- Pretorius et al. (2020), game-theoretic analysis of common-pool management
  with MARL. Useful for information structures and empirical game-theoretic
  analysis.
- Koster et al. (2024), deep RL to promote sustainable human behavior on a
  common-pool resource problem. Useful for planner/mechanism-discovery framing.
- Hardin (1968), for the classic tragedy-of-the-commons framing, but use
  Ostrom as the corrective: institutions can solve commons problems under the
  right monitoring/enforcement conditions.

Paper-specific angle:

- Resource Island v0 is a cautionary result: a tested world can still fail to
  exercise its institutions. The v1 pressure setting now creates contested
  resource access, unequal exchange, and specialization pressure.
- The current v1 result should be described as an activation-validated testbed,
  not as a final realistic model of property or trade institutions.

Future extension:

- Strict-local trade-radius ablations should be framed through spatial
  friction in common-pool and sequential-social-dilemma settings.

Search combinations:

- "common-pool resource" + "multi-agent reinforcement learning"
- "property rights" + "common-pool resource" + "reinforcement learning"
- "sequential social dilemma" + "commons" + "institution"
- "tragedy of the commons" + "MARL" + "sustainability"
- "redistribution" + "common-pool resource" + "deep reinforcement learning"

## 4. Auction House / Mechanism Design and RL Bidding

Main positioning:

Auction House is the theory-anchored environment. Unlike Resource Island, it has
clean benchmark behavior: truthful bidding in second-price auctions and bid
shading in first-price auctions.

Must cite:

- Vickrey (1961), for second-price truthfulness.
- Myerson (1981), for optimal auction design and reserve-price logic.
- Milgrom (2004), for auction theory and practical auction design.
- Banchio and Skrzypacz (2022), "Artificial Intelligence and Auction Design."
  This is directly aligned: Q-learning agents in repeated auctions, with
  first-price auctions producing tacit-collusive low bids while second-price
  auctions are more robust.
- Recent RL auction-equilibrium work, e.g. Rawat (2024), for using RL/self-play
  to approximate auction equilibria.
- Shah et al. (2025), "Learning from Synthetic Labs: Language Models as Auction
  Participants," for the later LLM-auction extension.
- Lotfi et al. (2026), "Large Language Models as Bidding Agents in Repeated
  HetNet Auction," as a directly relevant repeated-auction LLM-agent example.
- Roth and Ockenfels (2002), for observed strategic behavior in online
  second-price-style auctions and the importance of auction rules.
- Kagel and Levin, for experimental auction benchmarks if we expand the auction
  related-work section.

Paper-specific angle:

- Auction House should test whether the platform recovers known benchmark
  directions before making any new claim: second-price should move toward
  truthful bidding; first-price should show bid shading; reserves should trade
  off revenue and efficiency.

Search combinations:

- "Q-learning" + "first-price auction" + "collusion"
- "reinforcement learning" + "auction design"
- "bid shading" + "reinforcement learning"
- "second-price auction" + "learning agents" + "truthful"
- "large language models" + "auction participants"

## 5. Public Goods / Commons Future World

Main positioning:

This world should not be built as another arbitrary gridworld. It should be
anchored in public-goods games, punishment/reward institutions, and common-pool
resource sustainability.

Must cite:

- Public goods game literature on free riding, contribution incentives, and
  punishment/reward mechanisms.
- Sasaki and related work on institutional incentives and optional
  participation.
- Fehr and Gachter-style punishment-in-public-goods literature, if the final
  design includes sanctions.
- Ostrom, again, if framed as commons governance rather than pure public-good
  contribution.
- Isaac, Walker, and Williams (1994), for group size and voluntary provision of
  public goods.
- Kosfeld, Okada, and Riedl, for institution formation in public goods games.

Design implication:

- Metrics should include contribution rate, extraction rate, sustainability,
  collapse threshold, inequality, and welfare.

## 6. Labor Market / Matching Future World

Main positioning:

Labor Market should be anchored in two-sided matching and deferred acceptance,
not generic "employment simulation."

Must cite:

- Gale and Shapley (1962), "College Admissions and the Stability of Marriage."
- Roth and Sotomayor (1990), matching markets book.
- Roth (1984), matching medical interns/residents, if using NRMP motivation.
- Taywade, Goldsmith, and Harrison (2020), MARL for decentralized stable
  matching.
- Min et al. (2022), RL in Markov matching markets.
- Zong et al. (2026), "Learn to Match: Two-Sided Matching with Temporally
  Extended Feedback." This is useful if the labor-market world becomes dynamic
  with interviews, noisy match quality, or separations.
- Abdulkadiroglu and Sonmez, for school-choice mechanism design if the labor
  market becomes an education/matching variant.

Design implication:

- Benchmarks should include stability, blocking pairs, proposer-side
  strategy-proofness, match quality, and manipulation/regret under learned
  reporting.

Search combinations:

- "two-sided matching" + "multi-agent reinforcement learning"
- "deferred acceptance" + "reinforcement learning"
- "matching markets" + "strategic agents" + "learning"
- "labor market matching" + "Markov game"
- "stable matching" + "decentralized" + "MARL"

## 7. Central Planner / Tax Schedule

Main positioning:

This probably belongs as a cross-world institution or policy sweep unless a
real planner action/state model is specified.

Must cite:

- Zheng et al. (2020/2021), "The AI Economist." This is the closest prior art:
  learned tax policies in dynamic economies with adaptive agents.
- Standard optimal taxation references can be added later if the paper turns
  toward tax theory.

Design implication:

- If implemented, use it as a planner/institution benchmark over existing
  worlds first, not as a sixth world by default.

## 8. LLM Agents / Prompted Economic Agents

Main positioning:

LLM agents should be treated as a separate behavioral class, not a drop-in
replacement for rational agents or RL agents. The central risk is
interpretability and reproducibility: LLM agents may reason verbally, imitate
norms, fail rationality checks, or become sensitive to prompt framing.

Must cite:

- Park et al. (2023), "Generative Agents," for memory/planning/reflection
  architectures.
- Fan et al. (2023), "Can Large Language Models Serve as Rational Players in
  Game Theory?" for caution about using LLMs as game-theoretic agents.
- Shah et al. (2025), "Learning from Synthetic Labs," for LLMs in auction
  experiments.
- Backmann et al. (2025), "When Ethics and Payoffs Diverge," for LLM behavior
  in morally charged prisoner/public-goods dilemmas.
- Lotfi et al. (2026), for LLMs as bidding agents in repeated auctions.
- Li et al. (2026), behavioral consistency validation in stock-market
  simulation, for the warning that LLM-agent economic simulations need
  behavioral alignment checks.
- LLM market simulation papers such as ASFM and newer LLM-agent market papers
  as context, not yet core evidence.
- Apel, Erev, Reichart, and Tennenholtz (2020), for language-based persuasion
  games and the gap between numerical and natural-language strategic signals.

Design implication:

- The LLM track should start with a reduced-seed, high-scrutiny protocol:
  fixed prompts, logged observations/actions/reasoning if available, replayable
  transcripts, deterministic temperature where possible, and small worlds first.
- Metrics should compare LLM behavior against random, Q-learning, and DQN/PPO,
  but claims should be qualitative until variance and prompt sensitivity are
  measured.

Search combinations:

- "large language models" + "game theory" + "rational players"
- "LLM agents" + "public goods game"
- "large language models" + "auction participants"
- "LLM agents" + "market simulation"
- "LLM" + "economic agents" + "behavioral consistency"
- "LLM agents" + "social dilemma"

## 8.1 Search Matrix for Google Scholar / Semantic Scholar

Use these as exact combinations when doing manual Scholar searches. The goal is
to find papers at the intersection, not just canonical work from one column.

| World / theme | Learner terms | Institution terms | Metrics / benchmark terms |
| --- | --- | --- | --- |
| Pricing Arena | Q-learning; DQN; PPO; DRL | price cap; tax; audit; information disclosure | collusion index; Nash profit; monopoly profit; exploitability |
| Auction House | Q-learning; self-play; LLM bidder | first-price; second-price; reserve price; feedback disclosure | truthful bidding; bid shading; revenue; allocative efficiency; regret |
| Resource Island | MARL; independent learners; sequential social dilemma | property rights; redistribution; price controls; reputation | sustainability; inequality; cooperation; exclusion; commons |
| Public Goods | MARL; LLM agents; social preferences | punishment; reward; matching contributions; sanctions | contribution rate; free riding; welfare; collapse threshold |
| Labor Market | MARL; bandits; Markov matching | deferred acceptance; interviews; information disclosure | stability; blocking pairs; strategy-proofness; regret |
| Central Planner | two-level RL; planner-agent RL | taxation; redistribution; subsidies | equality-productivity tradeoff; welfare; manipulation |
| LLM Agents | GPT; LLM agents; generative agents | auctions; public goods; bargaining; markets | rationality; prompt sensitivity; behavioral consistency; transcript audit |

High-yield Boolean-style queries:

- `"algorithmic collusion" AND "deep reinforcement learning" AND pricing`
- `"price cap" AND "algorithmic pricing" AND "reinforcement learning"`
- `"Q-learning" AND "auction design" AND "first-price"`
- `"large language models" AND "auction participants"`
- `"common-pool resource" AND "multi-agent reinforcement learning"`
- `"property rights" AND "common-pool resource" AND "reinforcement learning"`
- `"two-sided matching" AND "multi-agent reinforcement learning"`
- `"deferred acceptance" AND "reinforcement learning"`
- `"LLM agents" AND "public goods game"`
- `"large language models" AND "game theory" AND rationality`

## 9. Immediate Paper Revision Tasks

1. Add the must-cite sources above to `paper/references.bib`.
2. Expand `paper/main.tex` related work into subsections:
   - algorithmic collusion,
   - MARL and sequential social dilemmas,
   - auction design,
   - economic simulation and policy learning,
   - LLM agents.
3. When the current full runs land, replace validation numbers with final
   n=20 Resource Island v1 and Auction House numbers.
4. Do not build LLM scaffolding until its protocol is written: prompt format,
   allowed memory, deterministic settings, transcript logging, and comparison
   baselines.

## 10. SerpAPI / Google Scholar Expansion Workflow

SerpAPI's Google Scholar endpoint is:

```text
https://serpapi.com/search?engine=google_scholar
```

The required query parameter is `q`; useful optional parameters include
`as_ylo`, `as_yhi`, `cites`, `cluster`, and date sorting with `scisbd`.

I added a local review-queue script:

```bash
SERPAPI_KEY=... python scripts/serpapi_scholar_lit.py \
  --output paper/serpapi_scholar_results.csv
```

Use this for breadth, not for automatic citation insertion. The correct workflow
is:

1. Run the script.
2. Sort by topic and citation count.
3. Manually inspect titles/abstracts.
4. Add only verified sources to `references.bib`.
5. Move accepted papers into the relevant section above.
