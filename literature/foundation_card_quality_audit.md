# Foundation Card Quality Audit

Generated after pulling the completed foundation-card fill from ofi1 and rerunning the local theory-obligation audit.

## Integrity Summary

- Foundation paper cards: 33.
- Foundation coverage table: `literature/foundation_papers.csv`, 33 rows.
- PDF queue: `literature/foundation_pdf_queue.csv`, 0 rows. All selected foundation papers now have extracted text or manual text available.
- Novelty/gap table: `literature/novelty_gap_table.csv`, 120 rows.
- Theory gap report: `literature/theory_gap_report.csv`, 5 rows, one per implemented world.
- Obligation audit: `literature/obligation_audit.csv`, 109 rows.
- Obligation status after local audit fix: 10 `pass`, 99 `partial`, 0 `missing`.

## Broad Scout Output Audit

The older broad scout outputs are useful for discovery but should not be treated as thesis-facing citation material.

- `literature/papers_ranked.csv`: 326 metadata records across Pricing Arena, Resource Island, Auction House, Public Goods, and Labor Market.
- Source mix: 283 OpenAlex records and 43 Semantic Scholar records.
- `literature/manual_pdf_queue.csv`: 125 broad-search rows, including 73 without extracted text.
- `literature/pdf_text_report.csv`: 150 broad-search text-extraction attempts.
- `literature/theory_coverage.csv`: 15 rows, all marked `covered_review_needed`.
- `literature/scholar_comparison_report.csv`: 39 rows, all still require manual Google Scholar top-title entry before any Scholar/API comparison can be trusted.
- `literature/paper_cards/`: 300 broad generated cards. This folder is noisy: it includes many low-confidence cards and irrelevant search hits such as quantum machine learning, major depressive disorder, biochar, mass spectrometry, smart cities, AI in education, and knowledge graphs.

Decision: use the broad outputs only as a search/discovery reservoir. Use `literature/foundation_paper_cards/` as the canonical paper-card set for writing and obligation auditing.

## Audit Fix

The original obligation audit reported 3 missing rows, all from the PPO method card under `cross_world_methods`. That was an audit-mapping defect, not a true missing implementation: method-paper obligations live under `minds/`, `worlds/mind_ladder.py`, and cross-world output tables, not under `worlds/cross_world_methods/`.

The local audit now maps `cross_world_methods` to method code and ladder outputs, and it prefers `literature/foundation_paper_cards/` over older broad card folders. After rerunning, there are no missing obligations.

## Coverage By World

- `pricing_arena`: 6 foundation cards, 30 novelty-gap rows, 20 obligation rows.
- `resource_island`: 6 foundation cards, 25 novelty-gap rows, 20 obligation rows.
- `auction_house`: 5 foundation cards, 30 novelty-gap rows, 17 obligation rows.
- `public_goods`: 6 foundation cards, 30 novelty-gap rows, 20 obligation rows.
- `labor_market`: 5 foundation cards, 5 novelty-gap rows, 17 obligation rows.
- `cross_world_methods`: 5 foundation cards, 15 method-obligation rows.

## What Is Strong

- The pipeline now gives each world a theory anchor, implementation obligation, metrics-to-compare list, failure-mode list, and code/output comparison target.
- Auction House is correctly grounded in truthfulness, bid shading, reserve revenue/efficiency tradeoffs, regret, revenue, and allocative efficiency.
- Pricing Arena is correctly grounded in profit-normalized collusion rather than only price-normalized collusion, with explicit price-cap quantity/profit-channel warnings.
- Public Goods is correctly grounded in free-rider/social-optimum brackets, contribution, punishment/rewards, reputation, and information effects.
- Labor Market is correctly grounded in deferred acceptance, stability, blocking pairs, side-proposing incentives, and strategy-proofness checks.
- Resource Island is correctly treated as an activation-sensitive common-pool/resource-governance world, not as a clean theorem-reproduction environment.

## Cards Requiring Human Review Before Citation

These cards are now cleaned for the most obvious generated overclaims, but their prose should still be checked against the source before final citation.

- `1954_the_pure_theory_of_public_expenditure.md`: medium confidence OCR/book-like source. Use for the public-goods incentive-divergence benchmark only after checking the original pages.
- `1961_counterspeculation_auctions_and_competitive_sealed_tenders.md`: cleaned. The card now treats truthfulness/efficiency as the theory benchmark and identifies regret as this repo's diagnostic proxy.
- `1962_college_admissions_and_the_stability_of_marriage.md`: cleaned. The card now treats Gale-Shapley as a theorem/procedure anchor, not simulation evidence.
- `1968_the_tragedy_of_the_commons.md`: cleaned. Use as conceptual commons framing, not as a literal inevitability theorem for Resource Island.
- `1988_the_theory_of_industrial_organization.md`: cleaned. Use as an industrial-organization reference scale for Nash/Bertrand and joint-profit benchmarks; check chapters/pages before citing specific claims.
- `1990_governing_the_commons.md`: cleaned. Use as governance-design framing and case-based institutional evidence, not as proof that Resource Island reproduces real governance.
- `2000_conditional_cooperation_and_voluntary_contributions_to_public_goods.md`: cleaned and confidence metadata normalized to `high`; still cite as experimental behavioral evidence, not a formal theorem.
- `2010_a_review_of_design_principles_for_community_based_natural_resource_management.md`: cleaned but remains low confidence. Use only as a checklist for Resource Island v1 activation diagnostics until manually reviewed.
- `2017_multi_agent_actor_critic_for_mixed_cooperative_competitive_environments.md`: cleaned. Use as a centralized-training/decentralized-execution design obligation, not as a formal guarantee.
- `2017_proximal_policy_optimization_algorithms.md`: cleaned. Use as an implementation-obligation card for clipped ratio, old log probabilities, advantage estimates, value loss, entropy, and minibatch epochs.
- `2020_artificial_intelligence_algorithmic_pricing_and_collusion.md`: cleaned. The card now states experimental/simulation evidence and model-specific robustness, not universal proof.

## Practical Verdict

The theory-scout stack is now complete enough to support writing: it has the foundation paper list, source-text-backed cards, world-level gap report, novelty-gap table, and obligation audit. It is not a citation database that can be trusted blindly. Treat the generated cards as a strict review queue: they tell us which theory obligations each world must satisfy and where the repo evidence lives, but final paper prose still needs human verification of the exact claims.

## Remaining Work Before Thesis Citation

- Manually verify exact final citation wording against the source text for the papers used in `paper/main.tex`.
- For the papers that become central citations, replace generic generated lines such as "not stated in supplied text" or "limitations visible from supplied text" with checked claims or delete them.
- Keep `literature/foundation_paper_cards/`, not the older broad `literature/paper_cards/`, as the canonical card set for the paper.
