# Theory Scout Audit

Audit date: 2026-07-20

## Verdict

The run completed cleanly and produced useful thesis infrastructure, but the output is not citation-ready without filtering. Treat `theory_obligations.md`, `theory_gap_report.csv`, and `novelty_gap_table.csv` as the usable high-level synthesis. Treat `paper_cards/` as a mixed candidate pool that needs manual triage before citation.

## Run Integrity

- `literature/scout_manifest.json` was written successfully.
- Runtime was about 4913 seconds.
- Raw metadata records: 326.
- Filled-card target: 75 cards, balanced as 15 per world.
- Final audit: 10 pass, 70 partial, 0 missing.
- Coverage status for every world: `covered_review_needed`.

## Coverage By World

| World | API Records | Extracted Text | Filled Cards | Main Status |
| --- | ---: | ---: | ---: | --- |
| Pricing Arena | 96 | 7 | 6 clean of 15 attempted | Strong core, poor PDF coverage |
| Auction House | 55 | 17 | 12 clean of 15 attempted | Strongest literature coverage |
| Public Goods | 56 | 13 | 13 clean of 15 attempted | Good theme coverage, noisy retrieval |
| Resource Island | 70 | 14 | 13 clean of 15 attempted | Useful commons/MARL leads, noisy retrieval |
| Labor Market | 49 | 13 | 13 clean of 15 attempted | Good classical anchors, noisy learning-side hits |

## Strong Artifacts

- `theory_obligations.md`: strongest thesis-facing output. It correctly translates each world into theory obligations and code checks.
- `theory_gap_report.csv`: compact five-world synthesis table.
- `novelty_gap_table.csv`: useful for mapping world x institution x mind rows to theory gaps.
- `obligation_audit.md`: useful repo-vs-theory checklist, but many rows are intentionally marked `partial` pending human review.

## Main Problems

1. Retrieval precision is too loose for broad worlds.
   - Public Goods pulled papers such as `Bounds on Multiprocessing Timing Anomalies` and plant-trait database material.
   - Resource Island pulled broad AI/edge-computing, CNN, HR-management, and architecture-search surveys.
   - Labor Market pulled GPT-4, education AI, psychiatric diagnosis, and generic ML dataset papers.

2. Some LLM cards are schema-complete but economically irrelevant.
   - A card can pass validation while still being useless as a thesis citation.
   - Validation currently checks shape, not relevance.

3. Some LLM extraction leaked project context into paper fields.
   - Example: one deferred-acceptance card put project world names into `Learning setup`.
   - This means filled cards must be reviewed before citation.

4. Semantic Scholar was present but rate-limited.
   - The run continued, but S2 did not contribute as reliably as intended.
   - `OPENALEX_MAILTO` and `UNPAYWALL_EMAIL` were not configured in the manifest.

5. `manual_pdf_queue.csv` is not purely "needs PDF".
   - It includes rows with extracted text already present.
   - Filter `has_extracted_text != True` before manual work.

## Citation-Ready Direction

Do not cite directly from the raw cards yet. Build the paper around these anchors first:

- Pricing Arena: Calvano et al. algorithmic pricing/collusion, Bertrand/static Nash, profit-normalized collusion.
- Auction House: Vickrey, Myerson, first-price bid shading, RegretNet/optimal auctions through deep learning, regret/DSIC/revenue/efficiency.
- Public Goods: free-rider/social optimum, sanctions/rewards, conditional cooperation, public-goods experiments, commons sustainability.
- Resource Island: Ostrom/common-pool governance, sequential social dilemmas, common-pool MARL, institution activation/monitoring.
- Labor Market: Gale-Shapley, deferred acceptance, stability, strategy-proofness for proposing side, matching-market design.

## Immediate Next Fixes

1. Add a relevance gate before card filling:
   - require at least one world anchor phrase in title or abstract,
   - penalize generic AI/ML survey papers unless query group explicitly asks for surveys,
   - reject ambiguous terms like `resource` unless paired with `common-pool`, `commons`, `property rights`, `trade`, or `social-ecological`.

2. Add a post-fill quarantine report:
   - flag cards with schema errors,
   - flag cards whose title/abstract lack world anchors,
   - flag cards containing project-world names in paper-specific fields.

3. Configure metadata identity:
   - add `OPENALEX_MAILTO`,
   - add `UNPAYWALL_EMAIL`,
   - keep keys only in `literature/secrets.env`.

4. Manually fetch priority PDFs only after filtering:
   - Pricing Arena is the top manual-PDF priority because it has only 7 extracted texts out of 96 records.
   - Auction House is already relatively strong.

## Bottom Line

The system works as a theory-discovery and obligation generator. It does not yet work as an unsupervised citation engine. The thesis-safe use is: use the generated obligations and gap tables now, then curate the actual cited papers from the clean anchor set plus manually reviewed paper cards.
