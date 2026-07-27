# Google Scholar Comparison Protocol

Google Scholar does not provide a stable official API for this workflow. Do not build thesis-critical automation on direct Scholar scraping: it is brittle, captcha-prone, and hard to reproduce. The reproducible path is a manual Scholar top-result check plus deterministic comparison.

## Files

- `literature/scholar_comparison_worksheet.csv`: one row per configured query with API top titles and a Google Scholar URL.
- `literature/scholar_comparison_report.csv`: generated overlap report after manual Scholar titles are entered.
- `literature/scholar_comparison_report.md`: readable summary of the comparison.

## Manual Step

For each important query:

1. Open `google_scholar_url`.
2. Copy the first 5-10 real paper titles from Google Scholar.
3. Paste them into `scholar_top_titles_manual`, separated by ` | `.
4. Ignore pure citation rows, patents, and clearly unrelated non-paper results.

Then run:

```bash
.venv/bin/python -m tools.theory_scout.cli scholar-compare
```

## Interpretation

- `strong_overlap`: metadata APIs are close enough to Scholar for this query.
- `partial_overlap`: review manually; either Scholar has important missing papers or the API ranking includes false positives.
- `weak_overlap`: do not trust the API ranking for this query until queries/ranking are fixed.
- `needs_manual_scholar_top_titles`: no manual Scholar titles have been pasted yet.

## Query Tuning

Use Scholar to test query quality, not as an unreviewed citation source.

Good Scholar queries are narrow and theory-anchored:

- `Calvano algorithmic pricing collusion Q-learning`
- `RegretNet auction design regret incentive compatibility`
- `Gale Shapley deferred acceptance strategy proof stable matching`
- `public goods punishment rewards contribution free rider`
- `Ostrom common pool resource property rights sanctions`

Bad queries are broad and invite false positives:

- `resource reinforcement learning`
- `multi agent learning matching`
- `public goods artificial intelligence`
- `deep learning resource allocation`

If a query produces broad AI/ML surveys, add mechanism terms such as `auction`, `public goods`, `common-pool`, `deferred acceptance`, `Bertrand`, `collusion`, `property rights`, or `free rider`.

## Optional Paid Automation

If later needed, use a third-party Google Scholar results API such as SerpAPI rather than scraping Scholar HTML directly. Keep any key in `literature/secrets.env`, and treat those results as metadata leads, not citation authority.
