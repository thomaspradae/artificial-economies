# Literature Theory Scout

This directory is the reproducible theory layer for the artificial economies paper.
It is not a PDF-scraping bot. It searches legitimate metadata APIs, caches the
results, creates strict paper-card templates, and generates obligation tables.

## Files

- `queries.yaml`: world-specific classical and learning search terms.
- `papers_raw.jsonl`: cached metadata records from scholarly APIs.
- `papers_ranked.csv`: relevance-ranked metadata view.
- `paper_cards/`: strict extraction templates for individual papers.
- `novelty_gap_table.csv`: `world | institution | mind | closest paper | theory benchmark | their metric | our metric | gap`.
- `theory_obligations.md`: world-level theory obligations.

## One-Command Pipeline

Overnight run:

```bash
scripts/run_theory_scout_full.sh
```

Strict overnight run that fails if Semantic Scholar is not configured:

```bash
scripts/run_theory_scout_full.sh --require-semantic
```

The wrapper loads `literature/secrets.env` if present, runs the metadata scout,
merges with the existing cache, rebuilds `papers_ranked.csv`, creates/updates
paper-card templates, writes `novelty_gap_table.csv`, writes
`theory_obligations.md`, and records `scout_manifest.json`.

Equivalent direct Python command:

```bash
python -m tools.theory_scout.cli full --env-file literature/secrets.env --per-query 5
```

Optional arXiv enrichment can be enabled, but arXiv may rate-limit or time out
on broad query sets:

```bash
python -m tools.theory_scout.cli full --include-arxiv
```

Optional legal PDF resolution/download:

```bash
python -m tools.theory_scout.cli full --resolve-pdfs --download
```

Use PDF download sparingly. Metadata and obligations are the main thesis-safety
layer; paper cards still need manual extraction before claims are final.

## Secrets

Create a local ignored secrets file:

```bash
scripts/configure_theory_scout_secrets.sh
```

Or create it manually:

```bash
cp literature/secrets.env.example literature/secrets.env
chmod 600 literature/secrets.env
```

Then fill in `literature/secrets.env`:

```bash
export OPENALEX_API_KEY="..."
export OPENALEX_MAILTO="you@example.com"
export SEMANTIC_SCHOLAR_API_KEY="..."
export UNPAYWALL_EMAIL="you@example.com"
```

`literature/secrets.env` is ignored by git. The run manifest records only which
secrets were present, never the secret values.

If a source is exhausted or rate-limited, the pipeline disables that source for
the rest of the run and continues from the cache plus remaining sources.

## Component Commands

Generate a first OpenAlex-backed cache:

```bash
python -m tools.theory_scout.cli search --per-query 5 --sources openalex --make-cards --card-limit 100
```

Enrich the existing cache with Semantic Scholar. This merges with
`papers_raw.jsonl` instead of replacing it and waits 1.1 seconds after each S2
request:

```bash
export SEMANTIC_SCHOLAR_API_KEY="..."
python -m tools.theory_scout.cli search --per-query 3 --sources semantic_scholar --semantic-delay-seconds 1.1 --make-cards --card-limit 120
python -m tools.theory_scout.cli obligations
```

Alternatively, use the ignored secrets file:

```bash
source literature/secrets.env
python -m tools.theory_scout.cli search --per-query 3 --sources semantic_scholar --semantic-delay-seconds 1.1 --make-cards --card-limit 120
```

`literature/secrets.env` is ignored by git.

Rebuild ranked CSV without network:

```bash
python -m tools.theory_scout.cli rerank
```

Rebuild the gap table and theory obligations without network:

```bash
python -m tools.theory_scout.cli obligations
```

API configuration:

- `OPENALEX_API_KEY`: optional OpenAlex key.
- `OPENALEX_MAILTO`: optional OpenAlex polite-pool email.
- `SEMANTIC_SCHOLAR_API_KEY`: optional Semantic Scholar key.
- `UNPAYWALL_EMAIL`: required only for Unpaywall DOI-to-PDF resolution.

Do not commit API keys.

## Current Status

The first cache was populated from OpenAlex. Semantic Scholar support is wired
with a default delay of 1.1 seconds between requests, matching the approved
1-request/second key limit. arXiv support exists but is optional because broad
query runs can time out or rate-limit. The generated paper cards are templates;
a card is thesis-ready only after its TODO fields are filled from the paper
text.
