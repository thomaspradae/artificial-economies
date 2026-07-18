# Literature Theory Scout

This directory is the reproducible theory layer for the artificial economies paper.
It is not a PDF-scraping bot. It searches legitimate metadata APIs, caches the
results, creates strict paper-card templates, and generates obligation tables.

## Files

- `queries.yaml`: world-specific classical and learning search terms.
- `papers_raw.jsonl`: cached metadata records from scholarly APIs.
- `papers_ranked.csv`: relevance-ranked metadata view.
- `paper_cards/`: strict extraction templates for individual papers.
- `pdf_text_report.csv`: per-paper PDF discovery/download/text-extraction status.
- `card_fill_manifest.json`: per-card local-LLM fill status.
- `theory_coverage.csv` / `theory_coverage.md`: per-world coverage report for metadata leads, PDF/text availability, and filled paper cards.
- `manual_pdf_queue.csv`: balanced per-world link list for manually finding PDFs and recording status.
- `scholar_comparison_worksheet.csv`: Google Scholar query worksheet comparing API top titles against manual Scholar checks.
- `novelty_gap_table.csv`: `world | institution | mind | closest paper | theory benchmark | their metric | our metric | gap`.
- `theory_obligations.md`: world-level theory obligations.
- `obligation_audit.md`: deterministic check that theory obligations have code/result evidence.

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

Full overnight run with PDF hydration and `ofi1` local-LLM card filling:

```bash
scripts/run_theory_scout_overnight_ofi1.sh
```

This opens an SSH tunnel to `ofi1`'s Ollama server, runs metadata search,
creates paper-card templates, resolves/downloads open-access PDFs from metadata
or Unpaywall, extracts canonical text into `literature/text/`, fills a balanced
per-world set of strict paper cards with `llama3.2:3b`, rebuilds obligation
tables, writes the deterministic obligation audit, and emits manual review
worksheets. It is the default “leave it running at night” pipeline.

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

With `full`, `--download` now runs the reporting hydration stage and writes
`literature/pdf_text_report.csv`. Use PDF download sparingly. The pipeline only
uses PDF URLs returned by scholarly metadata APIs or Unpaywall; it is not a
general web scraper.

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

Hydrate cached PDFs/text from the ranked metadata cache without rerunning search:

```bash
python -m tools.theory_scout.cli hydrate-text --limit 25 --resolve-pdfs
```

This writes `literature/pdf_text_report.csv`. Status rows make the failure mode
explicit: existing text, no PDF URL, download failure, extraction failure, or
extracted text size.

Fill strict paper cards using the local Ollama extractor. The recommended
default from the `ofi1` benchmark is `llama3.2:3b`:

```bash
python -m tools.theory_scout.cli fill-cards --limit 10 --model llama3.2:3b
```

For thesis coverage, prefer a balanced per-world fill instead of only the
globally highest-ranked papers:

```bash
python -m tools.theory_scout.cli fill-cards --per-world-limit 8 --model llama3.2:3b
```

To use `ofi1`'s user-local Ollama server from this local checkout, use the SSH
tunnel helper:

```bash
scripts/run_theory_llm_fill_ofi1.sh --limit 10
```

By default, `fill-cards` reads `literature/papers_ranked.csv`, so it fills
high-scoring papers first. You can target a canonical paper explicitly:

```bash
scripts/run_theory_llm_fill_ofi1.sh --world auction_house --title-contains "optimal auctions through deep learning" --limit 1
```

The filler uses cached PDF text from `literature/text/` when available and
falls back to metadata abstracts. It writes `literature/card_fill_manifest.json`
and rewrites only cards that still contain TODOs unless `--force` is passed.
The model is instructed to write `Not stated in supplied text.` when the
provided source does not support a field.

Audit whether theory obligations are represented in code/results:

```bash
python -m tools.theory_scout.cli audit-obligations
```

This writes:

- `literature/obligation_audit.csv`
- `literature/obligation_audit.md`
- `literature/theory_gap_report.csv`

The audit is deterministic. It checks for expected world files, benchmark
helpers, tests/output files, and result columns/terms. Filled paper-card fields
are added as review rows; unfilled TODO templates are skipped rather than
counted as failures. It does not replace human review of filled paper cards.

Generate coverage-first review files without network access:

```bash
python -m tools.theory_scout.cli review --per-world-limit 30
```

This writes:

- `literature/theory_coverage.csv` and `.md`: which worlds have metadata,
  PDF/text, and filled-card coverage.
- `literature/manual_pdf_queue.csv`: balanced link list with paper URLs, PDF
  URLs, and Google Scholar title/query links for manual PDF discovery.
- `literature/scholar_comparison_worksheet.csv`: one row per configured query,
  with the API top titles next to a Google Scholar search URL and blank columns
  for manual Scholar results. This is how to compare OpenAlex/Semantic
  Scholar/arXiv coverage against Scholar without scraping Google Scholar.

If you have a SerpAPI key and want an automated Google Scholar snapshot for
review only, `scripts/serpapi_scholar_lit.py` exists, but it is intentionally
separate from the reproducible pipeline because Google Scholar metadata is
noisy and terms of use differ from OpenAlex/Semantic Scholar/arXiv.

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
query runs can time out or rate-limit. The generated paper cards can now be
filled by a local LLM, but a card is thesis-ready only after its extracted
claims are checked against the cited source text.
