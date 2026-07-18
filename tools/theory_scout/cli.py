from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .build_gap_table import build_gap_rows, write_gap_table, write_theory_obligations
from .download_pdfs import download_pdf
from .extract_text import extract_pdf_text
from .http import read_jsonl, write_jsonl
from .make_paper_cards import make_blank_card
from .models import PaperRecord, record_key
from .query_config import iter_world_queries, load_queries
from .resolve_pdf import resolve_unpaywall_pdf
from .search_arxiv import search_arxiv
from .search_openalex import search_openalex
from .search_semantic_scholar import search_semantic_scholar
from .secrets import load_env_file, secret_presence


WORLD_KEYWORDS = {
    "pricing_arena": [
        "algorithmic",
        "collusion",
        "pricing",
        "price",
        "bertrand",
        "calvano",
        "q-learning",
        "reinforcement",
    ],
    "resource_island": [
        "common-pool",
        "commons",
        "resource",
        "resources",
        "ostrom",
        "property",
        "rights",
        "social-ecological",
        "sequential",
        "dilemma",
    ],
    "auction_house": [
        "auction",
        "auctions",
        "vickrey",
        "myerson",
        "bid",
        "bidding",
        "regret",
        "regretnet",
        "incentive",
    ],
    "public_goods": [
        "public",
        "goods",
        "free",
        "rider",
        "contribution",
        "punishment",
        "commons",
        "common-pool",
    ],
    "labor_market": [
        "matching",
        "match",
        "gale",
        "shapley",
        "deferred",
        "acceptance",
        "stable",
        "strategy-proof",
        "strategyproof",
    ],
}


ANCHOR_PHRASES = {
    "pricing_arena": ["algorithmic pricing", "algorithmic collusion", "calvano", "bertrand"],
    "resource_island": ["common-pool resource", "governing the commons", "ostrom", "sequential social dilemma"],
    "auction_house": ["auction", "vickrey", "myerson", "regretnet", "optimal auctions"],
    "public_goods": ["public goods", "free rider", "common-pool resource", "tragedy of the commons"],
    "labor_market": ["gale-shapley", "deferred acceptance", "stable matching", "matching markets"],
}


def dedupe(records: Iterable[PaperRecord]) -> list[PaperRecord]:
    seen = set()
    deduped: list[PaperRecord] = []
    for record in records:
        key = record_key(record)
        fallback_key = ("", key[1], key[2])
        effective_key = key if key[0] else fallback_key
        if effective_key in seen:
            continue
        seen.add(effective_key)
        deduped.append(record)
    return deduped


def rank_records(records: list[PaperRecord]) -> list[dict]:
    rows = [record.to_dict() for record in records]
    for row in rows:
        title = str(row.get("title") or "").lower()
        abstract = str(row.get("abstract") or "").lower()
        query = str(row.get("query") or "").lower()
        haystack = f"{title} {abstract}"
        query_tokens = {token for token in query.replace("-", " ").split() if len(token) > 3}
        world_terms = set(WORLD_KEYWORDS.get(row.get("world"), []))
        query_overlap = sum(1 for token in query_tokens if token in haystack)
        world_overlap = sum(1 for token in world_terms if token in haystack)
        anchor_bonus = sum(
            1 for phrase in ANCHOR_PHRASES.get(row.get("world"), []) if phrase in haystack
        )
        title_bonus = 2 * sum(1 for token in query_tokens | world_terms if token in title)
        citation_score = min(int(row.get("citation_count") or 0), 500) / 10
        pdf_score = 100 if row.get("pdf_url") else 0
        recency_score = max((int(row.get("year") or 1900) - 1990), 0)
        group_score = 50 if row.get("query_group") == "classical_terms" else 35
        row["has_pdf"] = bool(row.get("pdf_url"))
        row["relevance_score"] = 25 * query_overlap + 35 * world_overlap + 100 * anchor_bonus + title_bonus
        row["rank_score"] = row["relevance_score"] + citation_score + pdf_score + recency_score + group_score
    return sorted(rows, key=lambda row: (row["world"], -row["rank_score"], row["title"]))


def write_ranked_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "world",
        "query_group",
        "query",
        "source",
        "source_id",
        "title",
        "year",
        "authors",
        "doi",
        "url",
        "pdf_url",
        "citation_count",
        "has_pdf",
        "relevance_score",
        "rank_score",
        "abstract",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["authors"] = "; ".join(out.get("authors") or [])
            writer.writerow({field: out.get(field) for field in fields})


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "429" in message or "rate limit" in message or "retryafter" in message


def search_command(args: argparse.Namespace) -> None:
    if getattr(args, "env_file", None):
        load_env_file(Path(args.env_file))
    config = load_queries(Path(args.queries))
    records: list[PaperRecord] = []
    out_dir = Path(args.out_dir)
    raw_path = out_dir / "papers_raw.jsonl"
    if args.merge_existing and raw_path.exists():
        records.extend(PaperRecord(**row) for row in read_jsonl(raw_path))
    sources = set(args.sources)
    disabled_sources: set[str] = set()
    for world, query_group, query in iter_world_queries(config):
        print(f"[search] {world} / {query_group}: {query}")
        if "openalex" in sources and "openalex" not in disabled_sources:
            try:
                records.extend(search_openalex(query, world, query_group, per_page=args.per_query))
            except Exception as exc:
                print(f"[source failed] openalex: {query}: {exc}")
                if _is_rate_limit_error(exc):
                    disabled_sources.add("openalex")
                    print("[source disabled] openalex rate-limited; using existing cache/other sources")
        if "semantic_scholar" in sources and "semantic_scholar" not in disabled_sources:
            try:
                records.extend(search_semantic_scholar(query, world, query_group, limit=args.per_query))
            except Exception as exc:
                print(f"[source failed] semantic_scholar: {query}: {exc}")
                if _is_rate_limit_error(exc):
                    disabled_sources.add("semantic_scholar")
                    print("[source disabled] semantic_scholar rate-limited; using existing cache/other sources")
            if args.semantic_delay_seconds > 0:
                time.sleep(args.semantic_delay_seconds)
        if "arxiv" in sources and "arxiv" not in disabled_sources:
            try:
                records.extend(search_arxiv(query, world, query_group, max_results=args.per_query))
            except Exception as exc:
                print(f"[source failed] arxiv: {query}: {exc}")
                if _is_rate_limit_error(exc):
                    disabled_sources.add("arxiv")
                    print("[source disabled] arxiv rate-limited; using existing cache/other sources")

    records = dedupe(records)
    if args.resolve_pdfs:
        for record in records:
            if not record.pdf_url and record.doi:
                try:
                    record.pdf_url = resolve_unpaywall_pdf(record.doi)
                except Exception as exc:
                    print(f"[unpaywall failed] {record.doi}: {exc}")

    ranked_path = out_dir / "papers_ranked.csv"
    rows = rank_records(records)
    write_jsonl(raw_path, [record.to_dict() for record in records])
    write_ranked_csv(rows, ranked_path)
    print(f"[wrote] {raw_path} ({len(records)} records)")
    print(f"[wrote] {ranked_path}")

    if args.download:
        for row in rows:
            if not row.get("pdf_url"):
                continue
            pdf_path = download_pdf(
                row["pdf_url"],
                title=row["title"],
                year=row.get("year"),
                out_dir=out_dir / "pdfs",
            )
            if pdf_path:
                extract_pdf_text(pdf_path, out_dir / "text")

    if args.make_cards:
        limit = args.card_limit
        made = 0
        for row in rows:
            if limit is not None and made >= limit:
                break
            make_blank_card(row, out_dir / "paper_cards")
            made += 1
        print(f"[cards] wrote or reused {made} cards")


def _count_raw(path: Path) -> int:
    return sum(1 for _ in path.open("r", encoding="utf-8")) if path.exists() else 0


def cards_command(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.raw))
    ranked = sorted(
        rows,
        key=lambda row: (
            row.get("world") or "",
            -int(row.get("citation_count") or 0),
            row.get("title") or "",
        ),
    )
    count = 0
    for row in ranked:
        if args.limit is not None and count >= args.limit:
            break
        make_blank_card(row, Path(args.out_dir) / "paper_cards")
        count += 1
    print(f"[cards] wrote or reused {count} cards")


def obligations_command(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    rows = build_gap_rows(Path(args.queries), out_dir / "papers_raw.jsonl")
    write_gap_table(rows, out_dir / "novelty_gap_table.csv")
    write_theory_obligations(rows, out_dir / "theory_obligations.md")
    print(f"[wrote] {out_dir / 'novelty_gap_table.csv'} ({len(rows)} rows)")
    print(f"[wrote] {out_dir / 'theory_obligations.md'}")


def rerank_command(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.raw))
    records = [PaperRecord(**row) for row in rows]
    ranked = rank_records(records)
    write_ranked_csv(ranked, Path(args.out))
    print(f"[wrote] {args.out} ({len(ranked)} rows)")


def full_command(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_loaded = load_env_file(Path(args.env_file))
    started_at = datetime.now(timezone.utc)
    manifest = {
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "queries": args.queries,
        "out_dir": args.out_dir,
        "per_query": args.per_query,
        "semantic_delay_seconds": args.semantic_delay_seconds,
        "sources_requested": [],
        "sources_run": [],
        "secrets_loaded": sorted(env_loaded.keys()),
        "secret_presence": secret_presence(),
        "outputs": {
            "raw": str(out_dir / "papers_raw.jsonl"),
            "ranked": str(out_dir / "papers_ranked.csv"),
            "cards": str(out_dir / "paper_cards"),
            "gap_table": str(out_dir / "novelty_gap_table.csv"),
            "obligations": str(out_dir / "theory_obligations.md"),
        },
    }

    sources = ["openalex"]
    if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
        sources.append("semantic_scholar")
    elif args.require_semantic:
        raise SystemExit(
            "SEMANTIC_SCHOLAR_API_KEY is required for --require-semantic. "
            "Put it in literature/secrets.env or export it in the shell."
        )
    if args.include_arxiv:
        sources.append("arxiv")
    manifest["sources_requested"] = sources

    search_args = argparse.Namespace(
        queries=args.queries,
        out_dir=args.out_dir,
        per_query=args.per_query,
        sources=sources,
        resolve_pdfs=args.resolve_pdfs,
        download=args.download,
        make_cards=True,
        card_limit=args.card_limit,
        semantic_delay_seconds=args.semantic_delay_seconds,
        merge_existing=True,
        env_file=None,
    )
    search_command(search_args)
    manifest["sources_run"] = sources
    manifest["record_count"] = _count_raw(out_dir / "papers_raw.jsonl")

    rerank_command(
        argparse.Namespace(
            raw=str(out_dir / "papers_raw.jsonl"),
            out=str(out_dir / "papers_ranked.csv"),
        )
    )
    obligations_command(argparse.Namespace(queries=args.queries, out_dir=args.out_dir))

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["duration_seconds"] = (
        datetime.fromisoformat(manifest["finished_at"]) - started_at
    ).total_seconds()
    manifest_path = out_dir / "scout_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[wrote] {manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cached scholarly metadata scout")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search", help="query metadata APIs and write raw/ranked caches")
    search.add_argument("--queries", default="literature/queries.yaml")
    search.add_argument("--out-dir", default="literature")
    search.add_argument("--per-query", type=int, default=10)
    search.add_argument(
        "--sources",
        nargs="+",
        default=["openalex", "semantic_scholar", "arxiv"],
        choices=["openalex", "semantic_scholar", "arxiv"],
    )
    search.add_argument("--resolve-pdfs", action="store_true")
    search.add_argument("--download", action="store_true")
    search.add_argument("--make-cards", action="store_true")
    search.add_argument("--card-limit", type=int, default=50)
    search.add_argument(
        "--semantic-delay-seconds",
        type=float,
        default=1.1,
        help="Delay after every Semantic Scholar request; S2 keys are commonly 1 request/sec.",
    )
    search.add_argument(
        "--replace-cache",
        dest="merge_existing",
        action="store_false",
        help="Do not merge with an existing papers_raw.jsonl cache.",
    )
    search.add_argument("--env-file", default=None, help="Optional ignored secrets env file.")
    search.set_defaults(merge_existing=True)
    search.set_defaults(func=search_command)

    cards = subparsers.add_parser("cards", help="make strict paper-card templates from raw cache")
    cards.add_argument("--raw", default="literature/papers_raw.jsonl")
    cards.add_argument("--out-dir", default="literature")
    cards.add_argument("--limit", type=int, default=50)
    cards.set_defaults(func=cards_command)

    obligations = subparsers.add_parser("obligations", help="write obligation markdown and gap table")
    obligations.add_argument("--queries", default="literature/queries.yaml")
    obligations.add_argument("--out-dir", default="literature")
    obligations.set_defaults(func=obligations_command)

    rerank = subparsers.add_parser("rerank", help="rebuild ranked CSV from cached raw JSONL")
    rerank.add_argument("--raw", default="literature/papers_raw.jsonl")
    rerank.add_argument("--out", default="literature/papers_ranked.csv")
    rerank.set_defaults(func=rerank_command)

    full = subparsers.add_parser("full", help="run the overnight metadata scout pipeline")
    full.add_argument("--queries", default="literature/queries.yaml")
    full.add_argument("--out-dir", default="literature")
    full.add_argument("--env-file", default="literature/secrets.env")
    full.add_argument("--per-query", type=int, default=5)
    full.add_argument("--semantic-delay-seconds", type=float, default=1.1)
    full.add_argument("--card-limit", type=int, default=150)
    full.add_argument("--include-arxiv", action="store_true")
    full.add_argument("--resolve-pdfs", action="store_true")
    full.add_argument("--download", action="store_true")
    full.add_argument("--require-semantic", action="store_true")
    full.set_defaults(func=full_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
