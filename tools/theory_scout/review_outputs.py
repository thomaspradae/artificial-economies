from __future__ import annotations

import csv
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .fill_paper_cards import (
    parse_markdown_sections,
    read_records,
    select_records_for_fill,
    source_context_for_record,
)
from .make_paper_cards import CARD_SECTIONS
from .query_config import iter_world_queries, load_queries


@dataclass
class CoverageRow:
    world: str
    query_group: str
    query_count: int
    api_records: int
    pdf_url_records: int
    extracted_text_records: int
    filled_cards: int
    todo_cards: int
    top_titles: str
    status: str


def google_scholar_url(query: str) -> str:
    return "https://scholar.google.com/scholar?" + urllib.parse.urlencode({"q": query})


def _record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("doi") or "").lower().replace("https://doi.org/", "").strip(),
        " ".join(str(record.get("title") or "").lower().split()),
        str(record.get("year") or ""),
    )


def _dedupe_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out = []
    for record in records:
        key = _record_key(record)
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def _authors(record: dict[str, Any]) -> str:
    authors = record.get("authors") or []
    if isinstance(authors, str):
        return authors
    return "; ".join(str(author) for author in authors)


def _has_text(record: dict[str, Any], text_dir: Path) -> bool:
    _, basis = source_context_for_record(record, text_dir=text_dir, max_chars=200)
    return basis.startswith("text:")


def _card_counts(cards_dir: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    if not cards_dir.exists():
        return counts
    for path in cards_dir.glob("*.md"):
        text = path.read_text(encoding="utf-8", errors="replace")
        world = ""
        for line in text.splitlines():
            if line.startswith("World:"):
                world = line.split(":", 1)[1].strip()
                break
        if not world:
            sections = parse_markdown_sections(text)
            world = sections.get("World", "").strip()
        if not world:
            continue
        sections = parse_markdown_sections(text)
        has_todo = any("TODO" in sections.get(section, "") for section in CARD_SECTIONS)
        bucket = counts.setdefault(world, {"filled": 0, "todo": 0})
        if has_todo:
            bucket["todo"] += 1
        else:
            bucket["filled"] += 1
    return counts


def build_theory_coverage_rows(
    *,
    queries_path: Path,
    records_path: Path,
    cards_dir: Path,
    text_dir: Path,
) -> list[CoverageRow]:
    config = load_queries(queries_path)
    records = _dedupe_records(read_records(records_path))
    card_counts = _card_counts(cards_dir)
    query_counts: dict[tuple[str, str], int] = {}
    for world, query_group, _query in iter_world_queries(config):
        query_counts[(world, query_group)] = query_counts.get((world, query_group), 0) + 1

    worlds = sorted(config.get("worlds", {}).keys())
    rows: list[CoverageRow] = []
    for world in worlds:
        world_records = [record for record in records if record.get("world") == world]
        rows.append(
            _coverage_row(
                world=world,
                query_group="all",
                query_count=sum(count for (w, _), count in query_counts.items() if w == world),
                records=world_records,
                cards=card_counts.get(world, {}),
                text_dir=text_dir,
            )
        )
        for group in ("classical_terms", "learning_terms"):
            group_records = [
                record
                for record in world_records
                if str(record.get("query_group") or "") == group
            ]
            rows.append(
                _coverage_row(
                    world=world,
                    query_group=group,
                    query_count=query_counts.get((world, group), 0),
                    records=group_records,
                    cards=card_counts.get(world, {}),
                    text_dir=text_dir,
                )
            )
    return rows


def _coverage_row(
    *,
    world: str,
    query_group: str,
    query_count: int,
    records: list[dict[str, Any]],
    cards: dict[str, int],
    text_dir: Path,
) -> CoverageRow:
    pdf_count = sum(1 for record in records if record.get("pdf_url"))
    text_count = sum(1 for record in records if _has_text(record, text_dir))
    top_titles = " | ".join(str(record.get("title") or "") for record in records[:5])
    filled = cards.get("filled", 0)
    todo = cards.get("todo", 0)
    if not records:
        status = "missing_metadata"
    elif query_group == "all" and filled == 0:
        status = "needs_filled_cards"
    elif pdf_count == 0 and text_count == 0:
        status = "metadata_only"
    else:
        status = "covered_review_needed"
    return CoverageRow(
        world=world,
        query_group=query_group,
        query_count=query_count,
        api_records=len(records),
        pdf_url_records=pdf_count,
        extracted_text_records=text_count,
        filled_cards=filled,
        todo_cards=todo,
        top_titles=top_titles,
        status=status,
    )


def build_manual_pdf_queue_rows(
    *,
    records_path: Path,
    text_dir: Path,
    worlds: set[str] | None = None,
    limit: int | None = None,
    per_world_limit: int | None = None,
) -> list[dict[str, Any]]:
    records = select_records_for_fill(
        read_records(records_path),
        worlds=worlds,
        limit=limit,
        per_world_limit=per_world_limit,
    )
    rows = []
    for record in records:
        title = str(record.get("title") or "")
        query = str(record.get("query") or title)
        rows.append(
            {
                "world": record.get("world") or "",
                "query_group": record.get("query_group") or "",
                "query": query,
                "title": title,
                "year": record.get("year") or "",
                "authors": _authors(record),
                "doi": record.get("doi") or "",
                "url": record.get("url") or "",
                "pdf_url": record.get("pdf_url") or "",
                "has_extracted_text": _has_text(record, text_dir),
                "google_scholar_title_url": google_scholar_url(title),
                "google_scholar_query_url": google_scholar_url(query),
                "manual_pdf_url": "",
                "manual_status": "",
                "notes": "",
            }
        )
    return rows


def build_scholar_comparison_rows(
    *,
    queries_path: Path,
    records_path: Path,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    config = load_queries(queries_path)
    records = _dedupe_records(read_records(records_path))
    rows = []
    for world, query_group, query in iter_world_queries(config):
        matches = [
            record
            for record in records
            if record.get("world") == world
            and record.get("query_group") == query_group
            and record.get("query") == query
        ][:top_n]
        rows.append(
            {
                "world": world,
                "query_group": query_group,
                "query": query,
                "google_scholar_url": google_scholar_url(query),
                "api_record_count": len(matches),
                "api_pdf_count": sum(1 for record in matches if record.get("pdf_url")),
                "api_top_titles": " | ".join(str(record.get("title") or "") for record in matches),
                "api_top_urls": " | ".join(str(record.get("url") or "") for record in matches),
                "scholar_top_titles_manual": "",
                "scholar_missing_from_api": "",
                "api_false_positives": "",
                "notes": "",
            }
        )
    return rows


def write_csv(rows: Iterable[dict[str, Any] | CoverageRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [asdict(row) if hasattr(row, "__dataclass_fields__") else dict(row) for row in rows]
    if materialized:
        fields = list(materialized[0].keys())
    else:
        fields = []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def write_coverage_markdown(rows: Iterable[CoverageRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Theory Coverage Report",
        "",
        "This is a coverage/review report, not a citation-quality ranking. It answers whether each world has metadata leads, PDF/text leads, and filled paper-card obligations.",
        "",
        "| World | Scope | Queries | API records | PDF links | Extracted text | Filled cards | TODO cards | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.world} | {row.query_group} | {row.query_count} | {row.api_records} | "
            f"{row.pdf_url_records} | {row.extracted_text_records} | {row.filled_cards} | "
            f"{row.todo_cards} | {row.status} |"
        )
    lines.extend(
        [
            "",
            "Manual checks should prioritize rows with `metadata_only`, low extracted-text counts, or obviously irrelevant top titles.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
