from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .http import read_jsonl
from .make_paper_cards import CARD_SECTIONS, make_blank_card, slugify
from .ollama_client import DEFAULT_OLLAMA_URL, OllamaClient, extract_json_object


FILLED_EXTRA_SECTIONS = [
    "Extraction evidence",
    "Extraction metadata",
]

MODEL_KEYS = {
    "Paper": "paper",
    "World": "world",
    "Institution": "institution",
    "Agent type": "agent_type",
    "Theoretical benchmark": "theoretical_benchmark",
    "Learning setup": "learning_setup",
    "Metrics": "metrics",
    "Main result": "main_result",
    "What they prove": "what_they_prove",
    "What they only simulate": "what_they_only_simulate",
    "What they do NOT test": "what_they_do_not_test",
    "What we need to reproduce": "what_we_need_to_reproduce",
    "How our project differs": "how_our_project_differs",
}

REQUIRED_MODEL_KEYS = list(MODEL_KEYS.values()) + [
    "source_evidence",
    "confidence",
]

NOT_STATED = "Not stated in supplied text."


@dataclass
class FilledCardResult:
    card_path: Path
    title: str
    world: str
    source_basis: str
    model: str
    output_tokens_per_second: float | None
    changed: bool
    validation_errors: list[str]


def parse_markdown_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def _stringify(value: Any) -> str:
    if value is None:
        return NOT_STATED
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip()) or NOT_STATED
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value).strip()
    return text or NOT_STATED


def _validate_fields(fields: dict[str, Any]) -> list[str]:
    errors = []
    for key in REQUIRED_MODEL_KEYS:
        if key not in fields:
            errors.append(f"missing key: {key}")
        elif not _stringify(fields[key]):
            errors.append(f"blank key: {key}")
    return errors


def _evidence_excerpt(source_context: str, max_chars: int = 700) -> str:
    excerpt = " ".join(source_context.split())
    if len(excerpt) > max_chars:
        excerpt = excerpt[: max_chars - 3].rstrip() + "..."
    return excerpt or NOT_STATED


def _record_card_path(record: dict[str, Any], cards_dir: Path) -> Path:
    title = record.get("title") or "Untitled"
    year = record.get("year") or "unknown"
    return cards_dir / f"{year}_{slugify(title)}.md"


def _text_candidates(record: dict[str, Any], text_dir: Path) -> list[Path]:
    title = record.get("title") or "Untitled"
    year = record.get("year") or "unknown"
    stem = f"{year}_{slugify(title)}"
    candidates = [
        text_dir / f"{stem}.txt",
        text_dir / f"{slugify(title)}.txt",
    ]
    source_id = str(record.get("source_id") or "")
    if source_id:
        candidates.append(text_dir / f"{slugify(source_id)}.txt")
    return candidates


def source_context_for_record(
    record: dict[str, Any],
    *,
    text_dir: Path,
    max_chars: int = 7000,
) -> tuple[str, str]:
    for path in _text_candidates(record, text_dir):
        if path.exists() and path.stat().st_size > 200:
            text = path.read_text(encoding="utf-8", errors="replace")
            return text[:max_chars], f"text:{path}"
    abstract = str(record.get("abstract") or "").strip()
    if abstract:
        return abstract[:max_chars], "abstract"
    metadata = {
        "title": record.get("title"),
        "year": record.get("year"),
        "authors": record.get("authors"),
        "doi": record.get("doi"),
        "url": record.get("url"),
        "source_id": record.get("source_id"),
        "world": record.get("world"),
        "query": record.get("query"),
    }
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True), "metadata_only"


def build_extraction_prompt(record: dict[str, Any], source_context: str, source_basis: str) -> tuple[str, str]:
    system = (
        "You are a conservative literature extraction engine. "
        "Use only the supplied metadata/text. Do not use outside knowledge. "
        "If a field is not stated, write exactly 'Not stated in supplied text.' "
        "Return only valid compact JSON. No markdown and no prose."
    )
    user = {
        "task": "Fill a strict paper card for an economics/learning-agents thesis.",
        "required_json_keys": REQUIRED_MODEL_KEYS,
        "paper_metadata": {
            "title": record.get("title"),
            "year": record.get("year"),
            "authors": record.get("authors"),
            "doi": record.get("doi"),
            "url": record.get("url"),
            "pdf_url": record.get("pdf_url"),
            "source": record.get("source"),
            "source_id": record.get("source_id"),
            "world": record.get("world"),
            "query": record.get("query"),
            "query_group": record.get("query_group"),
        },
        "project_context": (
            "This repo compares learning-agent behavior across Pricing Arena, Resource Island, "
            "Auction House, Public Goods, and Labor Market worlds. The card should identify "
            "the classical benchmark, prior learning setup, metrics, and what this project must reproduce."
        ),
        "source_basis": source_basis,
        "source_text": source_context,
    }
    return system, json.dumps(user, ensure_ascii=False, sort_keys=True)


def render_filled_card(
    record: dict[str, Any],
    fields: dict[str, Any],
    *,
    source_basis: str,
    model: str,
    output_tokens_per_second: float | None,
) -> str:
    title = record.get("title") or fields.get("paper") or "Untitled"
    lines = [
        f"# {title}",
        "",
        "<!-- Generated by tools.theory_scout. Verify claims before citing. -->",
        "",
        f"Source: {record.get('source')}",
        f"Source ID: {record.get('source_id')}",
        f"Year: {record.get('year')}",
        f"Authors: {', '.join(record.get('authors') or [])}",
        f"DOI: {record.get('doi')}",
        f"URL: {record.get('url')}",
        f"PDF: {record.get('pdf_url')}",
        f"World: {record.get('world')}",
        f"Query: {record.get('query')}",
        "",
    ]
    for section in CARD_SECTIONS:
        key = MODEL_KEYS[section]
        value = fields.get(key)
        if section == "Paper":
            value = title
        if section == "World" and not value:
            value = record.get("world")
        lines.extend([f"## {section}", "", _stringify(value), ""])

    lines.extend(
        [
            "## Extraction evidence",
            "",
            _stringify(fields.get("source_evidence")),
            "",
            "## Extraction metadata",
            "",
            f"Model: {model}",
            f"Source basis: {source_basis}",
            f"Confidence: {_stringify(fields.get('confidence'))}",
            "Output tok/s: "
            + (
                f"{output_tokens_per_second:.2f}"
                if output_tokens_per_second is not None
                else "unknown"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def fill_card_for_record(
    record: dict[str, Any],
    *,
    cards_dir: Path,
    text_dir: Path,
    client: OllamaClient,
    model: str,
    force: bool = False,
    dry_run: bool = False,
    num_predict: int = 900,
    num_ctx: int = 4096,
    num_thread: int | None = None,
) -> FilledCardResult:
    cards_dir.mkdir(parents=True, exist_ok=True)
    card_path = _record_card_path(record, cards_dir)
    if not card_path.exists():
        make_blank_card(record, cards_dir)
    existing = card_path.read_text(encoding="utf-8") if card_path.exists() else ""
    existing_sections = parse_markdown_sections(existing)
    has_todo = any("TODO" in existing_sections.get(section, "") for section in CARD_SECTIONS)
    if existing and not force and not has_todo:
        return FilledCardResult(
            card_path=card_path,
            title=str(record.get("title") or "Untitled"),
            world=str(record.get("world") or ""),
            source_basis="skipped_existing_filled",
            model=model,
            output_tokens_per_second=None,
            changed=False,
            validation_errors=[],
        )

    context, source_basis = source_context_for_record(record, text_dir=text_dir)
    system, user = build_extraction_prompt(record, context, source_basis)
    try:
        response = client.chat(
            model=model,
            system=system,
            user=user,
            temperature=0.0,
            num_predict=num_predict,
            num_ctx=num_ctx,
            num_thread=num_thread,
            think=False,
            keep_alive=0,
            format_json=True,
        )
        fields = extract_json_object(response.content)
    except Exception as exc:
        return FilledCardResult(
            card_path=card_path,
            title=str(record.get("title") or "Untitled"),
            world=str(record.get("world") or ""),
            source_basis=source_basis,
            model=model,
            output_tokens_per_second=None,
            changed=False,
            validation_errors=[f"llm extraction failed: {exc}"],
        )
    evidence = _stringify(fields.get("source_evidence")).strip().lower()
    if evidence in {"abstract", "source text", "metadata", NOT_STATED.lower()} or len(evidence) < 40:
        fields["source_evidence"] = _evidence_excerpt(context)
    confidence = _stringify(fields.get("confidence")).strip()
    if confidence == NOT_STATED:
        fields["confidence"] = "low"
    errors = _validate_fields(fields)
    rendered = render_filled_card(
        record,
        fields,
        source_basis=source_basis,
        model=model,
        output_tokens_per_second=response.output_tokens_per_second,
    )
    if not dry_run and not errors:
        card_path.write_text(rendered, encoding="utf-8")
    return FilledCardResult(
        card_path=card_path,
        title=str(record.get("title") or "Untitled"),
        world=str(record.get("world") or ""),
        source_basis=source_basis,
        model=model,
        output_tokens_per_second=response.output_tokens_per_second,
        changed=not dry_run and not errors,
        validation_errors=errors,
    )


def select_records_for_fill(
    records: Iterable[dict[str, Any]],
    *,
    worlds: set[str] | None = None,
    title_contains: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    selected = []
    seen = set()
    title_filter = title_contains.lower() if title_contains else None
    for record in records:
        world = str(record.get("world") or "")
        if worlds and world not in worlds:
            continue
        title = str(record.get("title") or "")
        if title_filter and title_filter not in title.lower():
            continue
        key = (
            str(record.get("doi") or "").lower(),
            title.lower(),
            record.get("year"),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(record)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if isinstance(row.get("authors"), str):
                row["authors"] = [
                    author.strip() for author in str(row["authors"]).split(";") if author.strip()
                ]
            for int_field in ("year", "citation_count"):
                if row.get(int_field) in ("", None):
                    row[int_field] = None
                else:
                    try:
                        row[int_field] = int(float(str(row[int_field])))
                    except ValueError:
                        row[int_field] = None
        return rows
    return read_jsonl(path)


def fill_cards(
    *,
    raw_path: Path,
    cards_dir: Path,
    text_dir: Path,
    model: str = "llama3.2:3b",
    ollama_url: str = DEFAULT_OLLAMA_URL,
    worlds: set[str] | None = None,
    title_contains: str | None = None,
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    num_predict: int = 900,
    num_ctx: int = 4096,
    num_thread: int | None = None,
    client: OllamaClient | None = None,
) -> list[FilledCardResult]:
    records = select_records_for_fill(
        read_records(raw_path),
        worlds=worlds,
        title_contains=title_contains,
        limit=limit,
    )
    active_client = client or OllamaClient(base_url=ollama_url)
    results = []
    for record in records:
        result = fill_card_for_record(
            record,
            cards_dir=cards_dir,
            text_dir=text_dir,
            client=active_client,
            model=model,
            force=force,
            dry_run=dry_run,
            num_predict=num_predict,
            num_ctx=num_ctx,
            num_thread=num_thread,
        )
        results.append(result)
    return results
