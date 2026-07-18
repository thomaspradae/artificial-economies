from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .download_pdfs import download_pdf
from .extract_text import extract_pdf_text
from .fill_paper_cards import read_records, select_records_for_fill
from .make_paper_cards import slugify
from .resolve_pdf import resolve_unpaywall_pdf


DownloadFunc = Callable[[str, str, int | None, Path], Path | None]
ExtractFunc = Callable[[Path, Path, Path | None], Path | None]
ResolveFunc = Callable[[str | None], str | None]


@dataclass
class HydrationRow:
    world: str
    title: str
    year: str
    source: str
    source_id: str
    doi: str
    original_pdf_url: str
    resolved_pdf_url: str
    pdf_path: str
    text_path: str
    pdf_status: str
    text_status: str
    text_chars: int
    error: str


def canonical_text_path(record: dict[str, Any], text_dir: Path) -> Path:
    title = str(record.get("title") or "Untitled")
    year = record.get("year") or "unknown"
    return text_dir / f"{year}_{slugify(title)}.txt"


def _existing_text_path(record: dict[str, Any], text_dir: Path, min_text_chars: int) -> Path | None:
    title = str(record.get("title") or "Untitled")
    year = record.get("year") or "unknown"
    slug = slugify(title)
    candidates = [
        canonical_text_path(record, text_dir),
        text_dir / f"{slug}.txt",
    ]
    candidates.extend(sorted(text_dir.glob(f"{year}_{slug}*.txt")))
    for path in candidates:
        if path.exists() and path.stat().st_size >= min_text_chars:
            return path
    return None


def hydrate_record(
    record: dict[str, Any],
    *,
    pdf_dir: Path,
    text_dir: Path,
    resolve_pdfs: bool = False,
    min_text_chars: int = 1000,
    download_func: DownloadFunc = download_pdf,
    extract_func: ExtractFunc = extract_pdf_text,
    resolve_func: ResolveFunc = resolve_unpaywall_pdf,
) -> HydrationRow:
    existing = _existing_text_path(record, text_dir, min_text_chars)
    original_pdf_url = str(record.get("pdf_url") or "")
    resolved_pdf_url = original_pdf_url
    if existing is not None:
        text = existing.read_text(encoding="utf-8", errors="replace")
        return HydrationRow(
            world=str(record.get("world") or ""),
            title=str(record.get("title") or "Untitled"),
            year=str(record.get("year") or ""),
            source=str(record.get("source") or ""),
            source_id=str(record.get("source_id") or ""),
            doi=str(record.get("doi") or ""),
            original_pdf_url=original_pdf_url,
            resolved_pdf_url=resolved_pdf_url,
            pdf_path="",
            text_path=str(existing),
            pdf_status="skipped_existing_text",
            text_status="present",
            text_chars=len(text),
            error="",
        )

    if not resolved_pdf_url and resolve_pdfs:
        try:
            resolved_pdf_url = resolve_func(str(record.get("doi") or "") or None) or ""
        except Exception as exc:
            return _error_row(record, original_pdf_url, "", "resolve_failed", "missing", str(exc))

    if not resolved_pdf_url:
        return _error_row(record, original_pdf_url, "", "no_pdf_url", "missing", "no PDF URL in metadata")

    try:
        pdf_path = download_func(
            resolved_pdf_url,
            str(record.get("title") or "Untitled"),
            _int_year(record.get("year")),
            pdf_dir,
        )
    except Exception as exc:
        return _error_row(record, original_pdf_url, resolved_pdf_url, "download_failed", "missing", str(exc))
    if pdf_path is None:
        return _error_row(
            record,
            original_pdf_url,
            resolved_pdf_url,
            "download_failed",
            "missing",
            "download returned no PDF",
        )

    out_path = canonical_text_path(record, text_dir)
    try:
        text_path = extract_func(pdf_path, text_dir, out_path)
    except Exception as exc:
        return _error_row(
            record,
            original_pdf_url,
            resolved_pdf_url,
            "downloaded",
            "extract_failed",
            str(exc),
            pdf_path=pdf_path,
        )
    if text_path is None or not text_path.exists():
        return _error_row(
            record,
            original_pdf_url,
            resolved_pdf_url,
            "downloaded",
            "extract_failed",
            "extractor returned no text",
            pdf_path=pdf_path,
        )
    text = text_path.read_text(encoding="utf-8", errors="replace")
    status = "extracted" if len(text) >= min_text_chars else "too_short"
    return HydrationRow(
        world=str(record.get("world") or ""),
        title=str(record.get("title") or "Untitled"),
        year=str(record.get("year") or ""),
        source=str(record.get("source") or ""),
        source_id=str(record.get("source_id") or ""),
        doi=str(record.get("doi") or ""),
        original_pdf_url=original_pdf_url,
        resolved_pdf_url=resolved_pdf_url,
        pdf_path=str(pdf_path),
        text_path=str(text_path),
        pdf_status="downloaded",
        text_status=status,
        text_chars=len(text),
        error="" if status == "extracted" else f"text shorter than {min_text_chars} chars",
    )


def hydrate_texts(
    *,
    records_path: Path,
    pdf_dir: Path,
    text_dir: Path,
    report_path: Path,
    worlds: set[str] | None = None,
    title_contains: str | None = None,
    limit: int | None = None,
    per_world_limit: int | None = None,
    resolve_pdfs: bool = False,
    min_text_chars: int = 1000,
    download_func: DownloadFunc = download_pdf,
    extract_func: ExtractFunc = extract_pdf_text,
    resolve_func: ResolveFunc = resolve_unpaywall_pdf,
) -> list[HydrationRow]:
    records = select_records_for_fill(
        read_records(records_path),
        worlds=worlds,
        title_contains=title_contains,
        limit=limit,
        per_world_limit=per_world_limit,
    )
    rows = [
        hydrate_record(
            record,
            pdf_dir=pdf_dir,
            text_dir=text_dir,
            resolve_pdfs=resolve_pdfs,
            min_text_chars=min_text_chars,
            download_func=download_func,
            extract_func=extract_func,
            resolve_func=resolve_func,
        )
        for record in records
    ]
    write_hydration_report(rows, report_path)
    return rows


def write_hydration_report(rows: Iterable[HydrationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = list(asdict(rows[0]).keys()) if rows else list(HydrationRow("", "", "", "", "", "", "", "", "", "", "", "", 0, "").__dict__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def hydration_summary(rows: Iterable[HydrationRow]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in rows:
        key = f"{row.pdf_status}/{row.text_status}"
        summary[key] = summary.get(key, 0) + 1
    return summary


def _int_year(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def _error_row(
    record: dict[str, Any],
    original_pdf_url: str,
    resolved_pdf_url: str,
    pdf_status: str,
    text_status: str,
    error: str,
    *,
    pdf_path: Path | None = None,
) -> HydrationRow:
    return HydrationRow(
        world=str(record.get("world") or ""),
        title=str(record.get("title") or "Untitled"),
        year=str(record.get("year") or ""),
        source=str(record.get("source") or ""),
        source_id=str(record.get("source_id") or ""),
        doi=str(record.get("doi") or ""),
        original_pdf_url=original_pdf_url,
        resolved_pdf_url=resolved_pdf_url,
        pdf_path=str(pdf_path) if pdf_path is not None else "",
        text_path="",
        pdf_status=pdf_status,
        text_status=text_status,
        text_chars=0,
        error=error,
    )
