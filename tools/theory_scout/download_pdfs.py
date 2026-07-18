from __future__ import annotations

import hashlib
from pathlib import Path

from .http import download_bytes


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)[:120]


def paper_key(title: str, year: int | None = None) -> str:
    return hashlib.sha1(f"{title}_{year}".encode("utf-8")).hexdigest()[:12]


def download_pdf(pdf_url: str, title: str, year: int | None, out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{year or 'unknown'}_{safe_name(title)}_{paper_key(title, year)}.pdf"
    if path.exists() and path.stat().st_size > 10_000:
        return path
    try:
        payload, headers = download_bytes(pdf_url, timeout=60)
    except Exception as exc:
        print(f"[pdf failed] {pdf_url}: {exc}")
        return None
    content_type = headers.get("Content-Type", headers.get("content-type", "")).lower()
    if "pdf" not in content_type and not payload.startswith(b"%PDF"):
        print(f"[not pdf] {pdf_url}")
        return None
    path.write_bytes(payload)
    return path
