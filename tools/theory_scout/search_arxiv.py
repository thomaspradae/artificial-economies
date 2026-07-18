from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET

from .http import download_bytes
from .models import PaperRecord


ARXIV_SEARCH_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _text(entry: ET.Element, name: str) -> str | None:
    child = entry.find(f"atom:{name}", ATOM_NS)
    if child is None or child.text is None:
        return None
    return " ".join(child.text.split())


def search_arxiv(
    query: str,
    world: str,
    query_group: str,
    max_results: int = 20,
) -> list[PaperRecord]:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_SEARCH_URL}?{urllib.parse.urlencode(params)}"
    payload, _ = download_bytes(url, timeout=30)
    root = ET.fromstring(payload)
    records: list[PaperRecord] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        arxiv_id = _text(entry, "id") or ""
        title = _text(entry, "title") or ""
        published = _text(entry, "published")
        year = int(published[:4]) if published and published[:4].isdigit() else None
        authors = [
            name.text.strip()
            for name in entry.findall("atom:author/atom:name", ATOM_NS)
            if name.text
        ]
        pdf_url = None
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href")
                break
        records.append(
            PaperRecord(
                source="arxiv",
                source_id=arxiv_id,
                title=title,
                year=year,
                authors=authors,
                abstract=_text(entry, "summary"),
                doi=None,
                url=arxiv_id,
                pdf_url=pdf_url,
                citation_count=None,
                world=world,
                query=query,
                query_group=query_group,
            )
        )
    return records
