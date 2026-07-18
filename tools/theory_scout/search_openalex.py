from __future__ import annotations

import os

from .http import get_json
from .models import PaperRecord


OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def _abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    positioned: list[tuple[int, str]] = []
    for word, positions in index.items():
        for position in positions:
            positioned.append((position, word))
    return " ".join(word for _, word in sorted(positioned))


def search_openalex(
    query: str,
    world: str,
    query_group: str,
    per_page: int = 20,
) -> list[PaperRecord]:
    params = {
        "search": query,
        "per-page": per_page,
        "sort": "cited_by_count:desc",
    }
    mailto = os.getenv("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto
    api_key = os.getenv("OPENALEX_API_KEY")
    if api_key:
        params["api-key"] = api_key

    data = get_json(OPENALEX_WORKS_URL, params=params)
    records: list[PaperRecord] = []
    for item in data.get("results", []):
        authors = [
            authorship.get("author", {}).get("display_name")
            for authorship in item.get("authorships", [])
            if authorship.get("author", {}).get("display_name")
        ]
        primary = item.get("primary_location") or {}
        best_oa = item.get("best_oa_location") or {}
        doi = item.get("doi")
        if isinstance(doi, str) and doi.startswith("https://doi.org/"):
            doi = doi.replace("https://doi.org/", "")
        records.append(
            PaperRecord(
                source="openalex",
                source_id=item.get("id", ""),
                title=item.get("title") or item.get("display_name") or "",
                year=item.get("publication_year"),
                authors=authors,
                abstract=_abstract_from_inverted_index(item.get("abstract_inverted_index")),
                doi=doi,
                url=primary.get("landing_page_url") or item.get("doi"),
                pdf_url=primary.get("pdf_url") or best_oa.get("pdf_url"),
                citation_count=item.get("cited_by_count"),
                world=world,
                query=query,
                query_group=query_group,
            )
        )
    return records
