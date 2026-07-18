from __future__ import annotations

import os

from .http import get_json
from .models import PaperRecord


SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def search_semantic_scholar(
    query: str,
    world: str,
    query_group: str,
    limit: int = 20,
) -> list[PaperRecord]:
    fields = ",".join(
        [
            "title",
            "year",
            "authors",
            "abstract",
            "url",
            "externalIds",
            "citationCount",
            "openAccessPdf",
        ]
    )
    params = {
        "query": query,
        "limit": limit,
        "fields": fields,
    }
    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    data = get_json(SEMANTIC_SCHOLAR_SEARCH_URL, params=params, headers=headers)
    records: list[PaperRecord] = []
    for item in data.get("data", []):
        external = item.get("externalIds") or {}
        open_pdf = item.get("openAccessPdf") or {}
        records.append(
            PaperRecord(
                source="semantic_scholar",
                source_id=item.get("paperId", ""),
                title=item.get("title") or "",
                year=item.get("year"),
                authors=[a.get("name") for a in item.get("authors", []) if a.get("name")],
                abstract=item.get("abstract"),
                doi=external.get("DOI"),
                url=item.get("url"),
                pdf_url=open_pdf.get("url"),
                citation_count=item.get("citationCount"),
                world=world,
                query=query,
                query_group=query_group,
            )
        )
    return records
