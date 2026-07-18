from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PaperRecord:
    source: str
    source_id: str
    title: str
    year: int | None
    authors: list[str]
    abstract: str | None
    doi: str | None
    url: str | None
    pdf_url: str | None
    citation_count: int | None
    world: str
    query: str
    query_group: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_title(title: str) -> str:
    return " ".join(title.lower().strip().split())


def record_key(record: PaperRecord) -> tuple[str, str, int | None]:
    doi = (record.doi or "").lower().replace("https://doi.org/", "").strip()
    return doi, normalize_title(record.title), record.year
