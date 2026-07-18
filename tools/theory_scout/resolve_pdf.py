from __future__ import annotations

import os

from .http import get_json


UNPAYWALL_URL = "https://api.unpaywall.org/v2"


def resolve_unpaywall_pdf(doi: str | None) -> str | None:
    email = os.getenv("UNPAYWALL_EMAIL")
    if not email or not doi:
        return None
    clean_doi = doi.replace("https://doi.org/", "").strip()
    if not clean_doi:
        return None
    data = get_json(f"{UNPAYWALL_URL}/{clean_doi}", params={"email": email})
    best = data.get("best_oa_location") or {}
    return best.get("url_for_pdf") or best.get("url")
