from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "through",
    "to",
    "with",
}


def normalize_title(title: str) -> str:
    text = title.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def title_tokens(title: str) -> set[str]:
    return {token for token in normalize_title(title).split() if len(token) > 2 and token not in STOPWORDS}


def split_titles(value: str) -> list[str]:
    if not value:
        return []
    parts = re.split(r"\s+\|\s+|\n|;;", value)
    return [part.strip() for part in parts if part.strip()]


def title_similarity(left: str, right: str) -> float:
    left_norm = normalize_title(left)
    right_norm = normalize_title(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    left_tokens = title_tokens(left)
    right_tokens = title_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def best_match(title: str, candidates: Iterable[str]) -> tuple[str, float]:
    best_title = ""
    best_score = 0.0
    for candidate in candidates:
        score = title_similarity(title, candidate)
        if score > best_score:
            best_title = candidate
            best_score = score
    return best_title, best_score


def compare_worksheet_rows(
    rows: Iterable[dict[str, Any]],
    *,
    threshold: float = 0.72,
) -> list[dict[str, Any]]:
    compared = []
    for row in rows:
        output = dict(row)
        api_titles = split_titles(str(row.get("api_top_titles") or ""))
        scholar_titles = split_titles(str(row.get("scholar_top_titles_manual") or ""))

        matched_api: set[str] = set()
        scholar_missing = []
        match_notes = []
        for scholar_title in scholar_titles:
            api_title, score = best_match(scholar_title, api_titles)
            if api_title and score >= threshold:
                matched_api.add(api_title)
                match_notes.append(f"{scholar_title} ~= {api_title} ({score:.2f})")
            else:
                scholar_missing.append(scholar_title)

        if scholar_titles:
            api_false_positives = [title for title in api_titles if title not in matched_api]
            overlap_count = len(scholar_titles) - len(scholar_missing)
            overlap_ratio = overlap_count / len(scholar_titles)
            if overlap_ratio >= 0.8:
                status = "strong_overlap"
            elif overlap_ratio >= 0.4:
                status = "partial_overlap"
            else:
                status = "weak_overlap"
        else:
            api_false_positives = []
            overlap_count = 0
            overlap_ratio = ""
            status = "needs_manual_scholar_top_titles"

        output["scholar_missing_from_api"] = " | ".join(scholar_missing)
        output["api_false_positives"] = " | ".join(api_false_positives)
        output["matched_titles"] = " | ".join(match_notes)
        output["api_title_count"] = len(api_titles)
        output["scholar_title_count"] = len(scholar_titles)
        output["overlap_count"] = overlap_count
        output["overlap_ratio"] = overlap_ratio if overlap_ratio == "" else f"{overlap_ratio:.3f}"
        output["comparison_status"] = status
        compared.append(output)
    return compared


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row.keys():
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path, *, limit: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Google Scholar Comparison Report",
        "",
        "This report does not scrape Google Scholar. Paste the first 5-10 Google Scholar titles into `scholar_top_titles_manual` in `literature/scholar_comparison_worksheet.csv`, separated by ` | `, then rerun `scholar-compare`.",
        "",
        "| World | Query Group | Status | API Count | Scholar Count | Overlap | Query | Google Scholar |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows[:limit]:
        overlap = row.get("overlap_ratio") or ""
        url = row.get("google_scholar_url") or ""
        link = f"[open]({url})" if url else ""
        query = str(row.get("query") or "").replace("|", "\\|")
        lines.append(
            f"| {row.get('world','')} | {row.get('query_group','')} | {row.get('comparison_status','')} | "
            f"{row.get('api_title_count','')} | {row.get('scholar_title_count','')} | {overlap} | "
            f"{query} | {link} |"
        )

    needs = [row for row in rows if row.get("comparison_status") == "needs_manual_scholar_top_titles"]
    weak = [row for row in rows if row.get("comparison_status") == "weak_overlap"]
    partial = [row for row in rows if row.get("comparison_status") == "partial_overlap"]

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Rows needing manual Scholar titles: {len(needs)}",
            f"- Weak overlap rows: {len(weak)}",
            f"- Partial overlap rows: {len(partial)}",
            "",
            "## Manual Procedure",
            "",
            "1. Open the `Google Scholar` link for a query.",
            "2. Copy the first 5-10 result titles, ignoring citations/patents when they are not papers.",
            "3. Paste them into `scholar_top_titles_manual` separated by ` | `.",
            "4. Rerun:",
            "",
            "```bash",
            ".venv/bin/python -m tools.theory_scout.cli scholar-compare",
            "```",
            "",
            "Rows with `weak_overlap` are the ones where OpenAlex/Semantic Scholar are probably missing important Google Scholar hits or ranking bad papers too highly.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def compare_worksheet(
    *,
    worksheet_path: Path,
    out_csv: Path,
    out_md: Path,
    threshold: float = 0.72,
) -> list[dict[str, Any]]:
    rows = compare_worksheet_rows(read_csv(worksheet_path), threshold=threshold)
    write_csv(rows, out_csv)
    write_markdown(rows, out_md)
    return rows
