from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path


CITE_RE = re.compile(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]+)\}")
BIB_KEY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
SECTION_RE = re.compile(r"\\(section|subsection|subsubsection|paragraph)\*?\{([^}]+)\}")
OUTPUT_RE = re.compile(r"(outputs/[A-Za-z0-9_./-]+)")
CODE_RE = re.compile(r"\\code\{([^}]+)\}|\\path\{([^}]+)\}")


THEORY_TERMS = (
    "benchmark",
    "theory",
    "predict",
    "truthful",
    "truthfulness",
    "strategy-proof",
    "strategyproof",
    "deferred acceptance",
    "stable matching",
    "vickrey",
    "myerson",
    "nash",
    "joint-profit",
    "free-rider",
    "free rider",
    "public goods",
    "commons",
    "common-pool",
    "collusion",
    "auction",
    "bid shading",
)

METHOD_TERMS = (
    "q-learning",
    "dqn",
    "ppo",
    "independent-dqn",
    "independent learner",
    "centralized critic",
    "centralized-critic",
    "marl",
    "function approximation",
    "policy gradient",
)

RESULT_TERMS = (
    "full run",
    "full table",
    "n=20",
    "40,000",
    "validation",
    "validated",
    "audit",
    "output",
    "outputs/",
    "table~",
    "mean",
    "rate",
    "welfare",
    "profit",
    "survival",
    "sustainability",
    "exploitability",
    "trade_count",
)

LIMITATION_TERMS = (
    "cannot claim",
    "not claim",
    "does not prove",
    "not a theorem",
    "limitation",
    "caveat",
    "future work",
    "outside the claim",
    "not final",
)


CITATION_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("algorithmic pricing", "algorithmic collusion", "collusion", "bertrand"), ("calvano2020", "calvano2021")),
    (("q-learning", "reinforcement learning", "tabular"), ("sutton2018",)),
    (("deep q", "dqn", "replay", "target network"), ("mnih2015",)),
    (("ppo", "policy gradient", "proximal policy"), ("schulman2017",)),
    (("independent learner", "independent-dqn", "independent q"), ("tan1993",)),
    (("centralized critic", "multi-agent actor", "maddpg"), ("lowe2017", "tan1993")),
    (("vickrey", "second-price", "second price", "truthful"), ("vickrey1961",)),
    (("myerson", "reserve", "optimal auction", "revenue"), ("myerson1981", "milgrom2004")),
    (("first-price", "first price", "bid shading", "underbid"), ("milgrom2004",)),
    (("regret", "incentive compatibility", "regretnet"), ("dutting2019", "banchio2022")),
    (("public goods", "public-goods", "public good", "free-rider", "free rider", "undercontribution", "social optimum"), ("samuelson1954", "olson1965")),
    (("commons", "common-pool", "tragedy"), ("hardin1968", "ostrom1990")),
    (("punishment", "sanction", "contribution matching", "reputation"), ("fehrgachter2000", "isaac1994", "ostrom1990")),
    (("deferred acceptance", "stable matching", "gale-shapley", "blocking pair"), ("galeshapley1962", "rothsotomayor1990")),
    (("strategy-proof", "strategyproof", "truthful report", "matching market"), ("rothsotomayor1990", "roth1984")),
    (("ai economist", "tax", "redistribution", "planner"), ("zheng2020", "zheng2021")),
)


WORLD_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "Pricing Arena": (
        "outputs/phase3_full/mind_comparison.csv",
        "outputs/pricing_quantity_margin_audit/paired_seed_audit.csv",
        "outputs/known_answer_sanity_checks_current/summary.csv",
    ),
    "Resource Island": (
        "outputs/resource_island_v1_full/summary_aggregate.csv",
        "outputs/resource_island_v1_phase3_full/mind_comparison.csv",
        "outputs/known_answer_sanity_checks_current/summary.csv",
    ),
    "Auction House": (
        "outputs/auction_house_full/summary_aggregate.csv",
        "outputs/auction_house_phase3_full/mind_comparison.csv",
        "outputs/known_answer_sanity_checks_current/summary.csv",
    ),
    "Public Goods": (
        "outputs/public_goods_full/summary_aggregate.csv",
        "outputs/public_goods_phase3_full/mind_comparison.csv",
        "outputs/public_goods_group_size_sweep/summary_aggregate.csv",
        "outputs/known_answer_sanity_checks_current/summary.csv",
    ),
    "Labor Market": (
        "outputs/labor_market_full/summary_aggregate.csv",
        "outputs/labor_market_phase3_full/mind_comparison.csv",
        "outputs/labor_market_benchmark_cases.json",
        "outputs/known_answer_sanity_checks_current/summary.csv",
    ),
}


@dataclass(frozen=True)
class CitationAuditRow:
    claim_id: str
    section: str
    sentence: str
    claim_type: str
    support_type: str
    existing_citations: str
    missing_bib_keys: str
    suggested_citations: str
    artifact_support: str
    status: str
    fix: str


def parse_bib_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(BIB_KEY_RE.findall(path.read_text(encoding="utf-8", errors="replace")))


def parse_bib_titles(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    entries = re.split(r"\n(?=@\w+\s*\{)", text)
    titles: dict[str, str] = {}
    for entry in entries:
        key_match = BIB_KEY_RE.search(entry)
        title_match = re.search(r"\btitle\s*=\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", entry, re.IGNORECASE)
        if key_match and title_match:
            titles[key_match.group(1)] = re.sub(r"\s+", " ", title_match.group(1)).strip()
    return titles


def _strip_comments(tex: str) -> str:
    lines = []
    for line in tex.splitlines():
        if line.lstrip().startswith("%"):
            continue
        lines.append(re.sub(r"(?<!\\)%.*$", "", line))
    return "\n".join(lines)


def _clean_latex(text: str) -> str:
    text = text.replace("~", " ")
    text = re.sub(r"\\ref\{([^}]+)\}", r"ref{\1}", text)
    text = re.sub(r"\\label\{[^}]+\}", "", text)
    text = re.sub(r"\\(emph|textbf|texttt|code|path)\{([^}]+)\}", r"\2", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", "", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def iter_section_sentences(tex_path: Path) -> list[tuple[str, str]]:
    text = _strip_comments(tex_path.read_text(encoding="utf-8", errors="replace"))
    section = "preamble"
    started = False
    chunks: list[tuple[str, str]] = []
    buffer: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == r"\begin{abstract}":
            started = True
            section = "Abstract"
            buffer = []
            continue
        if stripped == r"\end{abstract}":
            if buffer:
                chunks.extend(_sentences_from_text(section, "\n".join(buffer)))
                buffer = []
            continue
        section_match = SECTION_RE.search(line)
        if section_match:
            started = True
            if buffer:
                chunks.extend(_sentences_from_text(section, "\n".join(buffer)))
                buffer = []
            section = _clean_latex(section_match.group(2))
            continue
        if not started:
            continue
        if stripped.startswith("\\begin{") or stripped.startswith("\\end{"):
            continue
        if "&" in line and r"\\" in line:
            continue
        buffer.append(line)
    if buffer:
        chunks.extend(_sentences_from_text(section, "\n".join(buffer)))
    return chunks


def _sentences_from_text(section: str, text: str) -> list[tuple[str, str]]:
    text = re.sub(r"\n\s*\n", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=(?:[A-Z]|\\|The |This |In |For |Under |Table|Figure))", text)
    sentences = []
    for part in parts:
        raw = part.strip()
        clean = _clean_latex(raw)
        if len(clean) >= 40:
            sentences.append((section, raw))
    return sentences


def extract_citations(sentence: str) -> list[str]:
    keys: list[str] = []
    for match in CITE_RE.findall(sentence):
        keys.extend(key.strip() for key in match.split(",") if key.strip())
    return keys


def _contains_any(sentence_lower: str, terms: tuple[str, ...]) -> bool:
    return any(term in sentence_lower for term in terms)


def classify_claim(sentence: str) -> str | None:
    lower = _clean_latex(sentence).lower()
    has_number = bool(re.search(r"(?<![A-Za-z])\d+(?:\.\d+)?", lower))
    labels: list[str] = []
    if _contains_any(lower, THEORY_TERMS):
        labels.append("theory")
    if _contains_any(lower, METHOD_TERMS):
        labels.append("method")
    if _contains_any(lower, RESULT_TERMS) or has_number:
        labels.append("result")
    if _contains_any(lower, LIMITATION_TERMS):
        labels.append("limitation")
    if not labels:
        return None
    return "+".join(labels)


def suggest_citations(sentence: str, available_keys: set[str]) -> list[str]:
    lower = _clean_latex(sentence).lower()
    suggestions: list[str] = []
    for terms, keys in CITATION_RULES:
        if any(term in lower for term in terms):
            suggestions.extend(key for key in keys if key in available_keys)
    seen: set[str] = set()
    return [key for key in suggestions if not (key in seen or seen.add(key))]


def suggest_artifacts(section: str, sentence: str, repo_root: Path) -> list[str]:
    lower = _clean_latex(f"{section} {sentence}").lower()
    explicit = [path for path in OUTPUT_RE.findall(sentence)]
    explicit.extend(match[0] or match[1] for match in CODE_RE.findall(sentence))
    artifacts: list[str] = []
    for path in explicit:
        if path and (repo_root / path).exists():
            artifacts.append(path)
    for world, paths in WORLD_ARTIFACTS.items():
        if world.lower() in lower:
            artifacts.extend(path for path in paths if (repo_root / path).exists())
    seen: set[str] = set()
    return [path for path in artifacts if not (path in seen or seen.add(path))]


def _status_and_fix(
    claim_type: str,
    citations: list[str],
    missing_bib_keys: list[str],
    suggestions: list[str],
    artifacts: list[str],
) -> tuple[str, str, str]:
    has_valid_citation = bool(citations) and not missing_bib_keys
    has_artifact = bool(artifacts)
    if missing_bib_keys:
        return (
            "citation_key_missing",
            "citation",
            "Fix or add missing BibTeX keys: " + ", ".join(missing_bib_keys),
        )
    if "limitation" in claim_type and (has_valid_citation or has_artifact):
        return ("supported_caveat", "citation+artifact" if has_artifact and has_valid_citation else "citation" if has_valid_citation else "artifact", "No action unless wording overclaims.")
    if "result" in claim_type and has_artifact:
        if has_valid_citation:
            return ("supported_citation_and_artifact", "citation+artifact", "No action.")
        return ("supported_artifact", "artifact", "No citation required unless the sentence also invokes prior theory.")
    if has_valid_citation:
        return ("supported_citation", "citation", "No action.")
    if "result" in claim_type:
        return (
            "needs_artifact",
            "none",
            "Add or name the output artifact/table supporting this numeric/result claim.",
        )
    if suggestions:
        return (
            "needs_citation",
            "none",
            "Consider adding " + ", ".join(f"\\citep{{{key}}}" for key in suggestions[:4]) + ".",
        )
    return (
        "manual_verify",
        "none",
        "Manually verify this claim against foundation cards or add a precise citation/artifact.",
    )


def build_citation_audit(
    repo_root: Path,
    tex_path: Path,
    bib_path: Path,
) -> list[CitationAuditRow]:
    bib_keys = parse_bib_keys(bib_path)
    rows: list[CitationAuditRow] = []
    for index, (section, sentence) in enumerate(iter_section_sentences(tex_path), start=1):
        claim_type = classify_claim(sentence)
        if not claim_type:
            continue
        citations = extract_citations(sentence)
        missing = [key for key in citations if key not in bib_keys]
        suggestions = [key for key in suggest_citations(sentence, bib_keys) if key not in citations]
        artifacts = suggest_artifacts(section, sentence, repo_root)
        status, support_type, fix = _status_and_fix(
            claim_type=claim_type,
            citations=citations,
            missing_bib_keys=missing,
            suggestions=suggestions,
            artifacts=artifacts,
        )
        rows.append(
            CitationAuditRow(
                claim_id=f"C{index:04d}",
                section=section,
                sentence=_clean_latex(sentence),
                claim_type=claim_type,
                support_type=support_type,
                existing_citations=", ".join(citations) or "none",
                missing_bib_keys=", ".join(missing) or "none",
                suggested_citations=", ".join(suggestions) or "none",
                artifact_support=", ".join(artifacts) or "none",
                status=status,
                fix=fix,
            )
        )
    return rows


def write_citation_audit_csv(rows: list[CitationAuditRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else list(CitationAuditRow("", "", "", "", "", "", "", "", "", "", "").__dict__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_citation_audit_markdown(rows: list[CitationAuditRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    lines = [
        "# Citation And Claim-Support Audit",
        "",
        "Generated by `python -m tools.theory_scout.cli citation-audit`.",
        "",
        "This audit is intentionally conservative. It does not decide that a claim is true; it checks whether the draft sentence is visibly supported by a BibTeX citation, an output/code artifact, or an explicit manual-review queue.",
        "",
        "Summary: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())),
        "",
    ]
    for status in sorted(counts):
        selected = [row for row in rows if row.status == status]
        lines.extend([f"## {status}", ""])
        for row in selected[:40]:
            lines.extend(
                [
                    f"### {row.claim_id} - {row.section}",
                    "",
                    row.sentence,
                    "",
                    f"- Type: `{row.claim_type}`",
                    f"- Existing citations: `{row.existing_citations}`",
                    f"- Suggested citations: `{row.suggested_citations}`",
                    f"- Artifact support: `{row.artifact_support}`",
                    f"- Fix: {row.fix}",
                    "",
                ]
            )
        if len(selected) > 40:
            lines.append(f"_Omitted {len(selected) - 40} additional `{status}` rows; see CSV._")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_citation_patch_suggestions(rows: list[CitationAuditRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    actionable = [
        row
        for row in rows
        if row.status in {"needs_citation", "needs_artifact", "citation_key_missing", "manual_verify"}
    ]
    lines = [
        "# Suggested Citation / Artifact Patch Queue",
        "",
        "This is a review queue, not an automatic rewrite. Apply only the suggested fixes that are substantively correct after checking the relevant paper card or output.",
        "",
    ]
    for row in actionable:
        lines.extend(
            [
                f"## {row.claim_id} - {row.section} - {row.status}",
                "",
                row.sentence,
                "",
                f"- Suggested citations: `{row.suggested_citations}`",
                f"- Artifact support: `{row.artifact_support}`",
                f"- Fix: {row.fix}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
