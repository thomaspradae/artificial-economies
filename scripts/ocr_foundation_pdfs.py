#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.theory_scout.make_paper_cards import slugify


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def pdf_page_count(pdf_path: Path) -> int | None:
    result = run(["pdfinfo", str(pdf_path)])
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def ocr_pdf(
    pdf_path: Path,
    out_text: Path,
    *,
    max_pages: int,
    dpi: int,
    lang: str,
) -> tuple[str, int, str]:
    page_count = pdf_page_count(pdf_path)
    pages_to_ocr = max_pages if page_count is None else min(page_count, max_pages)
    out_text.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="foundation_ocr_") as tmp:
        tmp_path = Path(tmp)
        for page in range(1, pages_to_ocr + 1):
            prefix = tmp_path / f"page_{page:04d}"
            render = run(
                [
                    "pdftoppm",
                    "-r",
                    str(dpi),
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-png",
                    str(pdf_path),
                    str(prefix),
                ]
            )
            if render.returncode != 0:
                errors.append(f"page {page} render failed: {render.stderr.strip()[:200]}")
                continue
            images = sorted(tmp_path.glob(f"{prefix.name}-*.png"))
            if not images:
                errors.append(f"page {page} render produced no image")
                continue
            ocr = run(["tesseract", str(images[0]), "stdout", "-l", lang, "--psm", "6"])
            if ocr.returncode != 0:
                errors.append(f"page {page} ocr failed: {ocr.stderr.strip()[:200]}")
                continue
            text = ocr.stdout.strip()
            if text:
                chunks.append(f"\\n\\n--- OCR page {page} ---\\n\\n{text}")
    out_text.write_text("\\n".join(chunks).strip() + "\\n", encoding="utf-8")
    size = out_text.stat().st_size if out_text.exists() else 0
    status = "ocr_text" if size >= 1000 else "ocr_failed_low_text"
    return status, size, " | ".join(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR remaining foundation PDFs with pdftoppm+tesseract.")
    parser.add_argument("--queue", default="literature/foundation_pdf_queue.csv")
    parser.add_argument("--pdf-dir", default="literature/pdfs/manual")
    parser.add_argument("--text-dir", default="literature/text/manual")
    parser.add_argument("--report", default="literature/foundation_ocr_report.csv")
    parser.add_argument("--max-pages", type=int, default=80)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--lang", default="eng")
    parser.add_argument("--min-existing-bytes", type=int, default=1000)
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.queue).open(encoding="utf-8")))
    report_rows = []
    for row in rows:
        stem = f"{row['world']}__{row['year']}__{slugify(row['title'])}"
        pdf_path = Path(args.pdf_dir) / f"{stem}.pdf"
        text_path = Path(args.text_dir) / f"{stem}.txt"
        if not pdf_path.exists():
            report_rows.append(
                {
                    "world": row["world"],
                    "title": row["title"],
                    "pdf_path": str(pdf_path),
                    "text_path": str(text_path),
                    "status": "missing_pdf",
                    "text_bytes": 0,
                    "notes": "",
                }
            )
            continue
        if text_path.exists() and text_path.stat().st_size >= args.min_existing_bytes:
            report_rows.append(
                {
                    "world": row["world"],
                    "title": row["title"],
                    "pdf_path": str(pdf_path),
                    "text_path": str(text_path),
                    "status": "existing_text",
                    "text_bytes": text_path.stat().st_size,
                    "notes": "",
                }
            )
            continue
        status, size, notes = ocr_pdf(
            pdf_path,
            text_path,
            max_pages=args.max_pages,
            dpi=args.dpi,
            lang=args.lang,
        )
        report_rows.append(
            {
                "world": row["world"],
                "title": row["title"],
                "pdf_path": str(pdf_path),
                "text_path": str(text_path),
                "status": status,
                "text_bytes": size,
                "notes": notes,
            }
        )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["world", "title", "pdf_path", "text_path", "status", "text_bytes", "notes"],
        )
        writer.writeheader()
        writer.writerows(report_rows)
    counts: dict[str, int] = {}
    for row in report_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"[wrote] {report_path}")
    print("[ocr] " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
