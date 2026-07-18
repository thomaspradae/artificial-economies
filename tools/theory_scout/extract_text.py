from __future__ import annotations

from pathlib import Path


def extract_pdf_text(pdf_path: Path, out_dir: Path) -> Path | None:
    try:
        import fitz  # type: ignore
    except ImportError:
        print("[text skipped] PyMuPDF is not installed; install pymupdf to extract PDF text")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdf_path.stem}.txt"
    if out_path.exists() and out_path.stat().st_size > 1_000:
        return out_path
    document = fitz.open(pdf_path)
    chunks = []
    for page_index, page in enumerate(document):
        chunks.append(f"\n\n--- PAGE {page_index + 1} ---\n\n{page.get_text('text')}")
    out_path.write_text("\n".join(chunks), encoding="utf-8")
    return out_path
