"""PDF parsing via pdfplumber with layout heuristics.

Extracts text while keeping paragraph/line structure and records layout
information (multi-column suspicion, tables) for the structure analyzer.
"""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

from schemas import ParsedDocument

_MIN_TEXT_CHARS = 80          # below this per document the PDF is likely scanned/broken
_SCANNED_PAGE_CHARS = 80      # below this per page (with images) the page is likely scanned


def parse_pdf(path: Path) -> ParsedDocument:
    """Parse a text-based PDF into a ParsedDocument. Raises ValueError on
    empty/unreadable PDFs so callers never silently accept a bad document."""
    warnings: list[str] = []
    text_parts: list[str] = []
    scanned_pages = 0
    page_count = 0
    table_count = 0
    multi_column = False

    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                page_count += 1
                text = page.extract_text(x_tolerance=1.5) or ""
                if len(text.strip()) < _SCANNED_PAGE_CHARS and page.images:
                    scanned_pages += 1
                text_parts.append(text)
                try:
                    table_count += len(page.find_tables())
                except Exception:                       # table detection is best-effort
                    pass
                multi_column = multi_column or _looks_multi_column(page)
    except ValueError:
        raise
    except Exception as e:                              # corrupted / encrypted / not a pdf
        raise ValueError(f"Could not read PDF {path.name}: {type(e).__name__}: {e}") from None

    full = "\n".join(text_parts).strip()
    if len(full) < _MIN_TEXT_CHARS:
        raise ValueError(
            f"PDF {path.name} contains too little extractable text "
            f"({len(full)} chars). It may be scanned or image-only; try OCR."
        )
    if scanned_pages:
        warnings.append(
            f"{scanned_pages}/{page_count} page(s) look scanned; "
            "text extraction may be incomplete."
        )
    if multi_column:
        warnings.append("PDF appears to be multi-column; reading order may confuse ATS parsers.")

    paragraphs = _split_paragraphs(full)
    return ParsedDocument(
        text=full,
        paragraphs=paragraphs,
        page_count=page_count,
        source_format="pdf",
        scanned=scanned_pages == page_count and page_count > 0,
        warnings=warnings,
        meta={"tables": table_count, "multi_column": multi_column},
    )


def _split_paragraphs(text: str) -> list[str]:
    """Split into paragraphs on blank lines; collapse runs of spaces."""
    paras: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        cleaned = re.sub(r"[ \t]+", " ", block).strip()
        if cleaned:
            paras.append(cleaned)
    return paras


def _looks_multi_column(page) -> bool:
    """Heuristic: a single page is suspicious of a multi-column layout when a
    large clear horizontal gap splits the word x-positions into two clusters."""
    try:
        words = page.extract_words()
    except Exception:
        return False
    if len(words) < 60:
        return False
    x0s = sorted({round(float(w["x0"])) for w in words})
    if len(x0s) < 8:
        return False
    left_limit, right_limit = page.width * 0.15, page.width * 0.85
    best_gap, best_i = 0.0, -1
    for i in range(len(x0s) - 1):
        gap = x0s[i + 1] - x0s[i]
        lo, hi = x0s[i], x0s[i + 1]
        # Only meaningful if both sides actually contain content.
        if lo < left_limit and hi > right_limit and gap > best_gap:
            best_gap, best_i = gap, i
    if best_i == -1 or best_gap < 150:
        return False
    split_x = (x0s[best_i] + x0s[best_i + 1]) / 2
    n_left = sum(1 for w in words if float(w["x0"]) < split_x)
    n_right = len(words) - n_left
    total = len(words)
    return n_left / total > 0.2 and n_right / total > 0.04
