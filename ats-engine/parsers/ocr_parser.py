"""OCR fallback for scanned PDFs (optional feature).

Requires the external Tesseract binary. Everything degrades gracefully:
when unavailable the caller keeps the explicit error so the user can decide.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from schemas import ParsedDocument

_MIN_TEXT_CHARS = 80


class OCRUnavailable(RuntimeError):
    """Raised when OCR cannot run (missing package or missing binary)."""


def ocr_available() -> bool:
    try:
        import pytesseract                              # noqa: F401
        return True
    except ImportError:
        return False


def parse_pdf_ocr(path: Path, resolution: int = 300) -> ParsedDocument:
    """Render each PDF page with pdfplumber and OCR it with Tesseract."""
    try:
        import pytesseract
    except ImportError as e:
        raise OCRUnavailable(
            "pytesseract is not installed; run: pip install pytesseract pillow"
        ) from e

    if shutil.which("tesseract") is None:
        raise OCRUnavailable(
            "OCR needs the Tesseract binary (not found on PATH). "
            "Install it locally or provide a text-based PDF instead."
        )

    import warnings as _warnings

    import pdfplumber

    pages_text: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            try:
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    image = page.to_image(resolution=resolution).original
            except Exception as e:
                raise ValueError(f"Could not render page for OCR: {e}") from e
            pages_text.append(pytesseract.image_to_string(image) or "")

    full = "\n".join(pages_text).strip()
    if len(full) < _MIN_TEXT_CHARS:
        raise ValueError(f"OCR produced too little text from {path.name}.")

    from parsers.pdf_parser import _split_paragraphs

    return ParsedDocument(
        text=full,
        paragraphs=_split_paragraphs(full),
        page_count=len(pages_text),
        source_format="ocr",
        scanned=True,
        warnings=["Document was image-based; text recovered via OCR (quality is best-effort)."],
    )
