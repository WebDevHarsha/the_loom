"""DOCX parsing via python-docx."""

from __future__ import annotations

import re
from pathlib import Path

from schemas import ParsedDocument

_MIN_TEXT_CHARS = 40


def parse_docx(path: Path) -> ParsedDocument:
    """Parse a .docx into a ParsedDocument. Raises ValueError on empty or
    malformed files instead of silently accepting them."""
    try:
        import docx
    except ImportError as e:                            # pragma: no cover
        raise RuntimeError(
            "python-docx is not installed; run: pip install python-docx"
        ) from e

    warnings: list[str] = []
    try:
        document = docx.Document(str(path))
        paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        table_rows: list[str] = []
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    table_rows.append(" | ".join(cells))
    except Exception as e:
        raise ValueError(f"Could not read DOCX {path.name}: {type(e).__name__}: {e}") from None

    full = "\n".join(paragraphs + table_rows).strip()
    if len(full) < _MIN_TEXT_CHARS:
        raise ValueError(f"DOCX {path.name} contains too little text to be a resume.")

    table_count = sum(1 for t in document.tables) if document.tables else 0
    if table_count:
        warnings.append(
            "DOCX uses tables; resume content inside tables can confuse ATS parsers."
        )

    text = re.sub(r"[ \t]+", " ", full)
    return ParsedDocument(
        text=text,
        paragraphs=[re.sub(r"[ \t]+", " ", p) for p in paragraphs],
        page_count=1,
        source_format="docx",
        warnings=warnings,
        meta={"tables": table_count},
    )
