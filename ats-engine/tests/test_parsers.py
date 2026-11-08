"""Parser tests: PDF, DOCX, TXT, and malformed-input handling."""

from __future__ import annotations

import pytest

from parsers import docx_parser, pdf_parser


def test_pdf_fixture_extracts_text(strong_resume_pdf):
    parsed = pdf_parser.parse_pdf(strong_resume_pdf)
    assert parsed.source_format == "pdf"
    assert "ANITA KRISHNAN" in parsed.text
    assert "PyTorch" in parsed.text
    assert parsed.page_count == 1


def test_docx_fixture_extracts_text(strong_resume_docx):
    parsed = docx_parser.parse_docx(strong_resume_docx)
    assert "ANITA KRISHNAN" in parsed.text
    assert "PostgreSQL" in parsed.text


def test_empty_pdf_rejected(tmp_path):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")
    with pytest.raises(ValueError):
        pdf_parser.parse_pdf(path)


def test_corrupt_pdf_rejected(tmp_path):
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"%PDF-1.4\n" + b"\x00garbage" * 500 + b" end")
    with pytest.raises(ValueError):
        pdf_parser.parse_pdf(path)


def test_corrupt_docx_rejected(tmp_path):
    path = tmp_path / "corrupt.docx"
    path.write_bytes(b"PK\x03\x04 not really a docx" * 20)
    with pytest.raises(ValueError):
        docx_parser.parse_docx(path)


def test_empty_txt_rejected(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")
    from main import _parse_resume

    with pytest.raises(ValueError):
        _parse_resume(path)


def test_unsupported_format_rejected(tmp_path):
    from main import _parse_resume

    path = tmp_path / "resume.rtf"
    path.write_text("{\\rtf}", encoding="utf-8")
    with pytest.raises(ValueError):
        _parse_resume(path)
