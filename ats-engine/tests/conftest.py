"""Shared test fixtures.

PDF/DOCX sample fixtures are *generated at test time* inside
tests/fixtures/ (deterministic content, no large binary blobs in git).
conftest also puts the engine root on sys.path so `import main` works.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ENGINE_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(ENGINE_ROOT))


# ---------------------------------------------------------------------------

def _make_pdf_bytes(lines: list[tuple[str, int]]) -> bytes:
    """Assemble a minimal valid single-page PDF with the given (text, y) lines.

    Standard objects: catalog, pages, page, content stream, font.
    """
    content_parts: list[str] = ["BT", "/F1 11 Tf"]
    for y, text in lines:
        safe = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        content_parts.append(f"1 0 0 1 72 {y} Tm ({safe}) Tj")
    content_parts.append("ET")
    content = "\n".join(content_parts).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
         b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"),
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return bytes(out)


RESUME_STRONG_LINES = [
    (792 - 60, "ANITA KRISHNAN"),
    (792 - 78, "anita.krishnan@example.com | +91 98765 43210 | linkedin.com/in/anitakrishnan"),
    (792 - 104, "SUMMARY"),
    (792 - 120, "Machine learning student who built scalable data pipelines and deployed models."),
    (792 - 140, "SKILLS"),
    (792 - 156, "Python, PyTorch, SQL, PostgreSQL, Machine Learning, Git, Docker, FastAPI"),
    (792 - 176, "EDUCATION"),
    (792 - 192, "B.Tech in Computer Science, Northfield Institute of Technology, 2026"),
    (792 - 214, "PROJECTS"),
    (792 - 230, "Recommender System for Course Feedback (Jan 2025 - May 2025)"),
    (792 - 246, "- Developed a PyTorch recommender that improved retrieval accuracy by 18%."),
    (792 - 262, "- Built data pipelines processing 250k rows with automated data quality checks."),
    (792 - 278, "- Deployed with Docker and FastAPI serving 500 daily users."),
    (792 - 298, "EXPERIENCE"),
    (792 - 314, "Machine Learning Intern, Prista Analytics (Jun 2025 - Oct 2025)"),
    (792 - 330, "- Analyzed customer datasets in PostgreSQL and reduced query latency by 30%."),
    (792 - 346, "- Automated weekly reports in Python, saving 6 hours per week."),
    (792 - 366, "CERTIFICATIONS"),
    (792 - 382, "AWS Cloud Practitioner"),
]

RESUME_WEAK_LINES = [
    (792 - 60, "SAM DOYLE"),
    (792 - 78, "sam.doyle@example.com | 90000 11122"),
    (792 - 100, "PROFILE"),
    (792 - 116, "Enthusiastic learner seeking opportunities. Hardworking and quick."),
    (792 - 136, "ACTIVITIES"),
    (792 - 152, "Volunteered at the college festival. Helped with various tasks."),
]

JD_SAMPLE_TEXT = """\
Machine Learning Engineering Intern

About Innovex AI
Innovex AI builds applied machine learning systems for retail analytics.

About the Role
You will build training pipelines and deploy small models with the team.

Responsibilities
- Build and maintain data pipelines that feed model training runs.
- Deploy experimental models behind internal APIs and measure quality.
- Analyze model errors and write lessons learned into runbooks.

Minimum Qualifications
- 1+ year of hands-on Python programming experience.
- Working knowledge of PyTorch or scikit-learn.
- Comfortable with SQL (joins, aggregations) and Git workflows.
- Bachelor's degree in Computer Science or a related field.

Nice to have
- Familiarity with Docker, FastAPI or Kubernetes.
- Exposure to AWS S3 or similar cloud storage.
- Coursework or projects in Natural Language Processing.
"""


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    static = FIXTURES_DIR / "sample_jd.txt"
    static.write_text(JD_SAMPLE_TEXT, encoding="utf-8")
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def sample_jd_text(fixtures_dir: Path) -> str:
    return (fixtures_dir / "sample_jd.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def strong_resume_pdf(fixtures_dir: Path) -> Path:
    path = fixtures_dir / "strong_resume.pdf"
    path.write_bytes(_make_pdf_bytes(RESUME_STRONG_LINES))
    return path


@pytest.fixture(scope="session")
def weak_resume_txt(fixtures_dir: Path) -> Path:
    path = fixtures_dir / "weak_resume.txt"
    path.write_text(
        "\n".join(text for _y, text in RESUME_WEAK_LINES), encoding="utf-8"
    )
    return path


@pytest.fixture(scope="session")
def strong_resume_docx(fixtures_dir: Path) -> Path:
    pytest.importorskip("docx")
    import docx

    path = fixtures_dir / "strong_resume.docx"
    d = docx.Document()
    for _y, text in RESUME_STRONG_LINES:
        d.add_paragraph(text)
    path.unlink(missing_ok=True)
    try:
        d.save(str(path))
    except PermissionError:                 # file open elsewhere; use a temp name
        path = fixtures_dir / "strong_resume_2.docx"
        d.save(str(path))
    return path
