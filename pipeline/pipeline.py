"""
Resume / cover-letter pipeline.

Reads:
    resume.txt        - base LaTeX code for the resume
    cover_letter.txt  - base LaTeX code for the cover letter
    projects.txt      - your project details
    jd.txt            - the job description

For each document, calls Gemini (gemini-3.6-flash) to tailor the LaTeX to the
job description and compiles it to PDF with PyLaTeX. Page-count is *not*
handled with Gemini content-shortening during tailoring or ATS scoring —
all content/ATS iteration happens first, on unconstrained source. Only at
the very end (after all ATS work) does a final layout step fit the result
to exactly one page by scaling font size / leading: shrinking (smaller
font + tighter leading) when the PDF is over one page, growing when the
first page looks empty. Gemini shortening is kept only as a last-resort
fallback when even the smallest font still overflows. Output:

    temp/                 - ONLY the two PDFs; replaced on every run
        Resume.pdf, cover_letter.pdf
    Resumes/<company>_<role>.pdf            - one file per company+role, permanent
    Cover_Letters/<company>_<role>.pdf      - one file per company+role, permanent
    Scores/<company>_<role>.md/.json        - same stem as the PDFs

Usage:
    python pipeline\\pipeline.py                (from the project root)
    python pipeline\\pipeline.py --company Acme --model gemini-3.6-flash

Company name: pass --company explicitly, otherwise it is extracted from
jd.txt deterministically (falls back to "Company").

Job role: pass --role explicitly, otherwise it is extracted from
jd.txt (first lines / Title: fields). Final PDFs are saved as
<company>_<role>.pdf (e.g. Machinify_Software_Engineer_Data_Platform.pdf)
so multiple roles at the same company never overwrite each other.
When no role is found, falls back to plain <company>.pdf.

Requires:
    pip install google-genai pylatex pypdf
    + a LaTeX compiler for PyLaTeX: latexmk or pdflatex (e.g. MiKTeX), or the
      portable TinyTeX shipped in this project at <project>/TinyTeX (its bin
      dir is added to PATH automatically).
    + GEMINI_API_KEY (or GOOGLE_API_KEY) set in the environment
"""

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from google import genai

from google.genai import errors as genai_errors
from google.genai import types
from pypdf import PdfReader
from pylatex import Document
from pylatex.errors import CompilerError

MODEL = "gemini-3.7-flash"  # default model for Gemini calls; can be overridden with --model

# Layout: pipeline/ holds code + input files; project root holds the shared
# venv, TinyTeX, and all output dirs (temp/, Resumes/, Cover_Letters/).
PIPELINE_DIR = Path(__file__).resolve().parent
BASE_DIR = PIPELINE_DIR.parent          # project root, e.g. F:\RB
TEMP_DIR = BASE_DIR / "temp"
RESUMES_DIR = BASE_DIR / "Resumes"
COVER_LETTERS_DIR = BASE_DIR / "Cover_Letters"


# ---------------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------------

def read_text(path: Path) -> str:
    if not path.is_file():
        sys.exit(f"ERROR: required file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

RESUME_PROMPT = """\
You are an expert resume writer. You will be given:
1. The LaTeX source code of my resume.
2. My project details.
3. A job description.

Task: edit the resume's LaTeX code so it is best tailored to the job \
description. Use the project details to phrase bullet points, reorder, \
emphasize or reword content where it strengthens the match. Work strictly \
from the material given - never invent experience, skills or facts. Keep the \
same overall structure. For each project, include only 2 subsections (bullet points). Aim for concise, roughly-one-page length, but \
prioritize preserving job-relevant content - do not over-prune to chase \
page count. Final fit to exactly one page is handled separately afterwards \
by font/leading scaling, so do not delete important content for length.

Return ONLY the complete LaTeX source code of the edited resume (from \
\\documentclass to \\end{{document}}), with no Markdown fences, no \\
explanations, and no other text.

=== RESUME LATEX ===
{template}

=== PROJECT DETAILS ===
{projects}

=== JOB DESCRIPTION ===
{jd}
"""

COVER_LETTER_PROMPT = """\
You are an expert cover letter writer. You will be given:
1. The LaTeX source code of my cover letter template.
2. My project details and experience highlights.
3. A job description.

Task: edit the cover letter's LaTeX code so it becomes a compelling, \
personalized cover letter for this exact job description. Reference the \
candidate's projects where relevant to show fit. Work strictly from the \
material given - never invent experience, skills or facts. Keep the letter \
concise and roughly one page, but prioritize completeness - final fit to \
one page is handled separately afterwards by font/leading scaling.

Return ONLY the complete LaTeX source code of the edited cover letter (from \
\\documentclass to \\end{{document}}), with no Markdown fences, no \\
explanations, and no other text.

=== COVER LETTER LATEX ===
{template}

=== PROJECT DETAILS ===
{projects}

=== JOB DESCRIPTION ===
{jd}
"""

COMPRESS_PROMPT = """\
You are an expert LaTeX resume and cover-letter writer. The following LaTeX \
document compiled to {pages} PDF pages, but it MUST fit on exactly ONE page. \
Trim it so it fits on one page while \
keeping the most important, most job-relevant content: shorten bullet \
points, merge duplicated points, reduce content that matters least, and \
tighten sections. Do not change the document class, packages or overall \
structure. Work strictly from the text that is already there - never \
invent experience, skills or facts. Also, use only LaTeX macros that \
already appear in the source below - never introduce new ones (e.g. do \
not use math-mode commands like \\vert, \\cdot or \\sim unless the source \
used them inside math mode; write | or - instead).

Return ONLY the complete LaTeX source code of the shortened document (from \
\\documentclass to \\end{{document}}), with no Markdown fences, no \\
explanations, and no other text.

=== CURRENT LATEX (compiles to {pages} pages) ===
{template}

=== JOB DESCRIPTION IT MUST STILL BE TAILORED TO ===
{jd}
"""


def extract_latex(text: str) -> str:
    """Strip Markdown fences and keep only the actual LaTeX document."""
    # Remove ``` ... ``` fences if the model added them anyway.
    if "```" in text:
        chunks = text.split("```")
        # Prefer a chunk that looks like LaTeX.
        inner = next((c for c in chunks if "\\documentclass" in c), chunks[1] if len(chunks) > 1 else chunks[0])
        text = inner
        # Drop a leading language tag like "latex\n"
        first_line, _, rest = text.partition("\n")
        if first_line.strip().lower() in {"latex", "tex"}:
            text = rest
    start = text.find("\\documentclass")
    end = text.rfind("\\end{document}")
    if start != -1 and end != -1:
        text = text[start : end + len("\\end{document}")]
    if "\\documentclass" not in text:
        raise ValueError(f"Model did not return valid LaTeX:\n{text[:500]}")
    return text.strip() + "\n"


def _quota_exit_if_rate_limited(exc: Exception) -> bool:
    """If `exc` is a Gemini 429/quota error, print a clean message and exit.

    Turns a 30-line traceback into one actionable line. Returns False for
    any other exception type so callers can re-raise."""
    text = str(exc)
    if "429" not in text and "RESOURCE_EXHAUSTED" not in text:
        return False
    m = re.search(r"retry in ([\dhms.]+)", text, re.IGNORECASE)
    delay = f" (API suggests retrying in {m.group(1)})" if m else ""
    sys.exit(
        f"ERROR: Gemini quota exceeded{delay}.\n"
        "  Options: wait for the free-tier window to reset, use a paid key,\n"
        "  or retry with a different model that still has quota: --model <model>"
    )


def generate_latex(template: str, prompt_template: str, projects: str, jd: str, model: str) -> str:
    client = genai.Client()  # picks up GEMINI_API_KEY / GOOGLE_API_KEY
    prompt = prompt_template.format(
        template=template,
        projects=projects,
        jd=jd,
    )
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
            ),
        )
    except genai_errors.APIError as e:
        _quota_exit_if_rate_limited(e)
        raise
    return extract_latex(response.text)


def extract_company_name(jd: str) -> str:
    """Deterministically pull the hiring company's name from the JD text.

    Gemini is reserved for the resume/cover-letter LaTeX only, so this is
    pure regex over common JD shapes. Falls back to 'Company'; override
    with --company."""
    lines = [ln.strip() for ln in jd.splitlines() if ln.strip()]

    # "Company: Acme Inc."
    m = re.search(r"^\s*company\s*[:\-]\s*(.+)$", jd, re.IGNORECASE | re.MULTILINE)
    if m:
        name = m.group(1).strip().strip(".,'\"")
        if 2 <= len(name) <= 60 and len(name.split()) <= 6:
            return name

    # heading "About <Company>" (skip About-the-role/-us variants)
    skip = {"the", "us", "you", "your", "our", "this", "myself", "me",
            "working", "how", "what"}
    for line in lines[:40]:
        m = re.match(r"^about\s+([A-Z][\w&.,'\- ]{1,60})$", line, re.IGNORECASE)
        if not m:
            continue
        name = m.group(1).strip().rstrip(".,")
        if name.split()[0].lower() in skip or len(name.split()) > 6:
            continue
        return name

    # "Innovex AI is a/an/the ..." in the first lines (a company intro sentence)
    for line in lines[:15]:
        m = re.match(
            r"^([A-Z][\w&.\-']*(?:\s+[A-Z][\w&.\-']*){0,5})\s+(?:is|are)\s+(?:a|an|the)\b",
            line,
        )
        if m and not m.group(1).lower().startswith(("we ", "this ", "it ")):
            return m.group(1)

    return "Company"


def sanitize_filename(name: str, *, fallback: str = "Company") -> str:
    """Turn a company or role name into a safe file basename part.

    - removes OS-illegal characters
    - normalizes dashes (— – -) and punctuation to underscores
    - collapses whitespace to single underscores
    e.g. "Software Engineer — Data Platform" -> "Software_Engineer_Data_Platform"
    """
    cleaned = name.strip()
    # Normalize unicode dashes to spaces for clean underscore separation
    cleaned = re.sub(r"[—–]", " ", cleaned)
    # Remove OS-illegal characters
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", cleaned)
    # Drop common punctuation that would look noisy in a filename
    cleaned = re.sub(r"[,;:()\[\]{}'\"]+", "", cleaned)
    # Remaining unusual characters -> underscore
    cleaned = re.sub(r"[^A-Za-z0-9\s\-_]+", "_", cleaned)
    # Whitespace and hyphens -> underscore
    cleaned = re.sub(r"[\s\-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    # Keep a sane max length (avoid ultra-long JDs being used as titles)
    if len(cleaned) > 80:
        cleaned = cleaned[:80].rsplit("_", 1)[0].strip("_") or cleaned[:80]
    return cleaned or fallback


_ROLE_KEYWORDS = {
    "engineer", "developer", "architect", "analyst", "scientist", "manager",
    "designer", "intern", "associate", "consultant", "lead", "specialist",
    "officer", "coordinator", "technician", "programmer", "researcher",
    "director", "head", "staff", "senior", "junior", "principal", "product",
    "operations", "marketing", "sales", "finance", "accountant", "legal",
    "recruiter", "support", "qa", "quality", "devops", "data", "fullstack",
    "frontend", "backend", "mobile", "security", "cloud", "ai", "ml",
}


def _looks_like_job_title(line: str) -> bool:
    """Heuristic: name-like, short, title-cased line that could be a job title."""
    if not line:
        return False
    low = line.lower()
    if low.startswith(
        ("about ", "about:", "location", "company:", "we ", "join ", "what ",
         "who ", "responsibilities", "qualifications", "overview", "description")
    ):
        return False
    words = line.split()
    if len(words) < 2 or len(words) > 9:
        return False
    if len(line) > 80:
        return False
    if line.endswith(".") and len(words) > 4:
        return False
    lowered = {w.lower().strip(",():;") for w in words}
    if lowered & _ROLE_KEYWORDS:
        return True
    title_like = sum(1 for w in words if w[:1].isupper()) / len(words)
    if title_like >= 0.6 and len(line) <= 60:
        return True
    return False


def extract_role_name(jd: str) -> str | None:
    """Deterministically pull the job role/title from the JD text.

    Tries, in order:
    1. Explicit fields: "Title: ...", "Role: ...", "Position: ...", "Job Title: ..."
    2. "We are hiring a/an <Title>" sentences in the first lines
    3. First short lines that look like a title (e.g. the JD's own heading
       "Software Engineer — Data Platform", or "Data Analyst" in jd.example).
    Returns None when nothing plausible is found so callers can fall back
    to plain <company>.pdf naming. Override with --role.
    """
    lines = [ln.strip() for ln in jd.splitlines() if ln.strip()]

    # 1. Explicit Title/Role fields
    m = re.search(
        r"^\s*(?:job\s+)?(?:title|role|position)\s*[:\-]\s*(.+)$",
        jd, re.IGNORECASE | re.MULTILINE,
    )
    if m:
        title = m.group(1).strip().strip(".,'\"")
        if 2 <= len(title) <= 80 and len(title.split()) <= 9:
            return title

    # 2. "We are hiring a Senior Software Engineer ..." in first lines
    for line in lines[:8]:
        m = re.search(
            r"we\s+are\s+hiring\s+(?:a|an|the)?\s*(.+?)\s*(?:for|to|at|in)?\.?$",
            line, re.IGNORECASE,
        )
        if m:
            cand = m.group(1).strip().strip(".,'\"")
            cand = re.sub(r"\s+(for|to|at).*", "", cand).strip()
            if _looks_like_job_title(cand):
                return cand

    # 3. First plausible title-like lines
    for line in lines[:6]:
        if line.lower().startswith("about "):
            continue
        if _looks_like_job_title(line):
            return line

    return None


def build_output_basename(company: str, role: str | None) -> str:
    """Return the file stem for Resumes/Cover_Letters/Scores.

    - with role:    <Company>_<Role>   e.g. Machinify_Software_Engineer_Data_Platform
    - without role: <Company>          (backward-compatible fallback)
    """
    company_part = sanitize_filename(company, fallback="Company")
    if not role:
        return company_part
    role_part = sanitize_filename(role, fallback="")
    if not role_part or role_part == company_part:
        return company_part
    combined = f"{company_part}_{role_part}"
    if len(combined) > 120:
        combined = combined[:120].rsplit("_", 1)[0].strip("_") or company_part
    return combined


# ---------------------------------------------------------------------------
# LaTeX -> PDF via PyLaTeX
# ---------------------------------------------------------------------------

class RawLatexDocument(Document):
    """A PyLaTeX Document that dumps a complete, ready-to-compile LaTeX
    source (e.g. produced by Gemini) instead of assembling its own."""

    def __init__(self, latex_source: str, **kwargs):
        self._latex_source = latex_source
        super().__init__(**kwargs)

    def dumps(self) -> str:
        return self._latex_source


def ensure_compiler_on_path() -> str | None:
    """Make a portable LaTeX shipped in the project dir visible to PyLaTeX.

    Returns the dir added (or None if nothing matched); safe to call again.
    """
    script_dir = BASE_DIR
    for candidate in (
        script_dir / "TinyTeX" / "bin" / "windows",   # TinyTeX (<project>/TinyTeX)
        script_dir / "tinytex" / "bin" / "windows",
        script_dir / "MiKTeX" / "miktex" / "bin" / "x64",
        script_dir / "miktex-portable" / "texmfs" / "install" / "miktex" / "bin" / "x64",
    ):
        if candidate.is_dir() and str(candidate) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")
            return str(candidate)
    return None


def compile_pdf(latex: str, workdir: Path, jobname: str, compiler: str | None) -> Path:
    """Compile finished LaTeX source with PyLaTeX. Returns the PDF path.

    PyLaTeX writes the .tex, compiles, then deletes it (clean_tex=True), so
    workdir only ever contains the .pdf. On a compile failure the .tex stays
    behind for debugging and is replaced quietly on the next run.
    """
    target = str(workdir / jobname)
    doc = RawLatexDocument(latex)
    ensure_compiler_on_path()
    try:
        # redirect_stdout: PyLaTeX prints the full transcript on failure;
        # our RuntimeError below already extracts the useful first errors.
        with contextlib.redirect_stdout(io.StringIO()):
            doc.generate_pdf(
                target,
                clean=True,
                clean_tex=True,    # delete the generated .tex -> PDFs only in temp/
                compiler=compiler, # None -> PyLaTeX tries latexmk, then pdflatex
                silent=True,       # hide pdflatex/latexmk chatter (class + package paths)
            )
    except CompilerError:
        sys.exit(
            "ERROR: PyLaTeX found no LaTeX compiler. Install one:\n"
            "  - MiKTeX (https://miktex.org) -> gives you pdflatex/latexmk on PATH, or\n"
            "  - use the portable TinyTeX in this project: run install_tinytex.bat "
            f"(expected at {BASE_DIR / 'TinyTeX'})."
        )
    except subprocess.CalledProcessError as e:
        # Summarize the LaTeX log instead of the whole latexmk traceback.
        log = Path(target + ".log")
        err_lines = []
        if log.is_file():
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            starts = [i for i, l in enumerate(lines) if l.lstrip().startswith("!")]
            for i in starts[:3]:
                err_lines.append("  " + lines[i])
                if i + 1 < len(lines):
                    err_lines.append("  " + lines[i + 1])
        raise RuntimeError(
            f"LaTeX compile error in {Path(target + '.tex').name}:\n"
            + "\n".join(err_lines[:6])
            if err_lines else
            f"LaTeX compile error in {Path(target + '.tex').name} (exit {e.returncode})"
        ) from None
    pdf_path = Path(target + ".pdf")
    if not pdf_path.is_file():
        raise RuntimeError(f"LaTeX compiler did not produce {pdf_path}")
    return pdf_path


def scratch_cleanup(workdir: Path, jobname: str) -> None:
    """Remove all scratch-tag files (tex/pdf/aux/log/...) of one jobname."""
    for f in workdir.glob(jobname + ".*"):
        f.unlink()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def count_pdf_pages(pdf: Path) -> int:
    try:
        return len(PdfReader(str(pdf)).pages)
    except Exception as e:  # unreadable pdf - treat as unknown, do not retry
        print(f"  (could not count pages of {pdf.name}: {e})")
        return 1


def pdf_content_span(pdf: Path) -> tuple[float, float, float] | None:
    """(span_height_pt, top_margin_pt, page_height_pt) measured from text boxes.

    Uses pypdf's text visitor to find the lowest (min y) and highest (max y)
    text fragment on page 1. Returns None when the PDF has no extractable text.
    """
    try:
        reader = PdfReader(str(pdf))
        page = reader.pages[0]
        page_height = float(page.mediabox.height)
        lows: list[float] = []
        highs: list[float] = []

        def _collect(text: str, _cm, tm, _fm, _fs):
            if text.strip():
                lows.append(float(tm[5]))
                highs.append(float(tm[5]))

        page.extract_text(visitor_text=_collect)
        if not lows:
            return None
        top, bottom = max(lows), min(lows)
        return (top - bottom, page_height - top, page_height)
    except Exception:
        return None


def pdf_fill_ratio(pdf: Path) -> float | None:
    """Content height as a fraction of the printable height.

    Printable height is approximated as page height minus two top-sized
    margins. Ratio ~0.70-0.85 looks noticeably empty; ~0.92+ looks full.
    """
    measured = pdf_content_span(pdf)
    if measured is None:
        return None
    span, top_margin, page_height = measured
    printable = page_height - 2.0 * top_margin
    if printable <= 0:
        return None
    return span / printable


def _documentclass_fontsize(latex: str) -> int:
    m = re.search(r"\\documentclass\s*\[(?:[^\]]*?,\s*)?(\d+)\s*pt", latex)
    return int(m.group(1)) if m else 10


def _set_documentclass_fontsize(latex: str, fontsize_pt: int) -> str:
    """Replace the fontsize option in \\documentclass[...].

    Handles any pt value (10/11/12 but also 8/9 from earlier attempts) so
    repeated scaling never appends a second fontsize. If the document has
    no [...] options at all, it is left unchanged (caller falls back to
    \\fontsize overrides).
    """
    m = re.search(r"\\documentclass\s*\[(.*?)\](\s*\{[^}]*\})", latex, re.DOTALL)
    if not m:
        return latex
    opts = m.group(1)
    if re.search(r"\b\d+(?:\.\d+)?\s*pt\b", opts):
        opts = re.sub(r"\b\d+(?:\.\d+)?\s*pt\b", f"{fontsize_pt}pt", opts, count=1)
    else:
        opts = (opts + f",{fontsize_pt}pt").lstrip(",")
    return latex[: m.start()] + rf"\documentclass[{opts}]" + m.group(2) + latex[m.end():]


def _inject_fontsize_override(latex: str, fontsize_pt: float, leading_pt: float) -> str:
    """Inject \\fontsize{...}{...}\\selectfont right after \\begin{document}.

    Unlike \\documentclass fontsize (article/letter only allow 10/11/12),
    a \\fontsize override can go below 10pt (e.g. 9/8.5). Used as a
    smaller-font fallback when documentclass sizes + tight leading still
    overflow. Strips any previous override so retries never stack.
    """
    latex = re.sub(r"\\fontsize\s*\{[^}]*\}\s*\{[^}]*\}\s*\\selectfont\s*\n?", "", latex, count=1)
    return re.sub(
        r"(\\begin\s*\{document\})",
        rf"\1\n\\fontsize{{{fontsize_pt:g}}}{{{leading_pt:g}}}\\selectfont",
        latex,
        count=1,
    )


def _inject_linespread(latex: str, spread: float) -> str:
    """Insert \\linespread{spread} right after \\begin{document}.

    Strip any previous \\linespread first so retries start from the *base*
    latex and never stack multiplications. Supports both growing
    (spread > 1.0) and shrinking (spread < 1.0); spread == 1.0 just strips.
    """
    latex = re.sub(r"\\linespread\s*\{[^}]*\}\s*\n?", "", latex, count=1)
    if abs(spread - 1.0) < 1e-9:
        return latex
    return re.sub(
        r"(\\begin\s*\{document\})",
        rf"\1\n\\linespread{{{spread:.2f}}}",
        latex,
        count=1,
    )


def _strip_scaling(latex: str) -> str:
    """Remove previous linespread + fontsize overrides so each candidate
    starts from the pristine base LaTeX."""
    latex = re.sub(r"\\linespread\s*\{[^}]*\}\s*\n?", "", latex, count=1)
    latex = re.sub(r"\\fontsize\s*\{[^}]*\}\s*\{[^}]*\}\s*\\selectfont\s*\n?", "", latex, count=1)
    return latex


def _scaled_latex(base_latex: str, base_size: int, fontsize_pt: int, spread: float) -> str:
    """Grow-path helper: documentclass fontsize + linespread."""
    latex = _strip_scaling(base_latex)
    latex = _set_documentclass_fontsize(latex, fontsize_pt)
    return _inject_linespread(latex, spread)


def _shrunk_latex(
    base_latex: str,
    base_size: int,
    *,
    doc_size: int | None = None,
    spread: float = 1.0,
    fontsize_override: float | None = None,
    leading_override: float | None = None,
) -> str:
    """Shrink-path helper: optional documentclass size, optional
    \\fontsize override (for <10pt), and linespread. Always starts from
    the stripped base so candidates are independent."""
    latex = _strip_scaling(base_latex)
    if doc_size is not None:
        latex = _set_documentclass_fontsize(latex, doc_size)
    if fontsize_override is not None:
        latex = _inject_fontsize_override(
            latex, fontsize_override, leading_override or round(fontsize_override * 1.20, 1)
        )
    return _inject_linespread(latex, spread)


def _grow_to_fill(pdf: Path, latex: str, jobname: str, compiler: str | None) -> tuple[Path, str]:
    """Grow a one-page PDF so it fills (nearly) the whole page.

    Text layout is unchanged - only the base font size (via
    \\documentclass) and leading (\\linespread) grow, in steps, until the
    content fills the page or one more step would spill onto page 2. The
    last candidate that stayed on one page wins. Never makes things worse:
    on any failure the original pdf/latex are returned untouched.
    Assumes pdf is already exactly 1 page.
    """
    ratio = pdf_fill_ratio(pdf)
    if ratio is not None and ratio >= 0.92:
        return pdf, latex  # already full

    base_size = _documentclass_fontsize(latex)
    # Build (fontsize, linespread) candidates ordered by effective scale
    # so the "first that looks full" and any spill are both handled by a
    # single ascending pass.
    sizes = [s for s in (11, 12) if s > base_size]
    if not sizes:
        sizes = [base_size]  # already 12pt: grow leading only
    spreads = [1.00, 1.05, 1.10, 1.15, 1.20]
    candidates: list[tuple[int, float, float]] = []  # (fontsize, spread, eff)
    for size in sizes:
        for spread in spreads:
            eff = (size / base_size) * spread
            if eff > 1.0 and eff <= 1.45:
                candidates.append((size, spread, eff))
    candidates.sort(key=lambda c: c[2])
    # Dedupe effective scales (12pt+1.0 == 12pt at 10->12 base, etc. keep first)
    deduped: list[tuple[int, float, float]] = []
    for c in candidates:
        if not deduped or abs(c[2] - deduped[-1][2]) > 0.005:
            deduped.append(c)
    candidates = deduped

    # Start near (but slightly below) the scale the measured gap suggests,
    # so we don't waste compiles on sizes that are obviously still empty.
    if ratio:
        needed = 0.95 / max(ratio, 0.5)
        start = 0
        for i, (_, _, eff) in enumerate(candidates):
            if eff >= needed - 0.05:
                start = max(0, i - 1)
                break
        else:
            start = max(0, (len(candidates) - 4))
        candidates = candidates[start:]

    print(f"  Page looks {100 * ratio:.0f}% full; scaling font/leading to fill one page ...")
    best_pdf, best_latex, best_ratio = pdf, latex, ratio if ratio is not None else 0.0
    for idx, (size, spread, eff) in enumerate(candidates):
        candidate_latex = _scaled_latex(latex, base_size, size, spread)
        tag = f"{jobname}__fill{idx}"
        try:
            candidate_pdf = compile_pdf(candidate_latex, TEMP_DIR, tag, compiler)
        except RuntimeError:
            scratch_cleanup(TEMP_DIR, tag)
            break  # bigger candidates will not help if this one won't build
        if count_pdf_pages(candidate_pdf) != 1:
            scratch_cleanup(TEMP_DIR, tag)
            break  # overflowed: previous candidate (if any) is the last good one
        new_ratio = pdf_fill_ratio(candidate_pdf)
        if new_ratio is not None and new_ratio > best_ratio:
            best_pdf, best_latex, best_ratio = candidate_pdf, candidate_latex, new_ratio
        if new_ratio is not None and new_ratio >= 0.93:
            break

    if best_pdf is pdf:
        return pdf, latex

    final = TEMP_DIR / f"{jobname}.pdf"
    shutil.copy2(best_pdf, final)
    for f in TEMP_DIR.glob(f"{jobname}__fill*.*"):
        f.unlink()
    print(f"  Page now ~{100 * best_ratio:.0f}% full.")
    return final, best_latex


def _try_shrink_to_one_page(pdf: Path, latex: str, jobname: str, compiler: str | None) -> tuple[Path, str] | None:
    """Try font/leading shrinking to force an over-long PDF onto 1 page.

    Tries, in descending effective-scale order (largest fitting font first):
    1. documentclass size down (12 -> 11 -> 10) x tighter \\linespread
    2. \\fontsize overrides below 10pt (9 / 8.5 / 8) x tighter leading
    The first candidate that compiles to exactly 1 page wins. Returns None
    when nothing fits (caller may fall back to Gemini shortening).
    """
    base_size = _documentclass_fontsize(latex)
    doc_sizes = [s for s in (12, 11, 10) if s <= base_size]
    spreads = [1.00, 0.97, 0.94, 0.90, 0.86, 0.82]
    phase1: list[tuple[int, float, float]] = []
    for ds in doc_sizes:
        for sp in spreads:
            eff = (ds / base_size) * sp
            if eff >= 1.0:
                continue
            phase1.append((ds, sp, eff))
    phase1.sort(key=lambda c: c[2], reverse=True)
    deduped1: list[tuple[int, float, float]] = []
    for c in phase1:
        if not deduped1 or abs(c[2] - deduped1[-1][2]) > 0.005:
            deduped1.append(c)
    phase1 = deduped1

    overrides = [(9.0, 10.8), (8.5, 10.2), (8.0, 9.6)]
    phase2: list[tuple[float, float, float, float]] = []
    for fs, lead in overrides:
        for sp in (1.00, 0.95, 0.90):
            eff = (fs / base_size) * sp
            if eff >= 1.0:
                continue
            phase2.append((fs, lead, sp, eff))
    phase2.sort(key=lambda c: c[3], reverse=True)

    print(f"  Document is {count_pdf_pages(pdf)} pages; shrinking font/leading to fit on one page ...")

    idx = 0
    for ds, sp, eff in phase1:
        cand_latex = _shrunk_latex(latex, base_size, doc_size=ds, spread=sp)
        tag = f"{jobname}__shrink{idx}"
        idx += 1
        try:
            cand_pdf = compile_pdf(cand_latex, TEMP_DIR, tag, compiler)
        except RuntimeError:
            scratch_cleanup(TEMP_DIR, tag)
            continue
        if count_pdf_pages(cand_pdf) == 1:
            final = TEMP_DIR / f"{jobname}.pdf"
            shutil.copy2(cand_pdf, final)
            for f in TEMP_DIR.glob(f"{jobname}__shrink*.*"):
                f.unlink()
            print(f"  Fit on one page with {ds}pt + linespread {sp:.2f} (scale ~{eff:.2f}).")
            return final, cand_latex
        scratch_cleanup(TEMP_DIR, tag)

    for fs, lead, sp, eff in phase2:
        cand_latex = _shrunk_latex(
            latex, base_size,
            spread=sp, fontsize_override=fs, leading_override=lead,
        )
        tag = f"{jobname}__shrink{idx}"
        idx += 1
        try:
            cand_pdf = compile_pdf(cand_latex, TEMP_DIR, tag, compiler)
        except RuntimeError:
            scratch_cleanup(TEMP_DIR, tag)
            continue
        if count_pdf_pages(cand_pdf) == 1:
            final = TEMP_DIR / f"{jobname}.pdf"
            shutil.copy2(cand_pdf, final)
            for f in TEMP_DIR.glob(f"{jobname}__shrink*.*"):
                f.unlink()
            print(f"  Fit on one page with fontsize override {fs:g}pt + linespread {sp:.2f} (scale ~{eff:.2f}).")
            return final, cand_latex
        scratch_cleanup(TEMP_DIR, tag)

    for f in TEMP_DIR.glob(f"{jobname}__shrink*.*"):
        f.unlink()
    return None


def fit_pdf_to_one_page(
    pdf: Path, latex: str, jobname: str, compiler: str | None,
    model: str | None = None, jd: str | None = None,
) -> tuple[Path, str]:
    """Final page-fit for a document: shrink if >1 page, grow if <full.

    This is the ONLY place where page fitting happens and it must be called
    *after* all content iteration (tailoring + ATS improvement) so Gemini is
    never asked to delete content just to chase page count mid-flow.

    - 1 page and already >=92% full -> return as-is.
    - 1 page but empty -> grow font/leading to fill (\\_grow_to_fill).
    - >1 pages -> shrink font/leading (documentclass -> \\fontsize
      overrides). First scale that fits wins.
    - Still >1 page at the smallest font AND model/jd given -> last-resort
      Gemini shortening (up to 2 tries, also *at the end*), then grow.

    Never makes things worse: on total failure the original pdf/latex are
    returned untouched.
    """
    pages = count_pdf_pages(pdf)
    if pages == 1:
        return _grow_to_fill(pdf, latex, jobname, compiler)

    if pages <= 1:
        return pdf, latex

    fitted = _try_shrink_to_one_page(pdf, latex, jobname, compiler)
    if fitted is not None:
        return fitted

    if not model or jd is None:
        print(f"  WARNING: still {pages} pages even at smallest font and no fallback available; saving as-is.")
        return pdf, latex

    try:
        client = genai.Client()
    except Exception as e:
        print(f"  WARNING: Gemini unavailable for last-resort shortening ({e}); saving {pages}-page version as-is.")
        return pdf, latex

    best_pdf, best_latex, best_pages = pdf, latex, pages
    cur_latex, cur_pages = latex, pages
    for attempt in range(1, 3):
        print(f"  Still over one page at smallest font; asking {model} to shorten content (last resort, attempt {attempt}/2) ...")
        try:
            shortened = generate_shortened_latex(cur_latex, cur_pages, jd, model, client)
            cand_pdf = compile_pdf(shortened, TEMP_DIR, f"{jobname}__gfit{attempt}", compiler)
            cand_pages = count_pdf_pages(cand_pdf)
        except (ValueError, RuntimeError) as e:
            print(f"  WARNING: last-resort shortening attempt {attempt} failed ({type(e).__name__}); {e}")
            scratch_cleanup(TEMP_DIR, f"{jobname}__gfit{attempt}")
            break
        cur_latex, cur_pages = shortened, cand_pages
        if cand_pages < best_pages:
            best_pdf, best_latex, best_pages = cand_pdf, shortened, cand_pages
        if cand_pages <= 1:
            break
        scratch_cleanup(TEMP_DIR, f"{jobname}__gfit{attempt}")

    if best_pages <= 1:
        if best_pdf != pdf:
            final = TEMP_DIR / f"{jobname}.pdf"
            shutil.copy2(best_pdf, final)
            for f in TEMP_DIR.glob(f"{jobname}__gfit*.*"):
                try:
                    f.unlink()
                except OSError:
                    pass
            best_pdf = final
        return _grow_to_fill(best_pdf, best_latex, jobname, compiler)

    for f in TEMP_DIR.glob(f"{jobname}__gfit*.*"):
        try:
            f.unlink()
        except OSError:
            pass
    print(f"  WARNING: still {best_pages} pages after last-resort shortening; saving best available version.")
    return pdf, latex


def fill_pdf_to_one_page(pdf: Path, latex: str, jobname: str, compiler: str | None, model: str | None = None, jd: str | None = None) -> tuple[Path, str]:
    """Backward-compatible alias. Prefer fit_pdf_to_one_page."""
    return fit_pdf_to_one_page(pdf, latex, jobname, compiler, model=model, jd=jd)


def generate_shortened_latex(latex: str, pages: int, jd: str, model: str, client) -> str:
    """Ask Gemini to shrink an over-long compiled document to one page."""
    prompt = COMPRESS_PROMPT.format(template=latex, pages=pages, jd=jd)
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.4),
        )
    except genai_errors.APIError as e:
        _quota_exit_if_rate_limited(e)
        raise
    return extract_latex(response.text)


def generate_pdf(kind: str, template: str, projects: str, jd: str,
                 model: str, workdir: Path, jobname: str,
                 compiler: str | None) -> tuple[Path, str]:
    """Generate the tailored LaTeX once and compile it.

    Page count is *not* handled here. The draft is returned as-is even if
    it is over one page — fitting to exactly one page happens only at the
    very end via fit_pdf_to_one_page (font scaling, after ATS), so Gemini
    is never asked to delete content just to chase page count mid-flow.
    """
    base_prompt = RESUME_PROMPT if kind == "Resume" else COVER_LETTER_PROMPT
    latex = generate_latex(template, base_prompt, projects, jd, model)

    try:
        draft_pdf = compile_pdf(latex, workdir, f"{jobname}__draft", compiler)
    except (ValueError, RuntimeError) as e:
        scratch_cleanup(workdir, f"{jobname}__draft")
        sys.exit(f"ERROR: initial {kind} draft failed to compile: {e}")

    pages = count_pdf_pages(draft_pdf)
    if pages > 1:
        print(f"  {kind} draft is {pages} pages; leaving as-is (final font fit will shrink it to one page after ATS).")

    final = workdir / f"{jobname}.pdf"
    shutil.copy2(draft_pdf, final)
    for f in workdir.glob(f"{jobname}__*.*"):
        f.unlink()
    return final, latex


def generate_one_page_pdf(kind: str, template: str, projects: str, jd: str,
                          model: str, workdir: Path, jobname: str,
                          compiler: str | None, max_fix_attempts: int = 2) -> tuple[Path, str]:
    """Backward-compatible alias. max_fix_attempts is ignored — page fitting
    now happens only at the end via fit_pdf_to_one_page(). Prefer generate_pdf().
    """
    return generate_pdf(kind, template, projects, jd, model, workdir, jobname, compiler)



def copy_final_pdf(pdf: Path, out_dir: Path, name: str) -> Path:
    """Copy the PDF into the permanent dir; the temp/ copy stays as-is.

    Overwrites same-named <company>_<role> files; other files are kept
    (so two roles at one company never overwrite each other).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / name
    shutil.copy2(pdf, final)
    return final


# ---------------------------------------------------------------------------
# ATS scoring (local engine in ../ats-engine) + optional Gemini blend
# ---------------------------------------------------------------------------

ENGINE_DIR = BASE_DIR / "ats-engine"
SCORES_DIR = BASE_DIR / "Scores"
MIN_SCORE_DEFAULT = 70.0
ATS_MAX_IMPROVE_ATTEMPTS = 2

ATS_JUDGE_PROMPT = """\
You are an ATS (Applicant Tracking System) scoring expert. You will be given a job \
description and the plain text of a candidate's resume.

Task: judge how likely this resume would pass keyword and semantic screening for \
this exact job. Work strictly from the material given - never assume facts that are \
not on the resume. Missing information is "unknown", not a failure; do not penalize \
the candidate for information the resume simply does not state. Given only plain \
text you cannot judge layout - ignore layout entirely.

Reply with JSON only, with exactly these fields:
  "score": integer 0-100, overall ATS compatibility,
  "missing_keywords": array of important keywords from the job description for \
which you found no supporting evidence in the resume (may be empty),
  "summary": one short paragraph explaining the score.

=== JOB DESCRIPTION ===
{jd}

=== RESUME TEXT ===
{resume_text}
"""

ATS_IMPROVE_PROMPT = """\
You are an expert resume writer. A resume tailored to the job description below \
scored {score}/100 on ATS compatibility. Edit the resume's LaTeX source so the \
missing evidence is better represented - but only when the candidate's real \
material supports it:
- Work strictly from the current LaTeX, project details and job description given. \
NEVER invent experience, tools or facts. If a missing keyword cannot honestly map \
to real material, leave it out.
- Prefer rewording existing bullets to surface skills that are already there in \
concrete form (e.g. write 'PostgreSQL' instead of just 'databases' - if that is \
in fact what was used).
- Do not worry about page count - final fit to one page is handled separately \
afterwards by font scaling. Prioritize ATS evidence over length. Do not change \
the document class or packages.
- Address: {missing}

Return ONLY the complete LaTeX source (from \\documentclass to \\end{{document}}), \
no fences, no explanation.

=== CURRENT LATEX ===
{latex}

=== PROJECT DETAILS ===
{projects}

=== JOB DESCRIPTION ===
{jd}
"""


def _engine_command() -> list[str]:
    """Command line for the local ATS engine, [] when it is not installed."""
    engine_main = ENGINE_DIR / "main.py"
    if not engine_main.is_file():
        return []
    venv_python = ENGINE_DIR / ".venv" / "Scripts" / "python.exe"
    python = str(venv_python) if venv_python.is_file() else sys.executable
    return [python, str(engine_main)]


def _gemini_ats_score(resume_text: str, jd_text: str, model: str, client) -> dict | None:
    """Optional Gemini-judged score (structured JSON); None on any failure."""
    try:
        prompt = ATS_JUDGE_PROMPT.format(jd=jd_text[:12000], resume_text=resume_text[:12000])
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        data = json.loads(response.text or "{}")
        score = float(data.get("score", 0))
        return {
            "score": max(0.0, min(100.0, score)),
            "missing_keywords": list(data.get("missing_keywords") or [])[:10],
            "summary": str(data.get("summary") or "")[:400],
        }
    except Exception as e:
        print(f"  WARNING: Gemini ATS judge unavailable ({type(e).__name__}: {str(e)[:120]}).")
        return None


def score_resume_ats(pdf: Path, jd_input: Path, *, model: str, use_llm: bool,
                     client=None) -> dict | None:
    """Run the local ATS engine; optionally blend a Gemini-judged score.

    Returns {"local", "gemini", "final", "missing", "recommendations", "md",
    "json"} or None when the engine is unavailable (warned, never fatal).
    """
    cmd = _engine_command()
    if not cmd:
        print("  WARNING: ATS engine missing (expected ats-engine/main.py); skipping ATS scoring.")
        print("           Set it up as described in ats-engine/README.md to enable scoring.")
        return None
    md_path = TEMP_DIR / "ATS_report.md"
    json_path = TEMP_DIR / "ATS_report.json"
    cmd += ["--resume", str(pdf), "--jd", str(jd_input),
            "--json", str(json_path), "--report", str(md_path)]
    print("  Scoring ATS compatibility with the local engine ...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  WARNING: ATS engine failed (exit {proc.returncode}): {(proc.stderr or '').strip()[:400]}")
        print("  TIP: if dependencies are missing, build the engine venv: see ats-engine/README.md")
        return None
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"  WARNING: could not read ATS report: {e}")
        return None

    local_score = float(raw["overall"]["score"])
    missing = list(raw.get("missing_skills") or [])
    recommendations = list(raw.get("recommendations") or [])
    gemini = None
    if use_llm and client is not None:
        resume_text = ""
        try:
            resume_text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)
        except Exception as e:
            print(f"  WARNING: could not extract PDF text for the Gemini judge ({e}).")
        if resume_text:
            gemini = _gemini_ats_score(resume_text, Path(jd_input).read_text(encoding="utf-8"),
                                       model, client)
            if gemini:
                gap = abs(local_score - gemini["score"])
                for kw in gemini["missing_keywords"]:
                    if kw not in missing:
                        missing.append(kw)
                if gap > 15:
                    print(f"  WARNING: Gemini ({gemini['score']:.0f}) and local engine "
                          f"({local_score:.0f}) diverge by {gap:.0f} points; treating both with care.")
    final_score = round((local_score + gemini["score"]) / 2, 1) if gemini else round(local_score, 1)
    return {
        "local": local_score,
        "gemini": gemini["score"] if gemini else None,
        "final": final_score,
        "missing": missing,
        "recommendations": recommendations,
        "md": md_path,
        "json": json_path,
    }


def score_and_maybe_improve(pdf: Path, latex: str, jd: str, projects: str,
                            args) -> tuple[Path, str, dict | None]:
    """Score the resume; below --min-score, re-tailor it with Gemini (bounded).

    Attempts compile under scratch names, and the best-scoring successfully
    compiled version wins regardless of page count. Page fitting is
    deferred to fit_pdf_to_one_page() at the very end (after this entire
    ATS loop), so improved versions are never shortened or discarded for
    being over-long — they are simply shrunk later via font scaling.
    """
    jd_input = TEMP_DIR / "ATS_jd_input.txt"
    jd_input.write_text(jd, encoding="utf-8")

    client = None
    if args.yes_llm or not args.no_improve and args.min_score is not None:
        try:
            client = genai.Client()
        except Exception as e:
            print(f"  WARNING: Gemini unavailable ({e}); scoring without the optional LLM layer, no auto-improve.")
            args.no_improve = True
            args.yes_llm = False

    report = score_resume_ats(pdf, jd_input, model=args.model,
                              use_llm=args.yes_llm, client=client)
    if report is None:
        return pdf, latex, None

    threshold = MIN_SCORE_DEFAULT if args.min_score is None else args.min_score
    print(f"  ATS score: {report['final']:.0f}/100 "
          + (f"(local {report['local']:.0f}, Gemini {report['gemini']:.0f})" if report["gemini"] is not None else "(local engine)")
          + f" | threshold {threshold:.0f}")

    best_pdf, best_latex, best_report = pdf, latex, report
    if args.no_improve:
        return best_pdf, best_latex, best_report

    for attempt in range(1, (ATS_MAX_IMPROVE_ATTEMPTS + 1) if client else 1):
        if best_report["final"] >= threshold:
            break
        print(f"  Score below threshold; asking {args.model} to improve the resume "
              f"(attempt {attempt}/{ATS_MAX_IMPROVE_ATTEMPTS}) ...")
        missing = ", ".join(best_report["missing"][:12]) or "(none reported)"
        prompt = ATS_IMPROVE_PROMPT.format(
            score=best_report["final"], missing=missing,
            latex=best_latex, projects=projects, jd=jd[:12000],
        )
        try:
            response = client.models.generate_content(
                model=args.model, contents=prompt,
                config=types.GenerateContentConfig(temperature=0.4),
            )
            candidate_latex = extract_latex(response.text)
            candidate_pdf = compile_pdf(
                candidate_latex, TEMP_DIR, f"Resume__ats{attempt}", args.compiler)
            pages = count_pdf_pages(candidate_pdf)
            if pages > 1:
                print(f"  Improved version is {pages} pages; keeping it if it scores better (final font fit will shrink it to one page after ATS).")
        except (ValueError, RuntimeError, subprocess.CalledProcessError) as e:
            print(f"  WARNING: improvement attempt {attempt} failed ({type(e).__name__}); keeping previous version.")
            scratch_cleanup(TEMP_DIR, f"Resume__ats{attempt}")
            continue
        new_report = score_resume_ats(candidate_pdf, jd_input, model=args.model,
                                      use_llm=args.yes_llm, client=client)
        if new_report and new_report["final"] > best_report["final"]:
            print(f"  ATS score {best_report['final']:.0f} -> {new_report['final']:.0f}: keeping improved version.")
            if best_pdf != pdf:
                scratch_cleanup(TEMP_DIR, f"Resume__ats{attempt-1}")
            best_pdf, best_latex, best_report = candidate_pdf, candidate_latex, new_report
        else:
            print("  Improved version did not score higher; keeping previous version.")
            scratch_cleanup(TEMP_DIR, f"Resume__ats{attempt}")

    if best_report["final"] < threshold and client is None:
        print("  NOTE: auto-improve and the optional Gemini layer both need Gemini (GEMINI_API_KEY); score kept as-is.")
    if best_pdf != pdf:
        final = TEMP_DIR / "Resume.pdf"
        shutil.copy2(best_pdf, final)  # copy BEFORE cleanup, else best is deleted
        best_pdf = final
    for f in TEMP_DIR.glob("Resume__ats*"):
        f.unlink()
    return best_pdf, best_latex, best_report


def file_score_report(report: dict, company_file: str) -> Path | None:
    """Copy the ATS report into Scores/<stem>.md + .json; returns md path.

    <stem> is <company>_<role> when a role was detected, else <company>.
    """
    scores = SCORES_DIR
    scores.mkdir(parents=True, exist_ok=True)
    md_target = scores / f"{company_file}.md"
    shutil.copy2(report["md"], md_target)
    shutil.copy2(report["json"], scores / f"{company_file}.json")
    return md_target


def run_pipeline(args) -> dict:
    resume = read_text(args.resume)
    cover_letter = read_text(args.cover_letter)
    projects = read_text(args.projects)
    jd = read_text(args.jd)

    ensure_compiler_on_path()

    company = args.company or extract_company_name(jd)
    role = getattr(args, "role", None) or extract_role_name(jd)
    print(f"Company: {company}")
    print(f"Role:    {role if role else '(not detected - use --role to set)'}")
    company_file = build_output_basename(company, role)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, template in (
        ("Resume", resume),
        ("cover_letter", cover_letter),
    ):
        print(f"Generating {name} with {args.model} ...")
        ats_report = None
        pdf, latex = generate_pdf(
            name, template, projects, jd, args.model,
            TEMP_DIR, name, args.compiler,
        )
        print(f"  -> {pdf}")

        if name == "Resume" and not args.no_score:
            pdf, latex, ats_report = score_and_maybe_improve(pdf, latex, jd, projects, args)
            print(f"  -> {pdf}")
            if ats_report:
                score_file = file_score_report(ats_report, company_file)
                print(f"  -> {score_file}")

        # FINAL PAGE-FIT (after ALL content iteration + ATS scoring):
        # shrink font/leading if the result is over one page, grow it if
        # the single page looks empty. No content changes happen here
        # except as a last-resort fallback when even the smallest font
        # still overflows. Layout-only in the normal path.
        pdf, latex = fit_pdf_to_one_page(
            pdf, latex, name, args.compiler, model=args.model, jd=jd,
        )
        print(f"  -> {pdf}")

        if name == "Resume":
            final = copy_final_pdf(pdf, RESUMES_DIR, f"{company_file}.pdf")
        else:
            final = copy_final_pdf(pdf, COVER_LETTERS_DIR, f"{company_file}.pdf")
        results[name] = {"pdf": pdf, "final": final, "ats": ats_report if name == "Resume" else None}
        print(f"  -> {final}")

    results["company"] = company
    results["role"] = role
    results["company_file"] = company_file
    results["file_stem"] = company_file
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Tailor resume & cover letter to a job description and compile PDFs (PyLaTeX).")
    parser.add_argument("--resume", type=Path, default=PIPELINE_DIR / "resume.txt")
    parser.add_argument("--cover-letter", type=Path, default=PIPELINE_DIR / "cover_letter.txt")
    parser.add_argument("--projects", type=Path, default=PIPELINE_DIR / "projects.txt")
    parser.add_argument("--jd", type=Path, default=PIPELINE_DIR / "jd.txt")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--company", default=None, help="Company name for the final PDF filenames (default: extracted from jd.txt)")
    parser.add_argument("--role", "--job-role", "--job-title", dest="role", default=None,
                        help="Job role/title for the filename: <company>_<role>.pdf (default: extracted from jd.txt)")
    parser.add_argument("--compiler", default=None, help="LaTeX compiler for PyLaTeX (default: latexmk, then pdflatex)")
    parser.add_argument("--no-score", action="store_true",
                        help="Skip ATS scoring entirely (also skips auto-improve)")
    parser.add_argument("--yes-llm", action="store_true",
                        help="Blend a Gemini-judged ATS score with the local engine score (off by default)")
    parser.add_argument("--min-score", type=float, default=70.0,
                        help="Auto-improve threshold for the ATS score (default: 70)")
    parser.add_argument("--no-improve", action="store_true",
                        help="Score only; never re-tailor the resume even below the threshold")
    args = parser.parse_args()

    try:
        results = run_pipeline(args)
    except (RuntimeError, ValueError) as e:
        sys.exit(f"ERROR: {e}")

    print("\nDone. Generated:")
    print(f"  temp/Resume.pdf        -> {results['Resume']['final']}")
    print(f"  temp/cover_letter.pdf  -> {results['cover_letter']['final']}")
    ats = results["Resume"].get("ats")
    if ats:
        company_file = results.get("company_file", sanitize_filename(results.get("company", "Company")))
        print(f"  ATS score              -> {ats['final']:.0f}/100 (report: Scores/{company_file}.md)")
    if results.get("role"):
        print(f"  Filename stem          -> {results['company_file']} (<company>_<role>)")


if __name__ == "__main__":
    main()
