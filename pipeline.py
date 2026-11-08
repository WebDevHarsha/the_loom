"""
Resume / cover-letter pipeline.

Reads:
    resume.txt        - base LaTeX code for the resume
    cover_letter.txt  - base LaTeX code for the cover letter
    projects.txt      - your project details
    jd.txt            - the job description

For each document, calls Gemini (gemini-3.6-flash) to tailor the LaTeX to the
job description, compiles it to PDF with PyLaTeX, and if the resume or cover
letter compiles longer than one page, automatically sends it back to Gemini
to be shortened until it fits on a single page (up to 2 retries). Output:

    temp/                 - ONLY the two PDFs; replaced on every run
        Resume.pdf, cover_letter.pdf
    Resumes/<company>.pdf            - one file per company, permanent
    Cover_Letters/<company>.pdf      - one file per company, permanent

Usage:
    python pipeline.py
    python pipeline.py --company Acme --resume other_resume.txt --model gemini-3.6-flash

Company name: pass --company explicitly, otherwise it is extracted from
jd.txt with a small Gemini call ( falls back to "Company").

Requires:
    pip install google-genai pylatex pypdf
    + a LaTeX compiler for PyLaTeX: latexmk or pdflatex (e.g. MiKTeX), or the
      portable TinyTeX shipped in this project at <project>/TinyTeX (its bin
      dir is added to PATH automatically).
    + GEMINI_API_KEY (or GOOGLE_API_KEY) set in the environment
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from google import genai
from google.genai import types
from pypdf import PdfReader
from pylatex import Document
from pylatex.errors import CompilerError

MODEL = "gemini-3.6-flash"

# All output stays inside the project dir (where pipeline.py lives).
BASE_DIR = Path(__file__).resolve().parent
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
same overall structure. The compiled result MUST fit on exactly ONE page - \
if content does not fit, shorten or drop less important bullet points.

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
material given - never invent experience, skills or facts. Keep it to one \
page and keep the letter concise.

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

COMPANY_PROMPT = """\
Below is a job description. What is the name of the company that is hiring? \
If the true employer is hidden behind an agency or "on behalf of" \
construction, still name the actual employer - otherwise name whatever \
company is most clearly the one advertising the job. Reply with ONLY the \
company name and nothing else. If no company name can be found, reply with \
exactly: UNKNOWN

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


def generate_latex(template: str, prompt_template: str, projects: str, jd: str, model: str) -> str:
    client = genai.Client()  # picks up GEMINI_API_KEY / GOOGLE_API_KEY
    prompt = prompt_template.format(
        template=template,
        projects=projects,
        jd=jd,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
        ),
    )
    return extract_latex(response.text)


def extract_company_name(jd: str, model: str) -> str:
    """Ask Gemini for the hiring company's name; falls back to 'Company'."""
    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=COMPANY_PROMPT.format(jd=jd[:8000]),
        config=types.GenerateContentConfig(temperature=0.0),
    )
    name = (response.text or "").strip().strip(".\"' \n\t")
    name = name.splitlines()[0].strip() if name else ""
    if not name or name.upper() == "UNKNOWN" or len(name) > 60:
        return "Company"
    return name


def sanitize_filename(name: str) -> str:
    """Turn a company name into a safe file basename."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name)
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "Company"


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
        doc.generate_pdf(
            target,
            clean=True,
            clean_tex=True,    # delete the generated .tex -> PDFs only in temp/
            compiler=compiler, # None -> PyLaTeX tries latexmk, then pdflatex
            silent=False,
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


def generate_shortened_latex(latex: str, pages: int, jd: str, model: str, client) -> str:
    """Ask Gemini to shrink an over-long compiled document to one page."""
    prompt = COMPRESS_PROMPT.format(template=latex, pages=pages, jd=jd)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.4),
    )
    return extract_latex(response.text)


def generate_one_page_pdf(kind: str, template: str, projects: str, jd: str,
                          model: str, workdir: Path, jobname: str,
                          compiler: str | None, max_fix_attempts: int = 2) -> Path:
    """Generate the tailored LaTeX, compile it, and if the PDF came out longer
    than one page, send it back to Gemini with page feedback and recompile
    (up to `max_fix_attempts` times).

    Every attempt is compiled under its own scratch name (<jobname>__draft,
    <jobname>__v1, ...) so a failed attempt never clobbers the good PDF. In
    the end the version with the fewest pages wins and is copied to
    <workdir>/<jobname>.pdf; all scratch files are removed."""
    client = genai.Client()  # picks up GEMINI_API_KEY / GOOGLE_API_KEY
    base_prompt = RESUME_PROMPT if kind == "Resume" else COVER_LETTER_PROMPT
    latex = generate_latex(template, base_prompt, projects, jd, model)

    def try_compile(source: str, tag: str) -> tuple[Path, int]:
        """Compile one version under a scratch tag. Returns (pdf, pages).
        On failure all partial scratch files of that tag are removed."""
        try:
            p = compile_pdf(source, workdir, f"{jobname}__{tag}", compiler)
            return p, count_pdf_pages(p)
        except (ValueError, RuntimeError):
            scratch_cleanup(workdir, f"{jobname}__{tag}")
            raise

    # The basic tailored draft must compile - if it fails, crash with the
    # (now short) error message.
    try:
        draft_pdf, draft_pages = try_compile(latex, "draft")
    except (ValueError, RuntimeError) as e:
        sys.exit(f"ERROR: initial {kind} draft failed to compile: {e}")

    versions = [(draft_pdf, draft_pages)]

    for attempt in range(1, max_fix_attempts + 1):
        best_pdf, best_pages = min(versions, key=lambda t: t[1])
        if best_pages <= 1:
            break
        print(f"  {kind} came out {best_pages} pages; asking {model} to fit it on one page (attempt {attempt}/{max_fix_attempts}) ...")
        try:
            shortened = generate_shortened_latex(latex, best_pages, jd, model, client)
            p, n = try_compile(shortened, f"v{attempt}")
            versions.append((p, n))
            latex = shortened
        except (ValueError, RuntimeError) as e:
            print(f"  WARNING: shortened version failed ({type(e).__name__}); keeping previous successful versions. {e}")

    # Winner: the successfully compiled version with the fewest pages
    # (ties -> the earliest, i.e. the untouched draft).
    best_pdf, best_pages = min(versions, key=lambda t: t[1])
    final = workdir / f"{jobname}.pdf"
    shutil.copy2(best_pdf, final)   # replaces any stale file of the last run
    for f in workdir.glob(f"{jobname}__*.*"):   # remove all scratch files
        f.unlink()
    if best_pages > 1:
        print(f"  WARNING: {kind} still exceeds one page after {max_fix_attempts} retries; saving the best available version.")
    return final



def copy_final_pdf(pdf: Path, out_dir: Path, name: str) -> Path:
    """Copy the PDF into the permanent dir; the temp/ copy stays as-is.

    Overwrites same-named company files; other companies' files are kept.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / name
    shutil.copy2(pdf, final)
    return final


def run_pipeline(args) -> dict:
    resume = read_text(args.resume)
    cover_letter = read_text(args.cover_letter)
    projects = read_text(args.projects)
    jd = read_text(args.jd)

    ensure_compiler_on_path()

    company = args.company or extract_company_name(jd, args.model)
    print(f"Company: {company}")
    company_file = sanitize_filename(company)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, template in (
        ("Resume", resume),
        ("cover_letter", cover_letter),
    ):
        print(f"Generating {name} with {args.model} ...")
        pdf = generate_one_page_pdf(
            name, template, projects, jd, args.model,
            TEMP_DIR, name, args.compiler,
        )
        print(f"  -> {pdf}")

        if name == "Resume":
            final = copy_final_pdf(pdf, RESUMES_DIR, f"{company_file}.pdf")
        else:
            final = copy_final_pdf(pdf, COVER_LETTERS_DIR, f"{company_file}.pdf")
        results[name] = {"pdf": pdf, "final": final}
        print(f"  -> {final}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Tailor resume & cover letter to a job description and compile PDFs (PyLaTeX).")
    parser.add_argument("--resume", type=Path, default=Path("resume.txt"))
    parser.add_argument("--cover-letter", type=Path, default=Path("cover_letter.txt"))
    parser.add_argument("--projects", type=Path, default=Path("projects.txt"))
    parser.add_argument("--jd", type=Path, default=Path("jd.txt"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--company", default=None, help="Company name for the final PDF filenames (default: extracted from jd.txt)")
    parser.add_argument("--compiler", default=None, help="LaTeX compiler for PyLaTeX (default: latexmk, then pdflatex)")
    args = parser.parse_args()

    try:
        results = run_pipeline(args)
    except (RuntimeError, ValueError) as e:
        sys.exit(f"ERROR: {e}")

    print("\nDone. Generated:")
    print(f"  temp/Resume.pdf        -> {results['Resume']['final']}")
    print(f"  temp/cover_letter.pdf  -> {results['cover_letter']['final']}")


if __name__ == "__main__":
    main()
