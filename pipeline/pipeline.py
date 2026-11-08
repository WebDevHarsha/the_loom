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
    python pipeline\\pipeline.py                (from the project root)
    python pipeline\\pipeline.py --company Acme --model gemini-3.6-flash

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

MODEL = "gemini-3.6-flash"

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


def generate_one_page_pdf(kind: str, template: str, projects: str, jd: str,
                          model: str, workdir: Path, jobname: str,
                          compiler: str | None, max_fix_attempts: int = 2) -> tuple[Path, str]:
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
    return final, latex



def copy_final_pdf(pdf: Path, out_dir: Path, name: str) -> Path:
    """Copy the PDF into the permanent dir; the temp/ copy stays as-is.

    Overwrites same-named company files; other companies' files are kept.
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
- Keep the document exactly ONE page. Do not change the document class or packages.
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

    Mirrors the one-page loop's semantics: attempts compile under scratch
    names, and the best-scoring successfully compiled version wins.
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
            if count_pdf_pages(candidate_pdf) > 1:
                print(f"  Improved version came out {count_pdf_pages(candidate_pdf)} pages; compressing first ...")
                candidate_latex = generate_shortened_latex(
                    candidate_latex, count_pdf_pages(candidate_pdf), jd, args.model, client)
                candidate_pdf = compile_pdf(
                    candidate_latex, TEMP_DIR, f"Resume__ats{attempt}s", args.compiler)
                if count_pdf_pages(candidate_pdf) > 1:
                    print("  Improved version still over one page; discarding this attempt.")
                    continue
        except (ValueError, RuntimeError, subprocess.CalledProcessError) as e:
            print(f"  WARNING: improvement attempt {attempt} failed ({type(e).__name__}); keeping previous version.")
            scratch_cleanup(TEMP_DIR, f"Resume__ats{attempt}")
            continue
        new_report = score_resume_ats(candidate_pdf, jd_input, model=args.model,
                                      use_llm=args.yes_llm, client=client)
        if new_report and new_report["final"] > best_report["final"]:
            print(f"  ATS score {best_report['final']:.0f} -> {new_report['final']:.0f}: keeping improved version.")
            best_pdf, best_latex, best_report = candidate_pdf, candidate_latex, new_report
        else:
            print("  Improved version did not score higher; keeping previous version.")
            scratch_cleanup(TEMP_DIR, f"Resume__ats{attempt}")

    for f in TEMP_DIR.glob("Resume__ats*"):
        f.unlink()
    if best_report["final"] < threshold and client is None:
        print("  NOTE: auto-improve and the optional Gemini layer both need Gemini (GEMINI_API_KEY); score kept as-is.")
    if best_pdf != pdf:
        shutil.copy2(best_pdf, TEMP_DIR / "Resume.pdf")
    return best_pdf, best_latex, best_report


def file_score_report(report: dict, company_file: str) -> Path | None:
    """Copy the ATS report into Scores/<company>.md + .json; returns md path."""
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
    print(f"Company: {company}")
    company_file = sanitize_filename(company)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, template in (
        ("Resume", resume),
        ("cover_letter", cover_letter),
    ):
        print(f"Generating {name} with {args.model} ...")
        ats_report = None
        pdf, latex = generate_one_page_pdf(
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

        if name == "Resume":
            final = copy_final_pdf(pdf, RESUMES_DIR, f"{company_file}.pdf")
        else:
            final = copy_final_pdf(pdf, COVER_LETTERS_DIR, f"{company_file}.pdf")
        results[name] = {"pdf": pdf, "final": final, "ats": ats_report if name == "Resume" else None}
        print(f"  -> {final}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Tailor resume & cover letter to a job description and compile PDFs (PyLaTeX).")
    parser.add_argument("--resume", type=Path, default=PIPELINE_DIR / "resume.txt")
    parser.add_argument("--cover-letter", type=Path, default=PIPELINE_DIR / "cover_letter.txt")
    parser.add_argument("--projects", type=Path, default=PIPELINE_DIR / "projects.txt")
    parser.add_argument("--jd", type=Path, default=PIPELINE_DIR / "jd.txt")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--company", default=None, help="Company name for the final PDF filenames (default: extracted from jd.txt)")
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
        print(f"  ATS score              -> {ats['final']:.0f}/100 (report: Scores/{sanitize_filename(company)}.md)")


if __name__ == "__main__":
    main()
