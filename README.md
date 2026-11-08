# Resume Tailoring + Local ATS Scoring

Two cooperating parts in one project directory — everything portable,
everything inside the project folder, no system installs, no admin rights:

```
pipeline/        LLM tailoring: base resume + JD -> tailored one-page PDFs
ats-engine/       Local ATS scorer: resume PDF vs JD -> explainable 0-100 report
Resumes/          permanent tailored PDFs, one per company+role
Cover_Letters/    permanent tailored cover letter PDFs, one per company+role
Scores/           ATS evaluation reports (.md + .json), one per company+role
temp/             latest-run files only (PDFs + ATS report, replaced every run)
venv/             shared Python venv for both parts
TinyTeX/          portable LaTeX (reinstall anytime with pipeline/install_tinytex.bat)
```

Run all commands from the project root. The code locates its own inputs and
outputs relative to the project folder (no absolute paths required).

## Quick start

```powershell
# 1. Python 3.10+ venv + all dependencies (pipeline + ATS engine)
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r pipeline\requirements.txt
.\venv\Scripts\python.exe -m pip install -r ats-engine\requirements.txt

# 2. LaTeX compiler (portable TinyTeX, ~74 MB, stays in .\TinyTeX)
.\pipeline\install_tinytex.bat

# 3. Gemini API key (current shell only; the pipeline does not load .env)
$env:GEMINI_API_KEY = "your_key"   # GOOGLE_API_KEY also works

# 4. Provide inputs (copy the templates, then replace the placeholders)
#    pipeline\resume.txt, pipeline\cover_letter.txt, pipeline\projects.txt, pipeline\jd.txt

# 5. Run end-to-end (tailor + compile + score)
.\venv\Scripts\python.exe pipeline\pipeline.py
```

Internet is needed once for the TinyTeX download and for the ATS engine's
`all-MiniLM-L6-v2` model (~90 MB, cached in `ats-engine\.models\`).
After that, ATS scoring is fully offline. Never commit your API key or the
personal `pipeline\*.txt` inputs (all gitignored; `*.example.txt` templates are kept).

## Part 1 — `pipeline/` (LLM tailoring)

Takes your base resume, cover letter and projects plus a job description,
asks Gemini (`gemini-3.6-flash` by default) to tailor the LaTeX to the job,
compiles it with PyLaTeX + a LaTeX compiler (portable TinyTeX in this
project, or `latexmk`/`pdflatex` from MiKTeX if already on `PATH`), then
runs ATS scoring/improvement. Page fitting happens **only at the very end**
(after all ATS work): a final layout step scales font size / leading to
exactly one page — shrinking (smaller font + tighter `\linespread`, down to
`\fontsize` 8pt overrides) when the PDF is over one page, growing
(10pt -> 11/12pt + `\linespread` > 1) when the single page looks ~25%
empty. Gemini is never asked to delete content just to chase page count
mid-flow; content-shortening is kept only as a last-resort fallback when
even the smallest font still overflows. On a Gemini 429/quota error the
pipeline exits with a one-line message instead of a stack trace.

Input files live in `pipeline/` (each has a `*.example.txt` template):

```
pipeline/resume.txt        base resume, full LaTeX source, exactly one page
pipeline/cover_letter.txt  base cover letter, full LaTeX source
pipeline/projects.txt      your real projects, described in detail
pipeline/jd.txt            job description to target
```

`--company "Acme Corp"` sets the company part of the output filenames.
`--role "Software Engineer"` (aliases `--job-role`, `--job-title`) sets
the role part. Final files are named `<company>_<role>.pdf`
(e.g. `Machinify_Software_Engineer_Data_Platform.pdf`) so multiple roles
at the same company never overwrite each other.

Otherwise both names are extracted from the JD deterministically
(company: `Company:` line or `About <Name>` heading; role: `Title:` line
or the JD's heading like `Software Engineer — Data Platform`; fallbacks
`Company` / plain `<company>.pdf`). Both are sanitized for use as a
filename. Gemini is not used for this step.

Run from the project root:

```powershell
.\venv\Scripts\python.exe pipeline\pipeline.py
```

Options: `--resume/--cover-letter/--projects/--jd` (input paths),
`--company`, `--role`, `--model`, `--compiler` (LaTeX compiler for PyLaTeX)
and the scoring flags below (`--no-score`, `--yes-llm`, `--min-score`,
`--no-improve`).

## Part 2 — `ats-engine/` (ATS scoring)

Local, deterministic resume-vs-JD evaluation engine (hybrid: rule/keyword
matching + MiniLM embeddings; **no LLM, no API calls**). Produces an
explainable score with category breakdown, matched/partial/missing skills
with evidence quotes, qualification checks, issues and recommendations.
Details are in `ats-engine/README.md`.

| Category | Weight | How |
|---|---:|---|
| Keyword Matching | 25% | exact / alias / conservative fuzzy matching of taxonomy skills, each with resume evidence quotes |
| Semantic Relevance | 20% | per-requirement cosine similarity of JD items vs resume evidence units (`all-MiniLM-L6-v2`) |
| Experience Relevance | 20% | relevance of experience *and projects*, tech coverage, quantified-achievement ratio |
| Qualifications | 10% | degrees / minimum years / certifications -> satisfied / not satisfied / **unknown** (missing info != failure) |
| Resume Structure | 15% | ATS parseability: sections, contact info, columns, tables, scanned pages, garbled text |
| Content Quality | 10% | action verbs, weak phrasing, repetition, quantified achievements, bullet length |

Standalone use (resume formats: `.pdf`, `.docx`, `.txt`):

```powershell
.\venv\Scripts\python.exe ats-engine\main.py --resume temp\Resume.pdf --jd pipeline\jd.txt --report Scores\Example.md --json Scores\Example.json
```

Add `--no-semantic` to skip the embedding model (its weight is then
redistributed). If the model or dependencies are missing, scoring degrades
with a warning instead of crashing. Weights and thresholds are configurable
in `ats-engine/config.py`; the 25/20/20/10/15/10 split is an initial
calibration, not an industry standard. Optional features (OCR needs
Tesseract; grammar checking needs Java) and the library API are documented
in `ats-engine/README.md`. Tests: `pytest -q` from `ats-engine\` (fixtures
are generated at test time).

## Pipeline + ATS integration

`pipeline.py` calls the ATS engine after generating the tailored resume:

- Scoring runs **by default** against `temp/Resume.pdf` using the local
  engine (shared root venv); the report goes to `Scores/<company>_<role>.md`
  (+ `.json`, same stem as the PDFs).
- `--yes-llm` adds an optional Gemini-judged score blended with the local
  score (off by default; uses your existing API key).
- Auto-improve: when the score is below `--min-score` (default **70**), the
  pipeline re-tailors with Gemini using the missing keywords and weak-bullet
  feedback (max 2 retry passes; best-scoring compiled version wins).
  `--no-improve` scores only, without re-tailoring.
- `--no-score` disables scoring (and auto-improve) entirely.

If the engine's dependencies are missing, the pipeline warns and continues
without scoring.

## Notes

- Nothing system-wide: venvs, TinyTeX, pip caches, model caches, downloads
  and outputs all live in this project directory.
- `Resumes/`, `Cover_Letters/`, `Scores/` are re-run safe: same
  `<company>_<role>` = overwrite; a different role at the same company is
  a new file, never an overwrite. Existing plain `<company>.pdf` files
  (pre-role runs) stay as-is; new runs use the `_<role>` suffix.
- `temp/` holds only the latest run's files (`Resume.pdf`, `cover_letter.pdf`,
  `ATS_report.md`/`.json`, `ATS_jd_input.txt`); a failed compile leaves the
  stray `.tex` for debugging, removed next run.
