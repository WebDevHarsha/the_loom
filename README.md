# the_loom

Two cooperating parts in one project directory — everything portable,
everything inside `F:\RB`, no system installs, no admin rights:

```
pipeline/        LLM tailoring: base resume + JD -> tailored one-page PDFs
ats-engine/      Local ATS scorer: resume PDF vs JD -> explainable 0-100 report
Resumes/         permanent tailored PDFs, one per company
Cover_Letters/   permanent tailored cover letter PDFs, one per company
Scores/          ATS evaluation reports, one per company
temp/            latest-run PDFs only (replaced every run)
venv/            pipeline Python venv (PyLaTeX, pypdf, Gemini)
TinyTeX/         portable LaTeX (reinstall anytime with pipeline/install_tinytex.bat)
```

## Part 1 — `pipeline/` (LLM tailoring)

Takes your base resume, cover letter and projects plus a job description,
asks Gemini (`gemini-3.6-flash`) to tailor the LaTeX to the job, compiles
with PyLaTeX + portable TinyTeX, and auto-shortens until the result is
exactly one page (up to 2 retries; the best successfully compiled version
always wins).

Setup (unchanged from the original pipeline):

1. Python 3.10+ venv:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r pipeline/requirements.txt
   ```

2. LaTeX compiler:

   ```powershell
   .\pipeline\install_tinytex.bat     # downloads + extracts TinyTeX into .\TinyTeX
   ```

3. Gemini API key (never committed; `.env` is gitignored):

   ```powershell
   $env:GEMINI_API_KEY = "your_key"   # GOOGLE_API_KEY also works
   ```

Input files live in `pipeline/` (matching `*.example.txt` templates):

```
pipeline/resume.txt        base resume, full LaTeX source, exactly one page
pipeline/cover_letter.txt  base cover letter, full LaTeX source
pipeline/projects.txt      your real projects, described in detail
pipeline/jd.txt            job description to target
```

Run from the project root:

```powershell
F:\RB\venv\Scripts\python.exe pipeline\pipeline.py
```

Company name is extracted from the JD by Gemini (override with
`--company "Acme Corp"`; also supports `--resume/--cover-letter/--projects/
--jd/--compiler/--model`).

## Part 2 — `ats-engine/` (ATS scoring)

Local, deterministic resume-vs-JD evaluation engine (hybrid: rule/keyword
matching + MiniLM embeddings; **no LLM, no API calls**). Produces an
explainable score with category breakdown, matched/partial/missing skills
with evidence quotes, qualification checks, issues and recommendations.

```powershell
.\venv\Scripts\Activate.ps1
pip install -r ats-engine\requirements.txt        # first run fetches the ~90 MB model into ats-engine\.models
F:\RB\venv\Scripts\python.exe ats-engine\main.py --resume ..\temp\Resume.pdf --jd pipeline\jd.txt --report Scores\<company>.md
```

See `ats-engine/README.md` for the scoring model, weights, optional
features (OCR needs Tesseract; grammar checking needs Java), and tests
(`pytest -q`; fixtures are generated at test time).

## Pipeline ↔ ATS integration

`pipeline.py` calls the ATS engine after generating the tailored resume:

- Scoring runs **by default** against `temp/Resume.pdf` using the local
  engine (installed in the shared root venv); report goes to
  `Scores/<company>.md` (+ `.json`).
- `--yes-llm` adds an optional Gemini-judged score blended with the local
  score (off by default; uses your existing API key).
- Auto-improve: when the score is below `--min-score` (default **70**), the
  pipeline re-tailors with Gemini using the missing keywords and weak-bullet
  feedback (max 2 retry passes; best-scoring compiled version wins).
- `--no-score` disables scoring entirely.

If the engine's dependencies are missing, the pipeline warns and continues
without scoring.

## Notes

- Nothing system-wide: venvs, TinyTeX, pip caches, model caches, downloads
  and outputs all live in this project directory.
- `Resumes/`, `Cover_Letters/`, `Scores/` are re-run safe: same company = overwrite.
- `temp/` contains only the two PDFs after a run; a failed compile leaves
  the stray `.tex` for debugging; it is removed next run.
- Weights and thresholds for the ATS engine are configurable in
  `ats-engine/config.py`; the 25/20/20/10/15/10 split is an initial
  calibration, not an industry standard.
