# Resume Tailor

Takes your base resume and cover letter, sends them (plus your project
details and a job description) to Gemini, compiles the tailored versions to
PDF with PyLaTeX + a portable TinyTeX, and files the results:

```
temp/                     latest run only (PDFs only, replaced every run)
    Resume.pdf
    cover_letter.pdf
Resumes/<company>.pdf     one per company, kept permanently
Cover_Letters/<company>.pdf
```

The resume and cover letter are checked for page count after compiling: if
anything spills past one page it is automatically sent back to Gemini to be
shortened (up to 2 retries). A shortened version that fails to compile never
raises - the best successfully compiled version wins instead.

## Setup

1. Python 3.10+ with a virtualenv:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. A LaTeX compiler. Either use the portable TinyTeX shipped for this
   project (nothing system-wide, no admin needed):

   ```powershell
   .\install_tinytex.bat      # downloads + extracts TinyTeX into .\TinyTeX
   ```

   ...or install MiKTeX yourself; PyLaTeX finds `latexmk`/`pdflatex` on PATH.

3. A Gemini API key in the environment (never committed; `.env` is
   gitignored):

   ```powershell
   $env:GEMINI_API_KEY = "your_key"     # GOOGLE_API_KEY also works
   ```

## Usage

Fill in your personal files (use the matching `*.example.txt` files as
starting points, they are tracked and compilable):

```
resume.txt        base resume, full LaTeX source, exactly one page
cover_letter.txt  base cover letter, full LaTeX source
projects.txt      your real projects, described in detail
jd.txt            the job description to target
```

Run from the project directory (input paths are relative):

```powershell
F:\RB\venv\Scripts\python.exe pipeline.py
```

The company name for the output filenames is extracted from `jd.txt` by
Gemini; override it with `--company "Acme Corp"`, or change the input files
with `--resume / --cover-letter / --projects / --jd`. `--compiler` forces a
specific LaTeX engine (default: latexmk, then pdflatex).

## How it works

1. `resume.txt` / `cover_letter.txt` + `projects.txt` + `jd.txt` are sent to
   `gemini-3.6-flash`, which returns the tailored full LaTeX source.
2. `pipeline.RawLatexDocument` (a `pylatex.Document` that dumps ready-made
   LaTeX) writes and compiles it via `generate_pdf`.
3. `pypdf` counts the pages of the result; the auto-shorten loop keeps
   everything to one page.
4. PDFs land in `temp/` and get copied to `Resumes/` / `Cover_Letters/`.

If a package is missing, TinyTeX users can add it locally, e.g.:

```powershell
F:\RB\TinyTeX\bin\windows\tlmgr.bat install fontawesome
```

## Notes

- Everything lives inside the project dir: the venv, TinyTeX, downloads, all
  outputs. Nothing is installed system-wide and no admin rights are needed.
- `Resumes/` and `Cover_Letters/` re-run safe: same company name =
  overwrite; other companies keep their files.
- `temp/` contains only the two PDFs after a run; on a compile failure the
  stray `.tex` stays there as a debugging aid and is removed next run.
