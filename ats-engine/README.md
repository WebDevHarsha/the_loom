# ats-engine

Explainable ATS (Applicant Tracking System) resume evaluation engine.
Compares a resume (.pdf / .docx / .txt) against a job description and
produces a 0-100 compatibility score with a full breakdown, evidence
quotes, and actionable recommendations.

- **No LLM, no API keys.** Scoring is deterministic NLP + embeddings.
- Everything (models, caches) stays **inside this directory** — nothing is
  installed system-wide and no admin rights are needed.

## How it scores

| Category | Weight | How |
|---|---:|---|
| Keyword Matching | 25% | exact / alias / conservative fuzzy matching of taxonomy skills, each with resume evidence quotes |
| Semantic Relevance | 20% | per-requirement cosine similarity of JD items vs resume evidence units (`all-MiniLM-L6-v2`) |
| Experience Relevance | 20% | relevance of experience *and projects* (internships/projects are first-class evidence), tech coverage, quantified-achievement ratio |
| Qualifications | 10% | degrees / minimum years / certifications → satisfied / not satisfied / **unknown** (missing info ≠ failure) |
| Resume Structure | 15% | ATS parseability: sections, contact info, columns, tables, scanned pages, garbled text |
| Content Quality | 10% | action verbs, weak phrasing, repetition, quantified achievements, bullet length |

Weights live in `config.py` / `scoring/weights.py` and are easy to
recalibrate. Skill vocabulary is data-driven: `data/skill_taxonomy.json`
plus `data/skill_aliases.json` (edit freely; no code changes needed).

## Setup

The engine shares the project's root venv (`F:\RB\venv`) with the pipeline
(3.10+ Python). From the project root:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r ats-engine\requirements.txt
```

First run downloads the `all-MiniLM-L6-v2` model (~90 MB) into
`ats-engine\.models\` (the code pins the cache folder) — internet is needed
once. After that it is fully offline.

Optional features, disabled by default and never blocking:

- **OCR for scanned PDFs** — `pip install pytesseract pillow` + the external
  [Tesseract binary](https://github.com/UB-Mannheim/tesseract/wiki).
- **Grammar checking** — needs Java *and* downloads a ~200 MB LanguageTool
  jar; therefore only runs when you set `ATS_ENGINE_GRAMMAR=1`.

## Usage

Library:

```python
from main import evaluate_resume
report = evaluate_resume("resume.pdf", "<job description text>")
print(report.overall.score, report.overall.explanations)
```

CLI:

```powershell
python main.py --resume resume.pdf --jd jd.txt
python main.py --resume resume.pdf --jd jd.txt --json report.json --report ATS_REPORT.md
python main.py --resume resume.docx --jd jd.txt --no-semantic     # skip embeddings
```

> Note: the text report should be saved with a `.txt` extension kept or
> `.md`; the file is plain text either way.

Report includes: overall score + weights, category breakdown with bars,
matched / partial / missing skills, qualification checks (each explained),
semantic best-match evidence, issues, recommendations, and the standard
caveat that different ATS products parse differently.

## Tests

```powershell
pytest -q
```

Fixtures are generated at test time into `tests/fixtures/` (a minimal PDF
writer in `conftest.py`), so no binary blobs are committed.

## Design notes / deliberate deviations

- **Deterministic core.** Sentence splitting, sectioning, and matching are
  rule-based (regex + RapidFuzz) for reproducibility; spaCy is not required.
  The transformer model is used only for semantic similarity, per spec.
- **Semantic similarity never proves a skill.** It only surfaces related
  evidence; skill possession requires exact/alias keyword hits.
- **Unknown ≠ failure.** Qualifications the resume simply does not mention
  are scored as Unknown (55/100) and flagged for verification, not penalized
  as unmet.
- Wrong/insane inputs fail loudly (empty, corrupt, or scanned files).
