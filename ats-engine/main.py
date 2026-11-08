"""ATS engine entry point.

Library use:
    from main import evaluate_resume
    report = evaluate_resume("resume.pdf", "<job description text>")
    print(report.overall.score)

CLI use:
    python main.py --resume sample_resume.pdf --jd sample_jd.txt [--json out.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluators import (
    content_analyzer,
    experience_matcher,
    keyword_matcher,
    qualification_matcher,
    structure_analyzer,
)
from models.semantic_model import SemanticModelUnavailable
from parsers import docx_parser, job_parser, ocr_parser, pdf_parser
from extractors.resume_extractor import extract_resume
from scoring.scoring_engine import build_report
from schemas import ATSReport, CategoryScore, ParsedDocument

_ENGINE_VERSION = "1.0.0"


def _parse_resume(path: Path) -> ParsedDocument:
    """Dispatch on extension; scanned PDFs get an OCR fallback attempt."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            return pdf_parser.parse_pdf(path)
        except ValueError:
            # likely scanned / image-only: try OCR before giving up
            if ocr_parser.ocr_available():
                try:
                    return ocr_parser.parse_pdf_ocr(path)
                except (ValueError, ocr_parser.OCRUnavailable):
                    pass           # fall through: surface the original clearer error
            raise
    if suffix in (".docx", ".docm"):
        return docx_parser.parse_docx(path)
    if suffix in (".txt", ".md", ".text"):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) < 40:
            raise ValueError(f"{path.name} contains too little text to be a resume.")
        return ParsedDocument(
            text=text,
            paragraphs=[p for p in text.split("\n\n") if p.strip()],
            source_format="text",
            meta={},
        )
    raise ValueError(f"Unsupported resume format: {path.name} (use .pdf, .docx or .txt)")


def _semantic_or_none(jd, parsed, profile, no_semantic: bool) -> CategoryScore | None:
    """Semantic category, or None when it cannot run (weight redistributes)."""
    if no_semantic:
        return None
    try:
        from evaluators.semantic_matcher import evaluate_semantic
        return evaluate_semantic(jd, parsed, profile)
    except SemanticModelUnavailable as e:
        return None          # reported through meta so the user knows
    except Exception as e:   # embedding runtime problems must not kill a run
        return CategoryScore(
            name="Semantic Relevance",
            score=0.0,
            weight=0.0,
            degraded=True,
            explanations=[f"Semantic matching failed unexpectedly: {e}"],
        )


def evaluate_resume(resume_path, job_description: str, *, no_semantic: bool = False) -> ATSReport:
    """Evaluate one resume file against one job description.

    Raises FileNotFoundError / ValueError with actionable messages.
    """
    if not job_description or not job_description.strip():
        raise ValueError("Job description is empty.")
    resume_path = Path(resume_path)
    if not resume_path.is_file():
        raise FileNotFoundError(f"Resume file not found: {resume_path}")

    parsed = _parse_resume(resume_path)
    jd = job_parser.parse_job(job_description)
    profile = extract_resume(parsed)

    semantic_cat = _semantic_or_none(jd, parsed, profile, no_semantic)

    kw_cat, matches = keyword_matcher.evaluate_keywords(jd, parsed)
    many_fuzzy = sum(1 for m in matches if m.status == "partial")
    fuzzy_note = (
        f"{many_fuzzy} skill(s) matched only fuzzily; verify they are real skills."
        if many_fuzzy else None
    )

    cats = [kw_cat]
    if semantic_cat is not None:
        cats.append(semantic_cat)
    cats.append(experience_matcher.evaluate_experience(jd, profile))
    cats.append(qualification_matcher.evaluate_qualifications(jd, profile))
    cats.append(structure_analyzer.evaluate_structure(parsed, profile))
    cats.append(content_analyzer.evaluate_content(profile))

    # ---- recommendation assembly (specific and actionable, never fabricated)
    recommendations: list[str] = []
    missing_mandatory = [
        m.requirement for m in matches
        if m.status == "missing" and m.requirement.mandatory
    ]
    for req in missing_mandatory[:6]:
        recommendations.append(
            f"Add evidence of '{req.canonical}' if you have worked with it "
            "(tasks, projects, coursework — no invented claims)."
        )
    for m in matches:
        if m.status == "partial" and m.evidence:
            recommendations.append(
                f"'{m.requirement.canonical}' appears only as close phrasing "
                f"(\"{m.evidence[0].quote[:60]}...\"); use the exact keyword if honest."
            )
    content_cat = next(c for c in cats if c.name == "Content Quality")
    recommendations.extend((content_cat.details or {}).get("recommendations") or [])
    structure_cat = next(c for c in cats if c.name == "Resume Structure")
    recommendations.extend(
        f"ATS format: {i}" for i in (structure_cat.details or {}).get("issues", [])[:3]
    )

    issues: list[str] = []
    issues.extend((structure_cat.details or {}).get("issues", []))
    if fuzzy_note:
        issues.append(fuzzy_note)

    relevant_experience: list[str] = []
    from evaluators.keyword_matcher import scan_for_terms
    from extractors import skill_extractor
    tech = [r.canonical for r in jd.all_skills()
            if (r.category or "soft_skills") != "soft_skills"]
    for item in (*profile.projects, *profile.experience):
        item_text = item.text()
        used = [c for c in tech if scan_for_terms(
            [c, *skill_extractor.aliases_for(c)], item_text)]
        reasons = []
        if used:
            reasons.append("demonstrates " + ", ".join(used[:4]))
        if item.quantified:
            reasons.append("includes quantified results")
        if reasons:
            label = "Project" if item.is_project else "Experience"
            title = item.title or "(untitled)"
            relevant_experience.append(f"{label} \"{title}\": " + "; ".join(reasons) + ".")

    semantic_used = semantic_cat is not None and not semantic_cat.degraded
    report = build_report(
        cats,
        matches,
        relevant_experience=relevant_experience,
        issues=issues,
        recommendations=recommendations,
        meta={
            "resume_source": resume_path.name,
            "jd_title": jd.title,
            "company": jd.company,
            "engine_version": _ENGINE_VERSION,
            "semantic_used": semantic_used,
            "resume_warnings": list(parsed.warnings),
            "jd_warnings": list(jd.warnings),
            "skill_counts": {
                "required": len(jd.required_skills),
                "preferred": len(jd.preferred_skills),
            },
        },
    )
    if not semantic_used:
        report.meta["semantic_note"] = (
            "Semantic relevance category skipped: sentence-transformers model "
            "unavailable. Weights were redistributed across tested categories."
        )
        report.recommendations.append(
            "Install sentence-transformers (see README) so semantic matching can run."
        )
    report.recommendations = report.recommendations[:10]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate resume-vs-job-description ATS compatibility "
                    "(local, deterministic + MiniLM semantics; no LLM)."
    )
    parser.add_argument("--resume", required=True, help="Resume file: .pdf, .docx or .txt")
    parser.add_argument("--jd", required=True, help="Job description text file")
    parser.add_argument("--json", default=None, help="Optional path to save the JSON report")
    parser.add_argument("--report", default=None, help="Optional path to save the text report")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Skip sentence-transformers semantic matching")
    args = parser.parse_args(argv)

    try:
        jd_text = Path(args.jd).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"ERROR: cannot read job description: {e}", file=sys.stderr)
        return 2

    try:
        report = evaluate_resume(args.resume, jd_text, no_semantic=args.no_semantic)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    from reports import json_report, text_report

    print(text_report.render_text(report))
    if args.json:
        json_report.save(report, Path(args.json))
        print(f"JSON report saved to {args.json}")
    if args.report:
        text_report.save(report, Path(args.report))
        print(f"Text report saved to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
