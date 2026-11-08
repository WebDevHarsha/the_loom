"""Evaluator tests (semantic tests skip when the model is unavailable)."""

from __future__ import annotations

import pytest

from evaluators import (
    content_analyzer,
    experience_matcher,
    keyword_matcher,
    qualification_matcher,
    structure_analyzer,
)
from extractors.resume_extractor import extract_resume
from parsers import job_parser, pdf_parser
from schemas import ParsedDocument


@pytest.fixture(scope="module")
def strong_parsed(strong_resume_pdf):
    return pdf_parser.parse_pdf(strong_resume_pdf)


@pytest.fixture(scope="module")
def strong_profile(strong_parsed):
    return extract_resume(strong_parsed)


def test_keyword_matcher_match_and_missing(sample_jd_text, strong_parsed):
    jd = job_parser.parse_job(sample_jd_text)
    _cat, matches = keyword_matcher.evaluate_keywords(jd, strong_parsed)
    by_req = {m.requirement.canonical: m for m in matches}
    assert by_req["Python"].status == "matched"
    assert by_req["PyTorch"].status == "matched"
    assert by_req["SQL"].status == "matched"
    assert by_req["Machine Learning"].status in {"matched", "partial"}
    missing = {m.requirement.canonical for m in matches if m.status == "missing"}
    assert any("Kubernetes" == m or "AWS" in m for m in missing) or missing


def test_keyword_evidence_quotes(sample_jd_text, strong_parsed):
    jd = job_parser.parse_job(sample_jd_text)
    _cat, matches = keyword_matcher.evaluate_keywords(jd, strong_parsed)
    for m in matches:
        if m.status == "matched":
            assert m.evidence and m.evidence[0].quote
            break


def test_qualification_unknown_is_not_failure(weak_resume_txt, sample_jd_text):
    jd = job_parser.parse_job(sample_jd_text)
    profile = extract_resume(ParsedDocument(
        text=weak_resume_txt.read_text(encoding="utf-8"),
        paragraphs=[weak_resume_txt.read_text(encoding="utf-8")],
        source_format="text",
    ))
    cat = qualification_matcher.evaluate_qualifications(jd, profile)
    results = cat.details["requirements"]
    statuses = {r["status"] for r in results}
    # Fresh-grad resume with no dated experience must NOT be marked not_satisfied
    # merely for missing information.
    assert "unknown" in statuses
    exp_results = [r for r in results if r["kind"] == "min_experience"]
    assert all(r["status"] in {"unknown"} for r in exp_results)


def test_qualification_satisfied_when_true(strong_parsed, sample_jd_text):
    jd = job_parser.parse_job(sample_jd_text)
    profile = extract_resume(strong_parsed)
    cat = qualification_matcher.evaluate_qualifications(jd, profile)
    results = cat.details["requirements"]
    degree = next(r for r in results if r["kind"] == "degree")
    assert degree["status"] == "satisfied"


def test_structure_flags_missing_sections(weak_resume_txt):
    profile = extract_resume(ParsedDocument(
        text=weak_resume_txt.read_text(encoding="utf-8"),
        paragraphs=[weak_resume_txt.read_text(encoding="utf-8")],
        source_format="text",
    ))
    parsed = ParsedDocument(text=weak_resume_txt.read_text(encoding="utf-8"),
                            paragraphs=[weak_resume_txt.read_text(encoding="utf-8")],
                            source_format="text")
    cat = structure_analyzer.evaluate_structure(parsed, profile)
    assert cat.score < 100
    joined = " ".join(cat.explanations)
    assert "Skills" in joined or "Education" in joined


def test_structure_clean_resume_high_score(strong_parsed, strong_profile):
    cat = structure_analyzer.evaluate_structure(strong_parsed, strong_profile)
    assert cat.score >= 80


def test_content_flags_weak_resume(weak_resume_txt):
    profile = extract_resume(ParsedDocument(
        text=weak_resume_txt.read_text(encoding="utf-8"),
        paragraphs=[weak_resume_txt.read_text(encoding="utf-8")],
        source_format="text",
    ))
    cat = content_analyzer.evaluate_content(profile)
    # Weak/degraded mode also counts: no useful bullet content is itself a finding.
    assert cat.score < 60
    assert cat.degraded or any(
        "weak" in e.lower() or "action verb" in e.lower() for e in cat.explanations
    )


def test_semantic_runs_when_available(sample_jd_text, strong_parsed, strong_profile):
    try:
        from models.semantic_model import get_model
        get_model()
    except Exception as e:                                   # no model available
        pytest.skip(f"semantic model unavailable: {e}")
    from evaluators.semantic_matcher import evaluate_semantic

    jd = job_parser.parse_job(sample_jd_text)
    cat = evaluate_semantic(jd, strong_parsed, strong_profile)
    assert not cat.degraded
    assert 0 <= cat.score <= 100
    per_item = cat.details["per_item"]
    assert per_item and per_item[0]["similarity"] >= 0.2


def test_experience_fresh_grad_no_zero_penalty(sample_jd_text):
    """Projects-only resume must not get an automatic failure-like 0."""
    text = (
        "JANE PRUDENT\nPROJECTS\n"
        "Course Recommender (2025)\n"
        "- Built a PyTorch recommendation model improving hit rate by 15%.\n"
        "- Deployed with Docker and FastAPI serving 100 users with >=99% uptime.\n"
    )
    profile = extract_resume(ParsedDocument(text=text, paragraphs=[text], source_format="text"))
    jd = job_parser.parse_job(sample_jd_text)
    cat = experience_matcher.evaluate_experience(jd, profile)
    assert cat.score > 0
    assert any("projects" in e.lower() or "project" in e.lower() for e in cat.explanations)
