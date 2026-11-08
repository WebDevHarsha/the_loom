"""Scoring engine tests: weights, math, no-inflation guarantees."""

from __future__ import annotations

from scoring.scoring_engine import build_report, score_categories
from scoring.weights import DEFAULT_WEIGHTS, normalize, assert_sums_to_100
from schemas import CategoryScore, ParsedDocument, SkillMatch, SkillRequirement

import pytest

import main


def test_default_weights_sum_to_100():
    assert_sums_to_100()
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 100.0) < 1e-9


def test_normalize_renormalizes_partial_weights():
    w = normalize({"Keyword Matching": 1.0, "Qualifications": 1.0})
    assert abs(sum(w.values()) - 100.0) < 1e-6


def test_weighted_average_math():
    cats = [
        CategoryScore(name="Keyword Matching", score=80),
        CategoryScore(name="Semantic Relevance", score=60),
        CategoryScore(name="Experience Relevance", score=100),
        CategoryScore(name="Qualifications", score=0),
        CategoryScore(name="Resume Structure", score=100),
        CategoryScore(name="Content Quality", score=100),
    ]
    # expected: (80*25 + 60*20 + 100*20 + 0*10 + 100*15 + 100*10)/100 = 77
    overall = score_categories(cats)
    assert overall.score == pytest.approx(77.0, abs=0.1)
    assert overall.confidence == pytest.approx(1.0)


def test_degraded_categories_lower_confidence_only_in_report_cats():
    cats = [
        CategoryScore(name="Keyword Matching", score=50, degraded=True),
        CategoryScore(name="Resume Structure", score=50),
    ]
    overall = score_categories(cats)
    assert overall.confidence < 1.0


def _empty_report(sections=("experience", "skills"), text="JOHN DOE\nprojects\n"):
    """A report where nothing is evidenced."""
    cat = CategoryScore(name="Keyword Matching", score=0.0)
    return cat, ParsedDocument(text=text, paragraphs=[text], source_format="text")


def test_no_evidence_means_no_inflation(tmp_path, sample_jd_text):
    """Adversarial resume: generic words, zero matching skills -> low score."""
    weak = tmp_path / "weak.txt"
    weak.write_text(
        "SAM DOYLE\nPROFILE\nEnthusiastic learner. Helped with various tasks.\n",
        encoding="utf-8",
    )
    weak_report = main.evaluate_resume(
        weak, sample_jd_text, no_semantic=True
    )
    kw = next(c for c in weak_report.categories if c.name == "Keyword Matching")
    assert kw.score < 20
    for truly_required in ("Python", "PyTorch", "SQL", "Machine Learning"):
        assert truly_required not in weak_report.matched_skills
    assert weak_report.overall.score < 55


def test_strong_resume_scores_higher_than_weak(tmp_path, strong_resume_pdf, weak_resume_txt, sample_jd_text):
    strong_report = main.evaluate_resume(
        strong_resume_pdf, sample_jd_text, no_semantic=True
    )
    weak_report = main.evaluate_resume(
        weak_resume_txt, sample_jd_text, no_semantic=True
    )
    assert strong_report.overall.score > weak_report.overall.score + 15
    assert strong_report.overall.score > 55


def test_build_report_lists_skills_correctly():
    req_match = SkillMatch(
        requirement=SkillRequirement(canonical="Python", mandatory=True),
        status="matched", score=1.0)
    req_missing = SkillMatch(
        requirement=SkillRequirement(canonical="Kubernetes", mandatory=True),
        status="missing", score=0.0)
    cats = [CategoryScore(name="Keyword Matching", score=50),
            CategoryScore(name="Resume Structure", score=90),
            CategoryScore(name="Experience Relevance", score=60),
            CategoryScore(name="Content Quality", score=60)]
    report = build_report(
        cats, [req_match, req_missing],
        relevant_experience=[], issues=[], recommendations=[])
    assert "Python" in report.matched_skills
    assert "Kubernetes" in report.missing_skills
    assert 0 < report.overall.score <= 100
