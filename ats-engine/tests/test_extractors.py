"""Extractor tests: skills, aliases, JD parsing, resume profiles."""

from __future__ import annotations

from parsers.job_parser import parse_job
from extractors import skill_extractor
from extractors.resume_extractor import extract_resume
from schemas import ParsedDocument


def test_scan_skills_exact_and_alias():
    text = "Experienced with Postgres and JS. Used PyTorch for ML experiments."
    canonicals = {m.canonical for m in skill_extractor.scan_skills(text)}
    assert "PostgreSQL" in canonicals
    assert "JavaScript" in canonicals
    assert "PyTorch" in canonicals
    assert "Machine Learning" in canonicals


def test_scan_skills_rejects_ordinary_words():
    text = "Read the rest of the runbook, then go to the dashboard for monitoring."
    canonicals = {m.canonical for m in skill_extractor.scan_skills(text)}
    assert "REST API" not in canonicals
    assert "Go" not in canonicals          # lowercase 'go' is a verb


def test_scan_skills_golike_case_sensitive():
    text = "Go developer experienced in Go, plus Go tooling."
    canonicals = {m.canonical for m in skill_extractor.scan_skills(text)}
    assert "Go" in canonicals


def test_canonicalize_alias_variants():
    assert skill_extractor.canonicalize("html5") == "HTML"
    assert skill_extractor.canonicalize("k8s") == "Kubernetes"
    assert skill_extractor.canonicalize("totally-not-a-skill---") is None


def test_parse_job_mandatory_vs_preferred(sample_jd_text):
    jd = parse_job(sample_jd_text)
    required = {r.canonical for r in jd.required_skills}
    preferred = {r.canonical for r in jd.preferred_skills}
    assert "Python" in required                      # listed under Minimum Qualifications
    assert "PyTorch" in required
    assert "SQL" in required
    assert "Natural Language Processing" in preferred  # coursework/nice-to-have
    # Docker/FastAPI/Kubernetes only appear with 'familiarity' wording
    for soft_req in ("Docker", "FastAPI", "Kubernetes"):
        assert soft_req not in required


def test_parse_job_extracts_years_and_degree(sample_jd_text):
    jd = parse_job(sample_jd_text)
    kinds = {q.kind for q in jd.qualifications}
    assert "min_experience" in kinds
    assert "degree" in kinds
    years_q = next(q for q in jd.qualifications if q.kind == "min_experience")
    assert years_q.min_years == 1.0


def test_parse_job_title_and_company(sample_jd_text):
    jd = parse_job(sample_jd_text)
    assert jd.title and "Intern" in jd.title
    assert jd.company == "Innovex AI"


def test_resume_profile_extraction(strong_resume_pdf):
    from parsers.pdf_parser import parse_pdf

    profile = extract_resume(parse_pdf(strong_resume_pdf))
    assert profile.name == "ANITA KRISHNAN"
    assert profile.email == "anita.krishnan@example.com"
    assert profile.phone
    assert profile.linkedin
    canon = {m.canonical for m in profile.skills}
    assert {"Python", "PyTorch", "PostgreSQL", "Docker", "FastAPI"} <= canon
    assert profile.education and profile.education[0].level == "bachelors"
    assert profile.is_fresh_grad_profile is False        # internship exists
    assert len(profile.experience) >= 1
    exp = profile.experience[0]
    assert exp.is_internship or "Intern" in exp.title


def test_weak_resume_fresh_grad_flag(weak_resume_txt):
    profile = extract_resume(ParsedDocument(
        text=weak_resume_txt.read_text(encoding="utf-8"),
        paragraphs=[weak_resume_txt.read_text(encoding="utf-8")],
        source_format="text",
    ))
    assert profile.name == "SAM DOYLE"
    assert profile.is_fresh_grad_profile is True
    assert not profile.experience
