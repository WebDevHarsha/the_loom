"""Qualification evaluation (spec section 10).

Every requirement resolves to satisfied / not_satisfied / unknown.
Missing information is NEVER treated as proof of failure.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from schemas import (
    CategoryScore,
    DegreeLevel,
    JobDescription,
    ResumeProfile,
)

_LEVEL_RANK: dict[DegreeLevel, int] = {
    "associate": 1,
    "bachelors": 2,
    "masters": 3,
    "phd": 4,
    "other": 0,
}

_SCORES = {"satisfied": 100.0, "unknown": 55.0, "not_satisfied": 0.0}


def evaluate_qualifications(jd: JobDescription, profile: ResumeProfile) -> CategoryScore:
    results: list[dict] = []

    for q in jd.qualifications:
        status, explanation = _evaluate_one(q, profile)
        results.append({
            "description": q.description[:200],
            "kind": q.kind,
            "status": status,
            "explanation": explanation,
        })

    if not results:
        return CategoryScore(
            name="Qualifications",
            score=0.0,
            weight=0.0,
            degraded=True,
            explanations=["No degree/experience/certification requirements detected in the JD."],
        )

    raw_scores = [_SCORES[r["status"]] for r in results]
    score = sum(raw_scores) / len(raw_scores)

    explanations = [
        f"{r['kind'].replace('_', ' ').capitalize()}: {r['status'].replace('_', ' ')} — {r['explanation']}"
        for r in results[:10]
    ]
    n_unknown = sum(1 for r in results if r["status"] == "unknown")
    if n_unknown:
        explanations.append(
            f"{n_unknown} requirement(s) could not be verified from the resume "
            "and are treated as Unknown, not as failure."
        )

    return CategoryScore(
        name="Qualifications",
        score=round(score, 1),
        weight=0.0,
        explanations=explanations,
        details={"requirements": results},
    )


def _evaluate_one(q, profile: ResumeProfile) -> "tuple[str, str]":
    if q.kind == "min_experience":
        if q.min_years is None:
            return "unknown", "Requirement phrasing could not be quantified."
        years = profile.approx_years_experience
        # Student-friendly: 0-1 year asks are satisfied by projects when the
        # resume offers strong project evidence but no formal employment.
        if years is None:
            if profile.is_fresh_grad_profile and q.min_years <= 1.0:
                return (
                    "unknown",
                    f"Asks {q.min_years:g} year(s); resume shows no dated employment "
                    "(project-based profile): assessed under Experience Relevance.",
                )
            return "unknown", "Resume does not state dated employment history."
        if years + 0.01 >= q.min_years:
            return "satisfied", f"Resume evidence covers ~{years:g} years (need {q.min_years:g}+)."
        if profile.is_fresh_grad_profile and q.min_years <= 1.0:
            return (
                "unknown",
                f"Formal experience ~{years:g}yr below {q.min_years:g}, but strong "
                "project evidence exists; judged under Experience Relevance.",
            )
        return "not_satisfied", f"Need {q.min_years:g}+ years; resume shows ~{years:g}."

    if q.kind == "degree":
        required_level = q.degree_level or "bachelors"
        if not profile.education:
            return "unknown", "No degree detected in the resume's education section."
        best_rank = max(_LEVEL_RANK[d.level] for d in profile.education)
        need_rank = _LEVEL_RANK[required_level]
        if best_rank >= need_rank:
            return "satisfied", f"Resume shows a {required_level}-level (or higher) degree."
        return "not_satisfied", (
            f"Resume's highest degree is below the required {required_level} level."
        )

    if q.kind == "certification":
        if not profile.certifications:
            return "unknown", "No certifications section found on the resume."
        for cert in profile.certifications:
            if fuzz.partial_ratio(q.description.lower()[:60], cert.lower()) >= 70:
                return "satisfied", f"Matches resume certification: \"{cert[:80]}\"."
        return "not_satisfied", "Resume certifications do not match this requirement."

    # kind == "other": conservative keyword evidence lookup
    terms = re.findall(r"[A-Za-z][\w+.#/\-]{2,}", q.description)
    flat = _profile_text(profile)
    hits = [t for t in terms if re.search(
        rf"(?<![\w&+.\-]){re.escape(t)}(?![\w&+.\-])", flat, re.IGNORECASE)]
    if hits:
        return "satisfied", f"Evidence found for: {', '.join(hits[:3])}."
    return "unknown", "Could not verify from the resume."


def _profile_text(profile: ResumeProfile) -> str:
    """Flatten structured sections back into one text blob."""
    parts: list[str] = []
    parts.extend(profile.sections.values())
    parts.extend(b for item in (*profile.experience, *profile.projects) for b in item.bullets)
    parts.extend(profile.certifications)
    parts.extend(d.field for d in profile.education)
    return "\n".join(p for p in parts if p)
