"""Weighted scoring engine (spec section 13).

No score inflation: every category starts from evidence-based sub-scores;
a resume with no evidence for mandatory requirements cannot score high on
keywords, and unknown qualifications are scored as Unknown (55), not pass.
"""

from __future__ import annotations

from scoring.weights import normalize
from schemas import ATSReport, CategoryScore, OverallScore, SkillMatch

_CATEGORIES_ORDER = (
    "Keyword Matching",
    "Semantic Relevance",
    "Experience Relevance",
    "Qualifications",
    "Resume Structure",
    "Content Quality",
)


def score_categories(
    categories: list[CategoryScore],
    weights: dict[str, float] | None = None,
) -> OverallScore:
    """Weighted average over the categories present; assigns weights onto them.

    Degraded categories (evaluator could not fully run) contribute their score
    but reduce the overall confidence indicator.
    """
    weights = normalize(weights)
    scored = [c for c in categories if c.score is not None]
    total_weight = sum(weights.get(c.name, 0.0) for c in scored)
    if total_weight <= 0:
        return OverallScore(score=0.0, confidence=0.0,
                            explanations=["No category produced a score."])

    overall = sum(c.score * weights.get(c.name, 0.0) for c in scored) / total_weight
    n_degraded = sum(1 for c in scored if c.degraded)
    confidence = 1.0 - (0.15 * n_degraded / max(1, len(scored)))

    explanations: list[str] = []
    ranked = sorted(scored, key=lambda c: c.score)
    weakest = ranked[:2]
    for c in weakest:
        if c.score < 50:
            explanations.append(
                f"'{c.name}' is the weakest area ({c.score:.0f}/100): "
                + (c.explanations[0] if c.explanations else "see category details.")
            )
    if n_degraded:
        explanations.append(
            f"{n_degraded} evaluator(s) ran in degraded mode; treat the overall "
            "score with reduced confidence."
        )

    for c in scored:
        c.weight = weights.get(c.name, 0.0)

    return OverallScore(
        score=round(overall, 1),
        confidence=round(max(0.0, confidence), 2),
        explanations=explanations,
    )


def ordered_categories(categories: list[CategoryScore]) -> list[CategoryScore]:
    by_name = {c.name: c for c in categories}
    ordered = [by_name.pop(name) for name in _CATEGORIES_ORDER if name in by_name]
    ordered.extend(by_name.values())
    return ordered


def build_report(
    categories: list[CategoryScore],
    matches: list[SkillMatch],
    *,
    relevant_experience: list[str],
    issues: list[str],
    recommendations: list[str],
    meta: dict | None = None,
) -> ATSReport:
    """Assemble the final ATSReport from evaluated categories."""
    overall = score_categories(categories)
    matched = sorted({m.requirement.canonical for m in matches if m.status == "matched"})
    partial = sorted({m.requirement.canonical for m in matches if m.status == "partial"})
    missing = sorted({m.requirement.canonical for m in matches if m.status == "missing"})

    return ATSReport(
        overall=overall,
        categories=ordered_categories(categories),
        matched_skills=matched,
        partially_matched_skills=partial,
        missing_skills=missing,
        skill_matches=matches,
        relevant_experience=relevant_experience[:8],
        issues=issues,
        recommendations=recommendations[:10],
        meta=meta or {},
    )
