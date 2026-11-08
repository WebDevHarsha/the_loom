"""Experience relevance evaluation (spec section 9).

Fresh-grad friendly: internships, personal and academic projects act as
first-class evidence. When the semantic model is unavailable, a keyword
overlap fallback keeps the category usable (marked degraded).
"""

from __future__ import annotations

import re

import numpy as np

from evaluators.semantic_matcher import remap_similarity
from schemas import (
    CategoryScore,
    JobDescription,
    ResumeProfile,
)

_NON_TECH_CATEGORIES = {"soft_skills"}
_QUANT_HINT = re.compile(
    r"(\d+%|\d+x|\b\d+(?:\.\d+)?\s*(?:users|clients|students|records|ms|s\b|min|hours?|days?|"
    r"points?|marks?|rank|thousand|million|k\b|m\b|gb|mb|tb))",
    re.IGNORECASE,
)


def _relevance_scores(item_texts: list[str], jd_texts: list[str]) -> np.ndarray | None:
    """Semantic best-match per item vs JD; None -> caller must fall back."""
    try:
        from models import semantic_model
        from models.semantic_model import SemanticModelUnavailable
        it = semantic_model.encode(item_texts)
        jt = semantic_model.encode(jd_texts)
        return semantic_model.cosine_matrix(it, jt)
    except Exception:
        return None


def _token_overlap(item_text: str, jd_text: str) -> float:
    """Lexical fallback similarity; Jaccard over content words."""
    stop = {
        "the", "and", "for", "with", "a", "an", "of", "to", "in", "on", "is",
        "are", "as", "at", "or", "by", "be", "your", "our", "you", "their",
        "this", "that", "will", "work", "using", "use", "used", "roles?",
    }
    toks = lambda s: {
        w for w in re.findall(r"[a-z0-9+.#/\-]{2,}", s.lower()) if w not in stop
    }
    a, b = toks(item_text), toks(jd_text)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def evaluate_experience(
    jd: JobDescription, profile: ResumeProfile
) -> CategoryScore:
    items = [*(profile.experience or []), *(profile.projects or [])]
    item_texts = [i.text() for i in items if i.text().strip()]

    jd_evidence_texts = list(jd.responsibilities) or [
        req.raw_text.strip() or req.canonical for req in jd.all_skills()
    ]
    if not item_texts or not jd_evidence_texts:
        return CategoryScore(
            name="Experience Relevance",
            score=0.0,
            weight=0.0,
            degraded=True,
            explanations=[
                "No experience or project entries could be parsed from the resume."
                if not item_texts
                else "No responsibilities found in the JD to compare against."
            ],
        )

    jd_text = " ".join(jd_evidence_texts)

    # 1) relevance of each item to the JD
    sims = _relevance_scores(item_texts, jd_evidence_texts)
    semantic_used = sims is not None
    if sims is not None:
        best = np.max(sims, axis=1)
        relevance = float(np.mean(remap_similarity(best)))
    else:
        relevance = 100.0 * float(np.mean([
            _token_overlap(t, jd_text) for t in item_texts
        ]))

    # 2) technologies actually used inside experience/project text
    from evaluators.keyword_matcher import scan_for_terms
    from extractors import skill_extractor

    tech_reqs = [
        r for r in jd.all_skills()
        if (r.category or "") != "soft_skills"
    ]
    used_set: set[str] = set()
    for req in tech_reqs:
        aliases = [req.canonical, *skill_extractor.aliases_for(req.canonical)]
        if any(scan_for_terms(aliases, text) for text in item_texts):
            used_set.add(req.canonical)
    tech_coverage = (
        len(used_set) / len(tech_reqs) if tech_reqs else 0.0
    )

    # 3) quantified achievements
    quantified_items = sum(
        1 for i in items if i.quantified or any(_QUANT_HINT.search(b) for b in i.bullets)
    )
    quantified_ratio = quantified_items / len(items)

    composite = 0.55 * relevance + 0.30 * (tech_coverage * 100.0) + 0.15 * (quantified_ratio * 100.0)

    explanations = [
        f"Relevance of experience/project content to this JD: {relevance:.0f}/100 "
        f"(semantic model {'used' if semantic_used else 'unavailable - keyword-overlap fallback'}).",
        f"JD technologies evidenced inside experience/projects: "
        f"{len(used_set)}/{len(tech_reqs)} ({', '.join(sorted(used_set)[:8]) or 'none'}).",
        f"Quantified bullets: {quantified_items}/{len(items)} "
        f"({'meets' if quantified_ratio >= 0.3 else 'below'} the 30% target).",
    ]
    if profile.is_fresh_grad_profile:
        explanations.append(
            "No formal employment found; projects/internships were used as "
            "first-class evidence (no penalty applied for missing work history)."
        )

    return CategoryScore(
        name="Experience Relevance",
        score=round(min(100.0, composite), 1),
        weight=0.0,
        explanations=explanations,
        degraded=not semantic_used,
        details={
            "tech_used": sorted(used_set),
            "items": [
                {"title": i.title, "bullets": len(i.bullets), "quantified": i.quantified}
                for i in items[:10]
            ],
        },
    )
