"""Keyword matching engine (spec section 7).

Strategies: exact canonical match -> alias match -> conservative fuzzy match.
Each match carries resume evidence; a similar phrase alone never proves a
skill, so fuzzy matches only ever count as *partial*.
"""

from __future__ import annotations

import re
from functools import lru_cache

from rapidfuzz import fuzz

from config import FUZZY_MIN_LEN, FUZZY_THRESHOLD
from extractors import skill_extractor
from schemas import (
    CategoryScore,
    Evidence,
    JobDescription,
    MatchType,
    ParsedDocument,
    SkillMatch,
)


@lru_cache(maxsize=8)
def _sentences(text: str) -> tuple[str, ...]:
    """Split resume text into candidate evidence units (lines+sentences)."""
    units: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        for p in parts:
            p = p.strip()
            if p and p not in units:
                units.append(p)
    units.sort(key=len, reverse=True)
    return tuple(units)


def _best_evidence(
    matched_text: str,
    sentence: str,
) -> Evidence:
    """The sentence with the best keyword-anchored evidence."""
    sentence = re.sub(r"\s+", " ", sentence).strip()
    shortened = sentence if len(sentence) <= 160 else sentence[:157] + "..."
    match_type: MatchType = "exact"
    if not re.search(re.escape(matched_text), sentence, re.IGNORECASE):
        match_type = "fuzzy"
    return Evidence(quote=shortened, match_type=match_type)


def evaluate_keywords(
    jd: JobDescription, parsed: ParsedDocument
) -> tuple[CategoryScore, list[SkillMatch]]:
    """Score keyword coverage; returns the category result plus every match."""
    resume_text = parsed.text
    units = _sentences(resume_text)
    matches: list[SkillMatch] = []

    for req in jd.all_skills():
        match = _evaluate_requirement(req, resume_text, units)
        matches.append(match)

    total_weight = 0.0
    weighted = 0.0
    for m in matches:
        w = 1.0 if m.requirement.mandatory else 0.7
        total_weight += w
        weighted += w * m.score
    score = 100.0 * weighted / total_weight if total_weight else 0.0

    n_mand = sum(1 for m in matches if m.requirement.mandatory)
    explanations = [
        f"{sum(1 for m in matches if m.status == 'matched')} of "
        f"{len(matches)} required-in-JD skills found in the resume "
        f"({n_mand} marked mandatory)."
    ]
    missing_mandatory = [
        m.requirement.canonical for m in matches
        if m.requirement.mandatory and m.status == "missing"
    ]
    if missing_mandatory:
        explanations.append(
            "Mandatory keywords with no evidence: " + ", ".join(missing_mandatory[:10])
        )
    if any(m.status == "partial" for m in matches):
        explanations.append(
            "Some skills only fuzzy-matched (close phrasing, not exact keyword); "
            "ATS keyword filters may miss them."
        )

    return (
        CategoryScore(
            name="Keyword Matching",
            score=round(score, 1),
            weight=0.0,        # set by the scoring engine
            explanations=explanations,
            details={
                "matched": sorted({m.requirement.canonical for m in matches if m.status == "matched"}),
                "partial": sorted({m.requirement.canonical for m in matches if m.status == "partial"}),
                "missing": sorted({m.requirement.canonical for m in matches if m.status == "missing"}),
            },
        ),
        matches,
    )


def _evaluate_requirement(req, resume_text: str, units: tuple[str, ...]) -> SkillMatch:
    aliases = skill_extractor.aliases_for(req.canonical)
    candidate_terms = [req.canonical, *aliases]

    best: SkillMatch | None = None

    # Pass 1: exact / alias (whole-document boundary search via taxonomy scans)
    hits = scan_for_terms(candidate_terms, resume_text)
    if hits:
        ev = [_best_evidence(h, unit_for(h, units)) for h in hits[:3]]
        best = SkillMatch(requirement=req, status="matched", score=1.0, evidence=ev)
        return best

    # Pass 2: fuzzy against evidence units (phrase-level similarity)
    fuzzy: list[tuple[float, str]] = []
    for unit in units:
        ratio = fuzz.partial_ratio(req.canonical.lower(), unit.lower())
        if ratio >= FUZZY_THRESHOLD:
            fuzzy.append((ratio, unit))
    if fuzzy and len(req.canonical) >= FUZZY_MIN_LEN:
        fuzzy.sort(reverse=True)
        ratio, unit = fuzzy[0]
        quote = unit if len(unit) <= 160 else unit[:157] + "..."
        ev = [
            Evidence(
                quote=quote,
                match_type="fuzzy",
                similarity=round(ratio / 100.0, 3),
            )
        ]
        return SkillMatch(requirement=req, status="partial", score=0.5, evidence=ev)

    return SkillMatch(requirement=req, status="missing", score=0.0)


def scan_for_terms(terms: list[str], text: str) -> list[str]:
    """Return the *matched surface forms* of `terms` found in `text`.

    Boundary-guarded, case-insensitive (case-sensitive for <=2-char terms).
    """
    found: list[str] = []
    for t in sorted(set(terms), key=len, reverse=True):
        pattern = skill_extractor._boundary(t, len(t) <= 2)
        m = pattern.search(text)
        if m:
            found.append(m.group(0))
    return found


def unit_for(surface: str, units: tuple[str, ...]) -> str:
    """The most relevant evidence unit containing the surface form."""
    for u in units:
        if surface.lower() in u.lower():
            return u
    return surface
