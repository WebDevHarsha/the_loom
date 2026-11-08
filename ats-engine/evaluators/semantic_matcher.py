"""Semantic matching engine (spec section 8).

Matches *individual* JD requirements/responsibilities against individual
resume evidence units (not whole-document vs whole-document).

IMPORTANT: a high cosine similarity never proves a skill is present on its
own. This module only surfaces relevant resume evidence and contributes the
"Semantic Relevance" category; skill possession is decided exclusively in
the keyword matcher.
"""

from __future__ import annotations

import re

import numpy as np

from config import SEMANTIC_CEILING, SEMANTIC_FLOOR
from schemas import CategoryScore, JobDescription, ParsedDocument, ResumeProfile

MAX_SEGMENTS = 220          # cap resume evidence units for CPU speed
MAX_SEG_CHARS = 512


def remap_similarity(cos):
    """Map cosine similarity into [0, 100] with neutral floor/ceiling anchors.

    Works on scalars and numpy arrays alike."""
    cos = np.clip(np.asarray(cos, dtype=float), 0.0, 1.0)
    if cos.shape == ():
        cos_val = float(cos)
        if cos_val <= SEMANTIC_FLOOR:
            return 0.0
        if cos_val >= SEMANTIC_CEILING:
            return 100.0
        return 100.0 * (cos_val - SEMANTIC_FLOOR) / (SEMANTIC_CEILING - SEMANTIC_FLOOR)
    mapped = 100.0 * (cos - SEMANTIC_FLOOR) / (SEMANTIC_CEILING - SEMANTIC_FLOOR)
    return np.where(cos <= SEMANTIC_FLOOR, 0.0, np.minimum(mapped, 100.0))


_CONTACTISH = re.compile(r"@\w+\.\w+|linkedin\.com|github\.com|\b\d{2,4}[-\s]?\d{3,8}\b")


def _segment_resume(parsed: ParsedDocument, profile: ResumeProfile) -> list[str]:
    """Meaningful resume evidence units: bullets and sentences.

    Contact/header lines are excluded - they are not evidence."""
    segments: list[str] = []
    seen: set[str] = set()
    candidates = list(profile.bullets)
    if not candidates:
        candidates = parsed.paragraphs
    for para in parsed.paragraphs:
        parts = re.split(r"(?<=[.!?])\s+", para)
        candidates.extend(p for p in parts if len(p.split()) >= 4)
    for s in candidates:
        s = re.sub(r"\s+", " ", s).strip()
        if not s or _CONTACTISH.search(s):
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        segments.append(s[:MAX_SEG_CHARS])
        if len(segments) >= MAX_SEGMENTS:
            break
    return segments


def _jd_items(jd: JobDescription) -> list[tuple[str, str]]:
    """JD items to compare: responsibilities first, then requirement phrases.

    Duplicates are removed (a skill's raw bullet is often the same line as a
    responsibility bullet)."""
    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for r in jd.responsibilities:
        text = re.sub(r"\s+", " ", r).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            items.append((text, "responsibility"))
    for req in jd.all_skills():
        label = req.raw_text.strip() or req.canonical
        if len(label.split()) < 3:
            label = req.canonical
        text = re.sub(r"\s+", " ", label)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            items.append((text, f"skill:{req.canonical}"))
    return items[:60]


def evaluate_semantic(
    jd: JobDescription, parsed: ParsedDocument, profile: ResumeProfile
) -> CategoryScore:
    """Semantic relevance of resume evidence to JD items."""
    resume_segments = _segment_resume(parsed, profile)
    jd_items = _jd_items(jd)
    if not resume_segments or not jd_items:
        return CategoryScore(
            name="Semantic Relevance",
            score=0.0,
            weight=0.0,
            degraded=True,
            explanations=["Not enough segmented text on one side to compare."],
        )

    # import is deferred: --no-semantic callers never load torch
    from models import semantic_model

    jd_texts = [t for t, _k in jd_items]
    jd_emb = semantic_model.encode(jd_texts)
    seg_emb = semantic_model.encode(resume_segments)
    sims = semantic_model.cosine_matrix(jd_emb, seg_emb)

    per_item: list[dict] = []
    scores: list[float] = []
    for i, (text, kind) in enumerate(jd_items):
        best_j = int(np.argmax(sims[i]))
        best_sim = float(sims[i, best_j])
        score = remap_similarity(best_sim)
        scores.append(score)
        per_item.append({
            "jd_text": text[:200],
            "kind": kind,
            "similarity": round(best_sim, 4),
            "score": round(score, 1),
            "best_evidence": resume_segments[best_j][:200],
        })
        if len(per_item) >= 14:
            break

    mean_score = float(np.mean(scores))
    weak = sorted(per_item, key=lambda x: x["score"])[:3]

    explanations = [
        f"Average best-match similarity of JD items against resume evidence: "
        f"{mean_score:.1f}/100 across {len(jd_texts)} items."
    ]
    for w_item in weak:
        if w_item["score"] < 35:
            explanations.append(
                f"Low relevance for \"{w_item['jd_text'][:80]}\" "
                f"(best similarity {w_item['similarity']:.2f})."
            )

    return CategoryScore(
        name="Semantic Relevance",
        score=round(mean_score, 1),
        weight=0.0,
        explanations=explanations,
        details={"per_item": per_item},
    )
