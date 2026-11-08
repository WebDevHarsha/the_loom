"""Resume structure / ATS-parseability analysis (spec section 11).

Checklist-style deductions. Explicitly conservative: different ATS products
parse differently, so findings are phrased as risks, never as guarantees.
"""

from __future__ import annotations

import re

from config import GARBLE_RATIO_LIMIT
from schemas import CategoryScore, ParsedDocument, ResumeProfile

_SPACING = "\n"


def evaluate_structure(parsed: ParsedDocument, profile: ResumeProfile) -> CategoryScore:
    issues: list[str] = []
    notes: list[str] = []
    deduction = 0.0

    def deduct(points: float, message: str, *, is_issue: bool = True) -> None:
        nonlocal deduction
        deduction += points
        (issues if is_issue else notes).append(message)

    # 1) parseable sections
    sections = set(profile.sections)
    for needed in ("education", "skills"):
        if needed not in sections:
            deduct(15, f"No clear '{needed.capitalize()}' section heading found — "
                        "ATS parsers rely on standard labels.")
    if not ({"experience", "projects"} & sections):
        deduct(15, "Neither an experience nor a projects section heading was found.")

    # 2) contact info reachable
    if not profile.email:
        deduct(10, "No parseable email address detected.")
    if not profile.phone:
        deduct(5, "No parseable phone number detected.")
    if not profile.name:
        deduct(5, "Could not identify a candidate name near the top of the resume.")

    # 3) page count sanity
    if parsed.page_count == 0:
        deduct(10, "PDF reports zero pages.")
    elif parsed.page_count > 3:
        deduct(5, f"Long document ({parsed.page_count} pages); "
                   "very long resumes sometimes get truncated by strict ATS filters.")

    # 4) bullet consistency
    bullet_lines = [b for b in profile.bullets if len(b.split()) >= 5]
    if bullet_lines and not profile.bullets:
        deduct(5, "Long paragraphs without any bulleted lines; bullet points parse more reliably.")

    # 5) layout hazards
    meta = parsed.meta or {}
    if meta.get("tables"):
        deduct(8, f"Content placed inside tables ({meta['tables']} detected) — "
                   "many ATS parsers scramble table order.")
    if meta.get("multi_column"):
        deduct(8, "Multi-column layout detected — a classic ATS reading-order hazard.")
    if parsed.scanned:
        deduct(35, "Resume looks scanned/image-based: text-based ATS parsers would see almost nothing.")
    else:
        garble = _garble_ratio(parsed.text)
        if garble > GARBLE_RATIO_LIMIT:
            deduct(
                10,
                f"Extraction produced {garble:.0%} non-printable/odd characters; "
                "fonts or encoding may garble ATS text extraction.",
            )

    score = max(0.0, 100.0 - deduction)
    explanations = []
    if deduction == 0:
        explanations.append("No structural hazards detected.")
    else:
        explanations.extend(issues[:12])
    explanations.extend(notes[:3])

    return CategoryScore(
        name="Resume Structure",
        score=round(score, 1),
        weight=0.0,
        explanations=explanations,
        details={"issues": issues, "notes": notes},
    )


def _garble_ratio(text: str) -> float:
    if not text:
        return 1.0
    bad = sum(1 for ch in text if not (ch.isprintable() and (ch.isascii() or ch in "\n\t")))
    return bad / len(text)
