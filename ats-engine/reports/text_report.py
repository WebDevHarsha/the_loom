"""Human-readable text report rendering."""

from __future__ import annotations

from schemas import ATSReport

_WIDTH = 64


def _header(title: str) -> str:
    return f"{title}\n{'-' * _WIDTH}"


def _score_bar(score: float, width: int = 20) -> str:
    filled = int(round(score / 100.0 * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def render_text(report: ATSReport) -> str:
    """Human-readable evaluation report in the documented format."""
    lines: list[str] = []
    lines.append(_header("ATS EVALUATION REPORT"))
    jd = report.meta.get("jd_title") or "(job title unknown)"
    src = report.meta.get("resume_source", "")
    lines.append(f"Resume: {src}    Job: {jd}")
    lines.append("")

    confidence = report.overall.confidence
    lines.append(f"Overall Compatibility Score: {report.overall.score:.0f}/100   "
                 f"(confidence {confidence:.0%})")
    lines.append("")
    lines.append("CATEGORY BREAKDOWN")
    for cat in report.categories:
        weight_txt = f" ({cat.weight:.0f}% weight)" if cat.weight else ""
        degraded = " [degraded]" if cat.degraded else ""
        lines.append(f"- {cat.name}: {cat.score:.0f}/100  {_score_bar(cat.score)}"
                     f"{weight_txt}{degraded}")
    lines.append("")

    def _skills(title: str, names: list[str]) -> None:
        lines.append(title)
        if names:
            lines.extend(f"- {n}" for n in names)
        else:
            lines.append("- (none)")
        lines.append("")

    _skills("MATCHED SKILLS", report.matched_skills)
    _skills("PARTIALLY MATCHED SKILLS (fuzzy - add the exact keyword if true)",
            report.partially_matched_skills)
    _skills("MISSING SKILLS (no evidence found)", report.missing_skills)

    if report.relevant_experience:
        lines.append("RELEVANT EXPERIENCE")
        lines.extend(f"- {e}" for e in report.relevant_experience[:2])
        lines.append("")

    if report.categories:
        for cat in report.categories:
            if cat.name == "Qualifications":
                reqs = (cat.details or {}).get("requirements") or []
                if reqs:
                    lines.append("QUALIFICATION CHECKS")
                    for r in reqs:
                        lines.append(f"- [{str(r['status']).upper():>13}] {r['description']}")
                        lines.append(f"         -> {r['explanation']}")
                    lines.append("")
                break
        sem = next((c for c in report.categories if c.name == "Semantic Relevance"), None)
        if sem and (sem.details or {}).get("per_item"):
            lines.append("BEST-MATCH EVIDENCE (semantic)")
            for item in (sem.details["per_item"])[:5]:
                lines.append(f"- JD: \"{item['jd_text'][:90]}\"")
                lines.append(f"    sim {item['similarity']:.2f}  |  resume: \"{item['best_evidence'][:90]}\"")
            lines.append("")

    if report.issues:
        lines.append("RESUME ISSUES")
        lines.extend(f"- {i}" for i in report.issues[:10])
        lines.append("")

    if report.recommendations:
        lines.append("RECOMMENDATIONS")
        lines.extend(f"- {r}" for r in report.recommendations[:10])
        lines.append("")

    lines.append("Note: different ATS products parse documents differently; "
                 "this report is an evidence-based compatibility estimate, "
                 "not a guarantee for any specific ATS.")
    return "\n".join(lines)


def save(report: ATSReport, path) -> None:
    path.write_text(render_text(report) + "\n", encoding="utf-8")
