"""Resume information extraction.

Turns a ParsedDocument into a ResumeProfile using rule-based parsing only.
Fields that cannot be found are left as None / empty - nothing is invented.
"""

from __future__ import annotations

import re

from extractors import experience_extractor, qualification_extractor, skill_extractor
from parsers import job_parser
from schemas import ParsedDocument, ResumeProfile, SkillMention

_HEADINGS: tuple[tuple[str, re.Pattern], ...] = (
    ("summary", re.compile(r"^(professional\s+)?(summary|profile|objective|about me|overview)\s*:?$", re.I)),
    ("experience", re.compile(r"^(work\s+|professional\s+|relevant\s+)?(experience|employment|employment history|work history|internships?)\s*:?$", re.I)),
    ("education", re.compile(r"^education(al)?(\s+(qualifications?|background|details?))?\s*:?$|^academics?\s*:?$", re.I)),
    ("skills", re.compile(r"^(technical\s+|core\s+|key\s+|relevant\s+)?skills(\s+(&\s*(interests?|abilities|tools?))|(\s+and\s+tools?))?\s*:?$|^technical\s+proficienc(y|ies)\s*:?$|^technologies\s*:?$", re.I)),
    ("projects", re.compile(r"^(personal\s+|academic\s+|key\s+|selected\s+)?projects?\s*:?$", re.I)),
    ("certifications", re.compile(r"^certifications?(\s+and\s+courses?)?\s*:?$|^licenses?\s*:?$|^courses\s*:?$|^training\s*:?$", re.I)),
    ("achievements", re.compile(r"^(achievements?|awards?|honors?|honours?|accomplishments?)\s*:?$|^extracurricular\s*:?$", re.I)),
)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.\-]+")
_PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}")
_LINKEDIN_RE = re.compile(r"(?:linkedin\.com/(?:in|pub)/[\w.\-/%]+)", re.I)
_GITHUB_RE = re.compile(r"(?:github\.com/[\w.\-]+)", re.I)

BulletMarker = re.compile(r"^\s*(?:[-–—•*·]|\d+[.)])\s+")


def extract_resume(parsed: ParsedDocument) -> ResumeProfile:
    """Full rule-based extraction pipeline for one resume."""
    lines = [l.strip() for l in parsed.text.splitlines() if l.strip()]
    sections = _split_sections(lines)
    text = parsed.text

    bullets = _collect_bullets(lines, sections)
    mentions = skill_extractor.scan_skills(text)
    mentions = _augment_with_skills_section(sections.get("skills", ""), mentions)

    experience = experience_extractor.split_experience(sections.get("experience", ""))
    projects = experience_extractor.split_projects(sections.get("projects", ""))
    education = qualification_extractor.extract_education(sections.get("education", ""))
    certifications = qualification_extractor.extract_certifications(
        sections.get("certifications", ""))
    years = experience_extractor.compute_total_years(experience)

    contact = _extract_contact(lines)
    # Anyone with no dated employment entries gets the student-friendly path:
    # projects/internships/coursework are the evidence, not formal work years.
    fresh_profile = not experience

    profile = ResumeProfile(
        name=contact["name"],
        email=contact["email"],
        phone=contact["phone"],
        linkedin=contact["linkedin"],
        github=contact["github"],
        sections=sections,
        bullets=bullets,
        skills=mentions,
        experience=experience,
        projects=projects,
        education=education,
        certifications=certifications,
        approx_years_experience=years,
        is_fresh_grad_profile=fresh_profile,
        warnings=list(parsed.warnings),
    )
    return profile


def _split_sections(lines: list[str]) -> dict[str, str]:
    """Map heading-line -> section text. Content before any heading is 'header'."""
    sections: dict[str, str] = {}
    current = "header"
    sections.setdefault(current, [])
    for line in lines:
        matched = None
        for key, rx in _HEADINGS:
            if rx.match(line.rstrip(":")) and len(line) <= 60:
                matched = key
                break
        if matched:
            current = matched
            sections.setdefault(current, [])
        else:
            sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items() if v or k != "header"}


def _collect_bullets(lines: list[str], sections: dict[str, str]) -> list[str]:
    """Bulleted lines anywhere in the resume (marker bullets first)."""
    bullets = [re.sub(BulletMarker, "", l).strip() for l in lines if BulletMarker.match(l)]
    # In experience/projects sections, long capitalized sentences without a
    # marker often act as bullets too (plain-paragraph resumes).
    for key in ("experience", "projects"):
        for line in (sections.get(key, "") or "").splitlines():
            line = line.strip()
            if (
                len(line.split()) >= 6
                and not BulletMarker.match(line)
                and re.match(r"^[A-Z]", line)
                and not _is_heading_like(line)
            ):
                bullets.append(line)
    dedup: list[str] = []
    for b in bullets:
        if b and b not in dedup:
            dedup.append(b)
    return dedup


def _is_heading_like(line: str) -> bool:
    return len(line) <= 60 and bool(re.match(r"\S+[^:]*:$", line))


def _augment_with_skills_section(skills_text: str, mentions: list[SkillMention]) -> list[SkillMention]:
    """Add free-form skills-section tokens (comma-separated etc.)."""
    if not skills_text:
        return mentions
    known = {m.canonical for m in mentions}
    result = list(mentions)
    for frag in re.split(r"[,\n;/|·•]+", skills_text):
        frag = frag.strip()
        if not frag:
            continue
        canonical = skill_extractor.canonicalize(frag)
        if canonical and canonical not in known:
            known.add(canonical)
            result.append(SkillMention(
                name=frag.strip(" ."),
                canonical=canonical,
                context=frag,
            ))
    return result


def _extract_contact(lines: list[str]) -> dict[str, "str | None"]:
    """Name/email/phone/links from the resume header area."""
    head = lines[:10]
    text_head = "\n".join(head)
    email = _EMAIL_RE.search(text_head)
    phone = _PHONE_RE.search(text_head)
    linkedin = _LINKEDIN_RE.search(text_head)
    github = _GITHUB_RE.search(text_head)

    name: str | None = None
    for line in head:
        if line is None or _EMAIL_RE.search(line) or _PHONE_RE.search(line):
            continue
        if not line or job_parser.ROLE_WORDS.search(line):
            continue
        words = line.split()
        if 2 <= len(words) <= 5 and all(w.isalpha() for w in words) and not line.isupper():
            name = line.strip()
            break
        if 2 <= len(words) <= 5 and all(re.fullmatch(r"[A-Z][A-Za-z'.\-]*", w) for w in words):
            name = line.strip()
            break

    return {
        "name": name,
        "email": email.group(0) if email else None,
        "phone": phone.group(0) if phone else None,
        "linkedin": linkedin.group(0) if linkedin else None,
        "github": github.group(0) if github else None,
    }
