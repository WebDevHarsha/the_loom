"""Job description parser.

Splits a raw JD into sections, identifies the title/company, separates
mandatory from preferred requirements, and turns the *taxonomy-anchored*
skill mentions into structured requirements. Only curriculum taxonomy terms
become skills - ordinary JD words never count as a skill (spec section 6).
"""

from __future__ import annotations

import re

from extractors import skill_extractor
from schemas import (
    DegreeLevel,
    JobDescription,
    QualificationRequirement,
    SkillRequirement,
)

ROLE_WORDS = re.compile(
    r"engineer|developer|intern|analyst|scientist|manager|architect|consultant|"
    r"administrator|designer|specialist|associate|operations|lead|programmer|"
    r"technician|offset|recruit|executive|coordinator",
    re.IGNORECASE,
)

_PREFERRABLE_WORDING = re.compile(
    r"nice[- ]to[- ]have|preferred|good[- ]to[- ]have|pluses?|plus\b|bonus|bonus points|"
    r"familiarity with|desirable|optional|exposure to|vis-|an asset|would be",
    re.IGNORECASE,
)

_YEARS_RE = re.compile(
    r"(?P<lo>\d+(?:\.\d+)?)\s*\+?\s*(?:-|–|—|to)\s*(?P<hi>\d+(?:\.\d+)?)?\s*\+?\s*(?:years|yrs\b|year\b|yr\b)",
    re.IGNORECASE,
)
_YEARS_RE_SIMPLE = re.compile(r"(?P<lo>\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs\b|year\b|yr\b)", re.IGNORECASE)

_DEGREE_PATTERNS: list[tuple[DegreeLevel, re.Pattern]] = [
    ("phd", re.compile(r"\bph\.?d\.?\b|\bdoctoral?\b|doctorate", re.IGNORECASE)),
    ("masters", re.compile(r"\bmasters?\b|\bm\.?s\.?\b|\bm\.?tech\b|\bm\.?eng\b|post[- ]?grad", re.IGNORECASE)),
    ("bachelors", re.compile(r"\bbachelors?\b|\bb\.?s\.?c?\b|\bb\.?e\.?\b|\bb\.?tech\b|\bb\.?eng\b|undergraduate", re.IGNORECASE)),
    ("associate", re.compile(r"\bassociate(?:'s)?\b|\bdiploma\b", re.IGNORECASE)),
]

_CERT_RE = re.compile(r"certif", re.IGNORECASE)

Section = str  # one of: "required", "preferred", "skills", "responsibilities", "about_role", "about", "other"

# order matters (more specific first)
_HEADING_RES: list[tuple[Section, re.Pattern]] = [
    ("responsibilities", re.compile(
        r"^(key\s+)?responsibilities\s*:?$|^(what you will (own|do|be doing)|what you'?ll (own|do)|your responsibilities|job duties|duties|the role|role & responsibilities)\s*:?", re.I)),
    ("preferred", re.compile(
        r"^(preferred|nice[- ]to[- ]haves?|good[- ]to[- ]have|good to have|pluses?|plus(es)?|bonus(es)?|bonus points|desirable|additional skills?|preferred qualifications?|preferred skills?)\s*:?", re.I)),
    ("required", re.compile(
        r"^(required|requirements|qualifications|key qualifications|minimum qualifications|basic qualifications|must[- ]haves?|must have|essential|skills (and|&) (experience|qualifications)|what we'?re looking for|what you'?ll need|what we look for|you have|about you|mandatory skills?)\s*:?", re.I)),
    ("skills", re.compile(r"^(technical skills?|skills? required|required skills?|core skills?)\s*:?", re.I)),
    ("about_role", re.compile(r"^(about the role|role overview|position summary|job summary|about the team|about this (position|role)|about the opportunity)\s*:?", re.I)),
    ("about", re.compile(r"^(about .{1,80}|company (overview|profile|background)|who we are)\s*:?", re.I)),
]

_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-–—•*·]|\d+[.)])\s+")


def _normalize_lines(raw_text: str) -> list[str]:
    """Strip bullet markers, trim trailing spaces; keep order."""
    lines: list[str] = []
    for line in raw_text.splitlines():
        stripped = re.sub(r"^\s*(?:[-–—•·*]|\d+[.)])\s*", "", line.strip())
        if not re.fullmatch(r"[-–—_*•·]{2,}", line.strip()):
            if stripped:
                lines.append(stripped)
    return lines


def _is_heading(line: str) -> Section | None:
    if len(line) > 70 or not line.strip():
        return None
    probe = line.rstrip(":").strip()
    for section, rx in _HEADING_RES:
        if rx.match(probe):
            return section
    if probe.lower().startswith("company ") or "experience:" == probe.lower():
        # "Company: X" handled elsewhere; not a section marker on its own
        return None
    return None


def _classify_context(section: Section, bullet_text: str) -> "tuple[bool, bool]":  # (mandatory, preferred)
    """Classify one requirement context as mandatory / preferred."""
    if _PREFERRABLE_WORDING.search(bullet_text):
        return False, True
    if section == "preferred":
        return False, True
    if section in {"required", "skills", "responsibilities", "about_role"}:
        return True, False
    return False, False        # unknown -> treated by full-text fallback


def parse_job(raw_text: str) -> JobDescription:
    """Parse a job description into structured requirements."""
    lines = _normalize_lines(raw_text)
    title, company = _find_title_and_company(lines)
    current: Section | None = None
    section_bullets: dict[Section, list[str]] = {}
    unsectioned: list[str] = []

    for line in lines:
        heading = _is_heading(line)
        if heading:
            current = heading
            continue
        if _is_requirement_line(line):
            if current:
                section_bullets.setdefault(current, []).append(line)
            else:
                unsectioned.append(line)

    responsibilities = list(section_bullets.get("responsibilities", []))
    requirement_contexts: list[tuple[str, Section]] = [
        (b, sec)
        for sec, bullets in section_bullets.items()
        if sec in {"required", "skills", "responsibilities", "about_role"}
        for b in bullets
    ] + [(b, "preferred") for b in section_bullets.get("preferred", [])]
    if not requirement_contexts:
        requirement_contexts = [(b, "required") for b in unsectioned]

    warnings: list[str] = []
    if not section_bullets:
        warnings.append(
            "JD has no recognizable requirement headings; extracted skills come "
            "from the whole text using wording heuristics (mandatory/preferred split may be imprecise)."
        )

    # Collect each skill's context information in one pass over requirement text.
    tax = skill_extractor.load_taxonomy()
    category_of: dict[str, str] = {}
    for cat, names in tax.items():
        for s in names:
            category_of[s] = cat

    info: dict[str, dict] = {}          # canonical -> {"mandatory": bool, "raw": str}
    for text, section in requirement_contexts:
        mand, _pref = _classify_context(section, text)
        for mention in skill_extractor.scan_skills(text):
            entry = info.setdefault(mention.canonical, {"mandatory": False, "raw": ""})
            if mand:
                entry["mandatory"] = True
            if not entry["raw"] and mention.context:
                entry["raw"] = mention.context

    # Whole-text pass for skills that were only mentioned outside requirement
    # bullets (their wording decides mandatory vs preferred).
    for mention in skill_extractor.scan_skills(raw_text):
        if mention.canonical in info or not mention.context:
            continue
        m_ctx = _PREFERRABLE_WORDING.search(mention.context)
        info[mention.canonical] = {"mandatory": not bool(m_ctx), "raw": mention.context}

    required_skills, preferred_skills = [], []
    for canonical in sorted(info):
        entry = info[canonical]
        req = SkillRequirement(
            canonical=canonical,
            category=category_of.get(canonical),
            raw_text=entry["raw"] or text_placeholder(entry),
            mandatory=entry["mandatory"],
        )
        (required_skills if entry["mandatory"] else preferred_skills).append(req)

    qualifications = _extract_qualifications(requirement_contexts)

    return JobDescription(
        raw_text=raw_text.strip(),
        title=title,
        company=company,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        responsibilities=responsibilities or [b for b, s in requirement_contexts if s == "about_role"],
        qualifications=qualifications,
        warnings=warnings,
    )


def mention_observed(text: str, canonical: str) -> bool:
    """True when `canonical` appears in `text` (canonical or alias form)."""
    return any(m.canonical == canonical for m in skill_extractor.scan_skills(text))


def text_placeholder(entry: dict) -> str:
    """Fallback raw text for a skill whose whole-text context was empty."""
    return entry.get("raw", "")


def _is_requirement_line(line: str) -> bool:
    return len(line) >= 3


def _find_title_and_company(lines: list[str]) -> "tuple[str | None, str | None]":
    title: str | None = None
    company: str | None = None
    for line in lines[:6]:
        if line and ROLE_WORDS.search(line) and len(line) <= 80:
            title = line.strip()
            break
    company_rx = re.compile(r"^company\s*[:\-]\s*(.+)$", re.IGNORECASE)
    for line in lines:
        m = company_rx.match(line)
        if m:
            company = m.group(1).strip()
            break
    if not company:
        # Heading "About <Company>" near the top; skip About-the-role variants.
        skip_starts = {"the", "us", "you", "your", "our", "this", "myself", "me"}
        for line in lines[:40]:
            m = re.match(r"^about\s+([A-Z][\w&.,'\- ]{1,60})", line, re.IGNORECASE)
            if not m:
                continue
            candidate = m.group(1).strip().rstrip(".,")
            if candidate.split()[0].lower() in skip_starts:
                continue
            company = candidate
            break
    return title, company


def _extract_qualifications(requirement_contexts: list[tuple[str, str]]) -> list[QualificationRequirement]:
    quals: list[QualificationRequirement] = []
    seen: set[str] = set()

    def add(q: QualificationRequirement) -> None:
        key = (q.kind, q.description)
        if key not in seen:
            seen.add(key)
            quals.append(q)

    for text, _section in requirement_contexts:
        m = _YEARS_RE.search(text) or _YEARS_RE_SIMPLE.search(text)
        if m:
            lo = float(m.group("lo"))
            add(QualificationRequirement(
                kind="min_experience",
                description=text,
                min_years=lo,
            ))
            continue

        level = _degree_level(text)
        if level:
            add(QualificationRequirement(
                kind="degree",
                description=text,
                degree_level=level,
            ))
            continue

        if _CERT_RE.search(text):
            add(QualificationRequirement(
                kind="certification",
                description=text,
            ))
    return quals


def _degree_level(text: str) -> str | None:
    for level, rx in _DEGREE_PATTERNS:
        if rx.search(text):
            return level
    return None
