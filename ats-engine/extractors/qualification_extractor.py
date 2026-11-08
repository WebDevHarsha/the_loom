"""Education / qualification extraction from resume text."""

from __future__ import annotations

import re

from schemas import DegreeRecord, DegreeLevel, ExperienceItem

_DEGREE_PATTERNS: list[tuple[DegreeLevel, re.Pattern]] = [
    ("phd", re.compile(r"\bph\.?d\.?\b|\bdoctoral?\b|doctorate", re.IGNORECASE)),
    ("masters", re.compile(r"\bmaster(?:'s)?\b|\bm\.?s\.?c?\b|\bm\.?tech\b|\bm\.?eng\b|post[- ]?graduate", re.IGNORECASE)),
    ("bachelors", re.compile(r"\bbachelor(?:'s)?\b|\bb\.?s\.?c?\b|\bb\.?e\.?\b|\bb\.?tech\b|\bb\.?eng\b|under[- ]?graduate", re.IGNORECASE)),
    ("associate", re.compile(r"\bassociate(?:'s)?\b|\bdiploma\b", re.IGNORECASE)),
]

_IN_FIELD_RE = re.compile(r"\b(?:in|of)\s+([A-Za-z][\w &\-/]{2,50})", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_INSTITUTION_HINTS = re.compile(
    r"university|college|institute|iit|nit|iiit|school of|academy|polytechnic", re.IGNORECASE
)

_CERT_LINE_MIN = 4


def extract_education(text: str) -> list[DegreeRecord]:
    """Parse degree lines from an EDUCATION section."""
    if not text or not text.strip():
        return []
    records: list[DegreeRecord] = []
    current: DegreeRecord | None = None
    for line in text.splitlines():
        line = line.strip(" \t•-|")
        if not line:
            continue
        level = _level_of(line)
        if level:
            field = ""
            m = _IN_FIELD_RE.search(line)
            if m and not _INSTITUTION_HINTS.search(m.group(1)):
                field = m.group(1).strip()
            year_match = _YEAR_RE.search(line)
            current = DegreeRecord(
                level=level,
                field=field,
                institution=_institution_on(line),
                grad_year=int(year_match.group(1)) if year_match else None,
            )
            records.append(current)
            continue
        if current is not None:
            inst = _institution_on(line)
            if inst and not current.institution:
                current.institution = inst
            elif not current.field and not _INSTITUTION_HINTS.search(line):
                current.field = line
    return records


def _level_of(line: str) -> DegreeLevel | None:
    for level, rx in _DEGREE_PATTERNS:
        if rx.search(line):
            return level
    return None


def _institution_on(line: str) -> str:
    if _INSTITUTION_HINTS.search(line):
        # institution usually the part before/beside the hint words
        if ", " in line:
            parts = [p.strip() for p in line.split(",")]
            for p in parts:
                if _INSTITUTION_HINTS.search(p):
                    return p.strip(" .|-")
        return line.strip(" .|-")
    return ""


def extract_certifications(text: str) -> list[str]:
    """Each meaningful line of a CERTIFICATIONS section is a certification."""
    if not text or not text.strip():
        return []
    certs: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip(" \t•-*|")
        if len(cleaned) >= _CERT_LINE_MIN and not re.match(r"^(page|resume|cv)\b", cleaned, re.I):
            certs.append(cleaned)
    return certs
