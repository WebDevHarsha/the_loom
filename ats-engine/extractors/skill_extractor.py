"""Skill recognition against a taxonomy + aliases, plus fuzzy canonicalization.

Matching strategies (spec section 7): exact match, alias match, fuzzy match.
Every canonical name and alias is compiled to a word-boundary-guarded regex;
short names (<=2 chars like Go/R/C) are matched case-sensitively to avoid
false positives on ordinary words.
"""

from __future__ import annotations

import bisect
import functools
import json
import re
from pathlib import Path

from rapidfuzz import fuzz, process

from config import (
    DATA_DIR,
    SKILL_ALIASES_FILE,
    SKILL_SECTION_FUZZY,
    SKILL_TAXONOMY_FILE,
)
from schemas import SkillMention

# A skill token must not be glued to word chars or connectors (+, -, &) on
# either side. '.' and '/' are allowed as neighbors: they end normal English
# sentences ("PyTorch.", "CSS/") while still matching inside names like
# "node.js" and "ci/cd", since those dots/slashes sit in the escaped core.
def _boundary(pattern_text: str, case_sensitive: bool) -> re.Pattern:
    """Compile a skill name into a boundary-guarded regex."""
    flags = 0 if case_sensitive else re.IGNORECASE | re.ASCII
    return re.compile(
        r"(?<![\w+\-])" + re.escape(pattern_text) + r"(?![\w+\-])",
        flags=flags,
    )


@functools.lru_cache(maxsize=4)
def load_taxonomy(path_str: str = str(SKILL_TAXONOMY_FILE)) -> dict[str, list[str]]:
    p = Path(path_str)
    if not p.is_file():
        raise FileNotFoundError(f"Skill taxonomy not found: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=4)
def load_aliases(path_str: str = str(SKILL_ALIASES_FILE)) -> dict[str, str]:
    p = Path(path_str)
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def canonical_names() -> tuple[str, ...]:
    tax = load_taxonomy()
    names: set[str] = set()
    for cat_skills in tax.values():
        names.update(cat_skills)
    names.update(load_aliases().values())
    return tuple(sorted(names))


@functools.lru_cache(maxsize=4)
def aliases_for(canonical: str) -> tuple[str, ...]:
    return tuple(sorted(a for a, c in load_aliases().items() if c == canonical))


@functools.lru_cache(maxsize=1)
def _compiled_patterns() -> tuple[tuple[str, str | None, re.Pattern], ...]:
    """(canonical, category, compiled_alias_pattern) triples.
    Aliases first (longest first), then the canonical name itself."""
    tax = load_taxonomy()
    category_of: dict[str, str] = {}
    for category, skills in tax.items():
        for s in skills:
            category_of.setdefault(s, category)
    triples: list[tuple[str, str | None, re.Pattern]] = []
    alias_map = load_aliases()
    for alias in sorted(alias_map, key=len, reverse=True):
        canonical = alias_map[alias]
        cs = len(canonical) <= 2 and len(alias) <= 2
        triples.append((canonical, category_of.get(canonical), _boundary(alias, cs)))
    for canonical in sorted(canonical_names(), key=len, reverse=True):
        cs = len(canonical) <= 2
        triples.append((canonical, category_of.get(canonical), _boundary(canonical, cs)))
    return tuple(triples)


def _context_for(text: str, line_starts: list[int], lines: list[str], start: int) -> str:
    """Return the full line containing the character offset `start`."""
    idx = bisect.bisect_right(line_starts, start) - 1
    return lines[idx].strip() if 0 <= idx < len(lines) else ""


def scan_skills(text: str) -> list[SkillMention]:
    """Find every taxonomy skill/alias occurrence in `text`.

    Overlapping matches are resolved longest-pattern-wins; each kept match
    records the line it appeared on as context.
    """
    lines = text.splitlines()
    # offsets of the first char of every line; separators may be \n or \r\n
    line_starts: list[int] = []
    pos = 0
    for line in lines:
        line_starts.append(pos)
        pos += len(line)
        if pos < len(text) and text[pos] == "\r":
            pos += 1
        if pos < len(text):
            pos += 1                      # the \n
    line_starts.append(len(text))

    spans: list[tuple[int, int, SkillMention]] = []
    for canonical, category, pattern in _compiled_patterns():
        for m in pattern.finditer(text):
            mention = SkillMention(
                name=m.group(0),
                canonical=canonical,
                category=category,
                context="",          # filled after de-overlapping
            )
            spans.append((m.start(), m.end(), mention))

    # Longest wins; earlier start wins for equal length.
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    kept: list[tuple[int, int, SkillMention]] = []
    last_end = -1
    for start, end, mention in spans:
        if start >= last_end:
            kept.append((start, end, mention))
            last_end = end

    for start, _end, mention in kept:
        mention.context = _context_for(text, line_starts, lines, start)
    return [m for _s, _e, m in kept]


def canonicalize(token: str) -> str | None:
    """Map a free-form fragment (e.g. from a skills section) to a canonical
    skill via exact/alias match, falling back to conservative fuzzy matching."""
    clean = re.sub(r"[\s·•]+", " ", token).strip(" .,\t;:|")
    if not clean or len(clean) > 40:
        return None
    lower = clean.lower()
    # exact canonical
    if clean in canonical_names():
        return clean
    tax = load_taxonomy()
    for category, skills in tax.items():
        for s in skills:
            if s.lower() == lower:
                return s
    aliases = load_aliases()
    if lower in aliases:
        return aliases[lower]
    # fuzzy fallback, conservative cutoff
    candidates = list(canonical_names()) + list(aliases.values())
    match = process.extractOne(clean, candidates, scorer=fuzz.WRatio, score_cutoff=SKILL_SECTION_FUZZY)
    if match:
        best = match[0]
        if best in aliases:
            return aliases[best]
        return best
    # keep unknown-but-plausible skill-section tokens as their own name
    if 2 <= len(clean) <= 30 and re.search(r"[A-Za-z]", clean):
        # skip obvious non-skill section words
        stop = {
            "interests", "hobbies", "languages", "references", "skills",
            "tools", "technologies", "certifications", "achievements",
            "activities", "projects", "experience", "responsibilities",
            "strengths", "attributes", "expertise", "profiles", "links",
        }
        if clean.lower() not in stop:
            # plausible = acronym, versioned, or multiword/capitalized
            plausible = (
                any(c in clean for c in "+#./&")
                or clean.isupper()
                or len(clean.split()) > 1
                or clean[0].isupper()
            )
            if plausible:
                return clean
    return None
