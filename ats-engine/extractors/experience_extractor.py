"""Work-experience and project section parsing (rule based)."""

from __future__ import annotations

import re

from schemas import ExperienceItem

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_DATE_RANGE_RE = re.compile(
    r"(?P<m1>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?[a-z]*\.?\s*"
    r"(?P<y1>(?:19|20)\d{2})\s*"
    r"(?:-|–|—|to)\s*"
    r"(?P<m2>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)?[a-z]*\.?\s*"
    r"(?P<y2>(?:19|20)\d{2}|present|current|now)",
    re.IGNORECASE,
)
_HAS_RANGE = re.compile(
    r"(?:19|20)\d{2}\s*(?:-|–|—|to)\s*((?:19|20)\d{2}|present|current|now)",
    re.IGNORECASE,
)
_HAS_MONTH = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.IGNORECASE)
_HAS_CURRENT = re.compile(r"\bpresent\b|\bcurrent\b|\bnow\b", re.IGNORECASE)


def _datey(line: str) -> bool:
    """True when the line contains an employment date range or month name."""
    return bool(_HAS_RANGE.search(line) or _HAS_MONTH.search(line))


def split_experience(text: str) -> list[ExperienceItem]:
    """Split an EXPERIENCE section into items keyed by date-range headers."""
    return _split_items(text, is_project=False, require_date=True)


def split_projects(text: str) -> list[ExperienceItem]:
    """Split a PROJECTS section into items keyed by non-bullet title lines."""
    return _split_items(text, is_project=True, require_date=False)


def _split_items(text: str, *, is_project: bool, require_date: bool) -> list[ExperienceItem]:
    if not text or not text.strip():
        return []
    items: list[ExperienceItem] = []
    current: ExperienceItem | None = None
    marker = re.compile(r"^(?:[-–—•*·]|\d+[.)])\s+")
    for raw_line in text.splitlines():
        raw = raw_line.strip()
        if not raw:
            continue
        is_marker_bullet = bool(marker.match(raw))
        line = re.sub(marker, "", raw)
        if require_date:
            headerish = (
                not is_marker_bullet
                and len(line) <= 150
                and bool(_HAS_RANGE.search(line) or _HAS_MONTH.search(line))
            )
        else:
            headerish = (
                not is_marker_bullet
                and len(line.split()) <= 16
                and len(line) <= 110
            )
        if headerish:
            current = _make_item(line, is_project=is_project)
            items.append(current)
            continue
        if is_marker_bullet and not headerish:
            if current is None:
                current = ExperienceItem(is_project=is_project)
                items.append(current)
            _absorb(current, line, is_project=is_project)
            continue
        if current is not None:
            _absorb(current, line, is_project=is_project)
        else:
            current = ExperienceItem(is_project=is_project)
            _absorb(current, line, is_project=is_project)
            items.append(current)
    filtered = [i for i in items if i.bullets or i.start or i.end or i.is_internship]
    if not filtered:
        filtered = items          # degenerate input: keep what we have
    return filtered


def _make_item(line: str, *, is_project: bool) -> ExperienceItem:
    start = end = ""
    m = _DATE_RANGE_RE.search(line)
    if m:
        m1 = MONTHS.get((m.group("m1") or "").lower())
        m2 = MONTHS.get((m.group("m2") or "").lower())
        y1 = m.group("y1")
        y2 = m.group("y2")
        start = f"{m1:02d}/{y1}" if m1 and y1 else (y1 or "")
        if y2 and y2.isdigit():
            end = f"{m2:02d}/{y2}" if m2 else y2
        else:
            end = "present"
    left = line
    if m:
        left = (line[: m.start()] + line[m.end():]).strip(" ,|-–—")
    # "Title — Company" / "Title | Company" / "Title at Company"
    title, org = left, ""
    sep = re.split(r"\s*(?:\||—|–|:|,| at |\bwith\b)\s*", left, maxsplit=1)
    if len(sep) == 2 and len(sep[0].split()) <= 8:
        title, org = sep[0].strip(" .|—– "), sep[1].strip(" .|—– ")
    return ExperienceItem(
        title=title,
        organization=org,
        start=start,
        end=end,
        is_project=is_project,
        is_internship=("intern" in line.lower()),
    )


def _absorb(item: ExperienceItem, line: str, *, is_project: bool) -> None:
    is_bullet = bool(re.match(r"^(?:[-–—•*·]\s+)", line)) or (len(line.split()) >= 6)
    if is_bullet:
        item.bullets.append(line)
        if re.search(r"\d", line):
            item.quantified = True
    else:
        item.title = f"{item.title} {line}".strip() if item.title else line


def compute_total_years(items: list[ExperienceItem]) -> float | None:
    """Sum employment spans, merging nothing (crude but conservative):
    overlapping ranges are only counted once via a union of month indices.
    Returns None when no dates could be parsed at all."""
    spans: set[int] = set()
    parsed_any = False
    for item in items:
        if item.is_project or not item.start or not item.end:
            continue
        try:
            start = _to_month_index(item.start)
            end = _to_month_index(item.end, present=True)
        except ValueError:
            continue
        parsed_any = True
        spans.update(range(start, min(end, start + 600) + 1))
    if not parsed_any:
        return None
    return round(len(spans) / 12.0, 1)


def _to_month_index(value: str, *, present: bool = False) -> int:
    if present and re.search(r"present|current|now", value, re.IGNORECASE):
        # treat as 'now'; caller bounds short spans anyway
        import datetime
        today = datetime.date.today()
        return today.year * 12 + (today.month - 1)
    m = re.match(r"(?:(\d{2})/)?((?:19|20)\d{2})", value.strip())
    if not m:
        raise ValueError(value)
    month = int(m.group(1)) if m.group(1) else 1
    year = int(m.group(2))
    return year * 12 + (month - 1)
