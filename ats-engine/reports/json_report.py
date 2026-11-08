"""JSON report rendering."""

from __future__ import annotations

import json

from schemas import ATSReport


def render_dict(report: ATSReport) -> dict:
    """Full structured report as a plain dict (pydantic v2)."""
    return report.model_dump(mode="json")


def render_json(report: ATSReport, indent: int = 2) -> str:
    return json.dumps(render_dict(report), indent=indent, ensure_ascii=False)


def save(report: ATSReport, path, indent: int = 2) -> None:
    path.write_text(render_json(report, indent), encoding="utf-8")
