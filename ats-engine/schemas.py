"""Structured data models shared by all engine modules (pydantic v2)."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

MatchType = Literal["exact", "alias", "fuzzy", "semantic", "none"]
SkillStatus = Literal["matched", "partial", "missing"]
QualStatus = Literal["satisfied", "not_satisfied", "unknown"]
DegreeLevel = Literal["phd", "masters", "bachelors", "associate", "other"]


class SkillMention(BaseModel):
    """A skill located inside some text."""
    name: str                      # exactly as it appeared
    canonical: str                 # taxonomy canonical name
    category: Optional[str] = None
    context: str = ""              # sentence / bullet where it was found


class SkillRequirement(BaseModel):
    """One skill the job asks for."""
    canonical: str
    raw_text: str = ""             # bullet/phrase where the JD asks for it
    mandatory: bool = True
    category: Optional[str] = None


class Evidence(BaseModel):
    quote: str = ""
    match_type: MatchType = "none"
    similarity: Optional[float] = None


class SkillMatch(BaseModel):
    requirement: SkillRequirement
    status: SkillStatus
    score: float = 0.0             # 0..1 confidence the skill is evidenced
    evidence: list[Evidence] = Field(default_factory=list)


class ExperienceItem(BaseModel):
    title: str = ""
    organization: str = ""
    start: str = ""
    end: str = ""
    is_internship: bool = False
    is_project: bool = False
    bullets: list[str] = Field(default_factory=list)
    quantified: bool = False       # has at least one numeric bullet

    def text(self) -> str:
        parts = [self.title, self.organization, f"{self.start} {self.end}".strip()]
        parts.extend(self.bullets)
        return " ".join(p for p in parts if p)


class DegreeRecord(BaseModel):
    level: DegreeLevel = "other"
    field: str = ""
    institution: str = ""
    grad_year: Optional[int] = None


class QualificationRequirement(BaseModel):
    kind: Literal["degree", "min_experience", "certification", "other"]
    description: str
    min_years: Optional[float] = None
    degree_level: Optional[DegreeLevel] = None
    status: QualStatus = "unknown"
    explanation: str = ""


class ParsedDocument(BaseModel):
    text: str
    paragraphs: list[str] = Field(default_factory=list)
    page_count: int = 1
    source_format: Literal["pdf", "docx", "ocr", "text"]
    scanned: bool = False
    warnings: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class ResumeProfile(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    sections: dict[str, str] = Field(default_factory=dict)
    bullets: list[str] = Field(default_factory=list)
    skills: list[SkillMention] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    projects: list[ExperienceItem] = Field(default_factory=list)
    education: list[DegreeRecord] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    approx_years_experience: Optional[float] = None
    is_fresh_grad_profile: bool = False
    warnings: list[str] = Field(default_factory=list)


class JobDescription(BaseModel):
    raw_text: str
    title: Optional[str] = None
    company: Optional[str] = None
    required_skills: list[SkillRequirement] = Field(default_factory=list)
    preferred_skills: list[SkillRequirement] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    qualifications: list[QualificationRequirement] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def all_skills(self) -> list[SkillRequirement]:
        return self.required_skills + self.preferred_skills


class CategoryScore(BaseModel):
    name: str
    score: float = 0.0             # 0..100
    weight: float = 0.0
    explanations: list[str] = Field(default_factory=list)
    degraded: bool = False         # evaluator could not run fully
    details: dict[str, Any] = Field(default_factory=dict)


class OverallScore(BaseModel):
    score: float = 0.0
    confidence: float = 1.0        # 0..1: how much of the evaluation has solid data
    explanations: list[str] = Field(default_factory=list)


class ATSReport(BaseModel):
    overall: OverallScore = Field(default_factory=OverallScore)
    categories: list[CategoryScore] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    partially_matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    skill_matches: list[SkillMatch] = Field(default_factory=list)
    relevant_experience: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
