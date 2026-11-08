"""Resume content quality analysis (spec section 12).

Rule-based: action verbs, weak/vague phrasing, repetition, quantified
achievements, bullet length. Grammar checking via language-tool-python is
strictly opt-in (env ATS_ENGINE_GRAMMAR=1) because it downloads a large
server jar and requires Java — disabled by default, never blocking.
"""

from __future__ import annotations

import json
import re

from config import (
    ACTION_VERB_TARGET,
    ACTION_VERBS_FILE,
    GRAMMAR_ENABLED,
    LONG_BULLET_WORDS,
    QUANTIFIED_TARGET,
    WEAK_PHRASES,
)
from schemas import CategoryScore, ResumeProfile

_NUM_RE = re.compile(r"\d")


def _load_action_verbs() -> set[str]:
    try:
        with open(ACTION_VERBS_FILE, encoding="utf-8") as f:
            return {v.strip().lower() for v in json.load(f) if v.strip()}
    except (OSError, json.JSONDecodeError):
        return set()


def maybe_grammar_check(text: str) -> list[str] | None:
    """Run LanguageTool only when explicitly enabled; any failure => None."""
    if not GRAMMAR_ENABLED:
        return None
    try:
        import language_tool_python
        tool = language_tool_python.LanguageTool("en-US")
        matches = tool.check(text[:20000])
        issues = [f"{m.ruleId}: {m.context.strip()}" for m in matches[:40]]
        tool.close()
        return issues
    except Exception:
        return None


def evaluate_content(profile: ResumeProfile) -> CategoryScore:
    bullets = [b for b in profile.bullets if len(b.split()) >= 4]
    issues: list[str] = []
    recommendations: list[str] = []
    deduction = 0.0

    action_verbs = _load_action_verbs()

    if not bullets:
        return CategoryScore(
            name="Content Quality",
            score=0.0,
            weight=0.0,
            degraded=True,
            explanations=["No bulleted content found to analyze."],
        )

    # 1) action verb usage
    strong = [b for b in bullets if b.split()[0].lower().strip(".,;: ") in action_verbs]
    ratio = len(strong) / len(bullets)
    if ratio < ACTION_VERB_TARGET:
        deduction += 10
        issues.append(f"Only {ratio:.0%} of bullets start with an action verb (target {ACTION_VERB_TARGET:.0%}).")
        examples = [b for b in bullets if b.split()[0].lower().strip(".,;: ") not in action_verbs][:2]
        if examples:
            recommendations.append(
                "Start bullets with strong action verbs (developed, built, automated, "
                f"analyzed, delivered); e.g. \"{examples[0][:80]}\"."
            )

    # 2) weak / vague phrasing
    weak_hits: list[tuple[str, str]] = []
    for b in bullets:
        low = b.lower()
        for phrase in WEAK_PHRASES:
            if phrase in low:
                weak_hits.append((phrase, b))
    if weak_hits:
        deduction += min(20.0, 4.0 * len(weak_hits))
        issues.append(f"{len(weak_hits)} bullet(s) use weak filler phrasing.")
        seen: set[str] = set()
        for phrase, b in weak_hits[:4]:
            if phrase not in seen:
                seen.add(phrase)
                recommendations.append(
                    f"Replace vague phrasing \"{phrase}\" in: \"{b[:80]}...\" — say what you "
                    "specifically did and the outcome (only facts already on the resume)."
                )

    # 3) quantified achievements
    quant = sum(1 for b in bullets if _NUM_RE.search(b))
    q_ratio = quant / len(bullets)
    if q_ratio < QUANTIFIED_TARGET:
        deduction += 10
        issues.append(f"Only {q_ratio:.0%} of bullets include numbers/metrics (target {QUANTIFIED_TARGET:.0%}).")
        unquantified = [b for b in bullets if not _NUM_RE.search(b)][:2]
        for b in unquantified:
            recommendations.append(
                f"Quantify where you truthfully can: \"{b[:80]}...\" (impact, scale, users, %, latency)."
            )

    # 4) overly long bullets
    long_bullets = [b for b in bullets if len(b.split()) > LONG_BULLET_WORDS]
    if long_bullets:
        deduction += min(12.0, 3.0 * len(long_bullets))
        issues.append(f"{len(long_bullets)} bullet(s) are longer than {LONG_BULLET_WORDS} words; split or tighten them.")

    # 5) repeated phrasing across bullets
    bigrams: dict[str, int] = {}
    for b in bullets:
        words = re.findall(r"[a-z']+", b.lower())
        for i in range(len(words) - 2):
            gram = " ".join(words[i:i + 3])
            bigrams[gram] = bigrams.get(gram, 0) + 1
    repeated = [(g, c) for g, c in bigrams.items() if c >= 4]
    if repeated:
        deduction += 8
        repeated.sort(key=lambda x: -x[1])
        issues.append(f"Repeated phrasing across bullets, e.g. \"{repeated[0][0]}\" used {repeated[0][1]}x.")
        recommendations.append("Vary wording across bullets; keep every bullet specific to one accomplishment.")

    # 6) optional grammar (env-gated)
    flat = "\n".join(bullets)
    grammar = maybe_grammar_check(flat)
    if grammar is not None:
        n = len(grammar)
        deduction += min(10.0, 1.0 * n)
        issues.append(f"{n} potential grammar/language issues found.")
        recommendations.extend("Grammar: " + g for g in grammar[:3])

    score = max(0.0, 100.0 - deduction)
    if not issues:
        issues.append("No content-quality red flags detected by the rule checks.")
    if not recommendations:
        recommendations.append("Content reads strong; keep bullets specific, quantified, and verb-led.")

    return CategoryScore(
        name="Content Quality",
        score=round(score, 1),
        weight=0.0,
        explanations=issues[:12],
        details={"recommendations": recommendations[:10]},
    )
