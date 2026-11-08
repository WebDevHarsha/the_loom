"""Category weights for the scoring engine.

The weights below are initial product choices for calibration - they are NOT
universal ATS industry standards. Adjust with care and re-run tests.
"""

from __future__ import annotations

from config import DEFAULT_WEIGHTS  # path: config.py at package root


def normalize(weights: dict[str, float] | None = None) -> dict[str, float]:
    """Return a copy of the weights whose values sum to exactly 100."""
    w = dict(weights if weights is not None else DEFAULT_WEIGHTS)
    total = sum(w.values())
    if total <= 0:
        raise ValueError("Weights must sum to a positive number")
    if abs(total - 100.0) > 1e-9:                       # renormalize if needed
        w = {k: v * 100.0 / total for k, v in w.items()}
    return w


def assert_sums_to_100() -> None:
    """Raise if the default configuration is internally inconsistent."""
    total = sum(DEFAULT_WEIGHTS.values())
    if abs(total - 100.0) > 1e-9:
        raise ValueError(f"DEFAULT_WEIGHTS sum to {total}, expected 100")
