"""Transparent earnings-surprise calculations from actuals and consensus estimates."""
from __future__ import annotations

import math


def calculate_surprise(actual: float, estimate: float) -> dict:
    actual, estimate = float(actual), float(estimate)
    if not math.isfinite(actual) or not math.isfinite(estimate):
        raise ValueError("Actual and estimate must be finite numbers.")
    if estimate == 0:
        raise ValueError("Percentage surprise is undefined when consensus estimate is zero.")
    delta = actual - estimate
    pct = delta / abs(estimate) * 100.0
    return {
        "actual": actual,
        "consensus_estimate": estimate,
        "surprise": round(delta, 6),
        "surprise_pct": round(pct, 4),
        "direction": "beat" if delta > 0 else "miss" if delta < 0 else "inline",
    }


def classify_earnings_surprise(actual_eps: float | None, estimate_eps: float | None,
                               actual_revenue: float | None = None,
                               estimate_revenue: float | None = None) -> dict:
    """Calculate EPS/revenue surprises only when a valid estimate is supplied."""
    result = {}
    if actual_eps is not None and estimate_eps is not None:
        result["eps"] = calculate_surprise(actual_eps, estimate_eps)
    if actual_revenue is not None and estimate_revenue is not None:
        result["revenue"] = calculate_surprise(actual_revenue, estimate_revenue)
    if not result:
        raise ValueError("At least one actual-versus-consensus metric pair is required.")
    return result
