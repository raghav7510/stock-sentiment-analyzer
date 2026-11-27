"""Dependency-light sentiment aggregation and validation."""
from __future__ import annotations

import math
from collections.abc import Iterable


def aggregate_sentiment(items: Iterable[dict]) -> dict:
    """Aggregate article scores using explicit positive weights.

    Empty coverage is represented as no_data, never as neutral sentiment.
    """
    rows = list(items)
    if not rows:
        return {"score": 0.0, "article_count": 0, "coverage": "no_data"}

    weighted: list[tuple[float, float]] = []
    for row in rows:
        try:
            score = float(row["score"])
            weight = float(row.get("weight", 1.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Each item requires a numeric score and optional numeric weight.") from exc
        if not math.isfinite(score) or not -1 <= score <= 1:
            raise ValueError("Sentiment scores must be finite and between -1 and 1.")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Article weights must be finite and positive.")
        weighted.append((score, weight))

    total_weight = sum(weight for _, weight in weighted)
    score = sum(value * weight for value, weight in weighted) / total_weight
    return {"score": round(score, 4), "article_count": len(rows), "coverage": "scored"}
