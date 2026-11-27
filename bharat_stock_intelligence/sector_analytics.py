"""Sector-level sentiment aggregates with explicit sample coverage."""
from __future__ import annotations

from collections import defaultdict


def aggregate_sector_sentiment(rows: list[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        sector = (row.get("sector") or "Unclassified").strip()
        if row.get("score") is None:
            continue
        buckets[sector].append(row)
    output = []
    for sector, items in sorted(buckets.items()):
        weights = [max(0.000001, float(item.get("weight", 1.0))) for item in items]
        score = sum(float(item["score"]) * w for item, w in zip(items, weights)) / sum(weights)
        output.append({
            "sector": sector,
            "sentiment_score": round(score, 4),
            "article_count": len(items),
            "company_count": len({item.get("symbol") for item in items if item.get("symbol")}),
            "coverage": "scored" if items else "no_data",
        })
    return output
