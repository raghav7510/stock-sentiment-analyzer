"""PostgreSQL persistence for normalized articles and model outputs."""
from __future__ import annotations

import hashlib
from datetime import datetime
from functools import lru_cache

from sqlalchemy import create_engine, text

from .config import get_settings


@lru_cache(maxsize=1)
def get_engine():
    return create_engine(get_settings().database_url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def _published_at(value: str | None) -> str | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.isoformat()


def persist_article_results(articles: list[dict], model_name: str) -> int:
    """Idempotently upsert articles and their model-specific sentiment outputs."""
    stored = 0
    statement = text("""
        INSERT INTO news_articles (url, title, source, published_at, content_hash)
        VALUES (:url, :title, :source, CAST(:published_at AS timestamptz), :content_hash)
        ON CONFLICT DO NOTHING
    """)
    lookup = text("""
        SELECT id FROM news_articles
        WHERE url = :url
           OR (content_hash = :content_hash
               AND published_at IS NOT DISTINCT FROM CAST(:published_at AS timestamptz))
        ORDER BY CASE WHEN url = :url THEN 0 ELSE 1 END
        LIMIT 1
    """)
    sentiment_statement = text("""
        INSERT INTO article_sentiment (article_id, model_name, label, score, confidence)
        VALUES (:article_id, :model_name, :label, :score, :confidence)
        ON CONFLICT (article_id, model_name) DO UPDATE SET
            label = EXCLUDED.label,
            score = EXCLUDED.score,
            confidence = EXCLUDED.confidence,
            created_at = NOW()
    """)
    with get_engine().begin() as connection:
        for article in articles:
            published_at = _published_at(article.get("published_at"))
            identity_text = article.get("analysis_text") or article["title"]
            content_hash = hashlib.sha256(identity_text.encode("utf-8")).hexdigest()
            params = {
                "url": article["url"], "title": article["title"],
                "source": article.get("source"), "published_at": published_at,
                "content_hash": content_hash,
            }
            connection.execute(statement, params)
            row = connection.execute(lookup, params).first()
            if row is None:
                continue
            connection.execute(sentiment_statement, {
                "article_id": row.id, "model_name": model_name,
                "label": article["sentiment"], "score": float(article["score"]),
                "confidence": float(article["confidence"]),
            })
            stored += 1
    return stored
