"""NewsAPI ingestion; missing credentials never trigger fabricated news."""
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

import requests

from .config import get_settings


def _canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    # Tracking query strings and fragments are excluded from article identity.
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", ""))


def fetch_company_news(query: str, page_size: int = 50) -> list[dict]:
    if not query or not query.strip():
        raise ValueError("Company or ticker query is required.")
    settings = get_settings()
    if not settings.news_api_key:
        raise RuntimeError("NEWS_API_KEY is not configured; live news is unavailable.")
    response = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": query.strip(),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(max(page_size, 1), 100),
            "apiKey": settings.news_api_key,
        },
        timeout=settings.request_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "ok":
        raise RuntimeError(f"News provider error: {payload.get('code', 'unknown')}")

    unique: dict[str, dict] = {}
    for item in payload.get("articles", []):
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        if not title or not url or title == "[Removed]":
            continue
        canonical_url = _canonical_url(url)
        if not canonical_url:
            continue
        description = (item.get("description") or "").strip()
        content = " ".join(part for part in (title, description) if part)
        unique.setdefault(
            canonical_url,
            {
                "title": title,
                "analysis_text": content,
                "url": url,
                "source": (item.get("source") or {}).get("name"),
                "published_at": item.get("publishedAt"),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "description": description or None,
            },
        )
    return list(unique.values())
