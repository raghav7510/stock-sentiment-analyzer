"""NewsAPI ingestion; missing credentials never trigger fabricated news."""
from datetime import datetime, timezone
import requests
from .config import get_settings

def fetch_company_news(query: str, page_size: int = 50) -> list[dict]:
    if not query or not query.strip():
        raise ValueError("Company or ticker query is required.")
    settings = get_settings()
    if not settings.news_api_key:
        raise RuntimeError("NEWS_API_KEY is not configured; live news is unavailable.")
    response = requests.get("https://newsapi.org/v2/everything", params={
        "q": query.strip(), "language": "en", "sortBy": "publishedAt",
        "pageSize": min(max(page_size, 1), 100), "apiKey": settings.news_api_key,
    }, timeout=settings.request_timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "ok":
        raise RuntimeError(f"News provider error: {payload.get('code', 'unknown')}")
    articles = []
    for item in payload.get("articles", []):
        title, url = (item.get("title") or "").strip(), item.get("url")
        if not title or not url or title == "[Removed]":
            continue
        articles.append({"title": title, "url": url, "source": (item.get("source") or {}).get("name"),
                         "published_at": item.get("publishedAt"), "retrieved_at": datetime.now(timezone.utc).isoformat(),
                         "description": item.get("description")})
    unique = {}
    for article in articles:
        unique.setdefault(" ".join(article["title"].lower().split()), article)
    return list(unique.values())
