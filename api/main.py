"""HTTP API for Bharat Stock Intelligence."""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from bharat_stock_intelligence.config import get_settings
from bharat_stock_intelligence.market_data import get_daily_history
from bharat_stock_intelligence.news import fetch_company_news
from bharat_stock_intelligence.sentiment import analyse_text, aggregate_sentiment

app = FastAPI(title="Bharat Stock Intelligence API", version="0.1.0", description="Research analytics API; not investment advice.")

class AnalyseRequest(BaseModel):
    text: str = Field(min_length=3, max_length=5000)

@app.get("/health")
def health():
    return {"status": "ok", "service": get_settings().app_name}

@app.post("/v1/sentiment")
def sentiment(payload: AnalyseRequest):
    try:
        result = analyse_text(payload.text, get_settings().finbert_model)
        return {"label": result.label, "score": result.score, "confidence": result.confidence, "probabilities": result.probabilities}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Sentiment model unavailable: {type(exc).__name__}") from exc

@app.get("/v1/news")
def news(company: str = Query(min_length=1, max_length=120), limit: int = Query(50, ge=1, le=100)):
    try:
        articles, scored = fetch_company_news(company, limit), []
        for article in articles:
            result = analyse_text(article["title"], get_settings().finbert_model)
            scored.append({**article, "sentiment": result.label, "score": result.score, "confidence": result.confidence})
        return {"company_query": company, "articles": scored, "summary": aggregate_sentiment(scored), "data_status": "live_news_provider_response"}
    except (RuntimeError, LookupError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"News analysis failed: {type(exc).__name__}") from exc

@app.get("/v1/market/{ticker}")
def market(ticker: str, period: str = Query("1y", pattern="^(1mo|3mo|6mo|1y|2y|5y)$")):
    try:
        return get_daily_history(ticker, period)
    except (ValueError, LookupError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
