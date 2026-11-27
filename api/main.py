"""HTTP API for Bharat Stock Intelligence."""
import logging
from datetime import date

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from bharat_stock_intelligence.config import get_settings
from bharat_stock_intelligence.database import get_engine, persist_article_results
from bharat_stock_intelligence.event_study import calculate_event_study
from bharat_stock_intelligence.logging_config import configure_logging
from bharat_stock_intelligence.market_data import get_daily_history
from bharat_stock_intelligence.news import fetch_company_news
from bharat_stock_intelligence.sentiment import analyse_text, aggregate_sentiment

configure_logging()
logger = logging.getLogger("bsi.api")

app = FastAPI(
    title="Bharat Stock Intelligence API",
    version="0.2.0",
    description="India-focused research analytics API; not investment advice.",
)


class AnalyseRequest(BaseModel):
    text: str = Field(min_length=3, max_length=5000)


class DailyReturn(BaseModel):
    date: date
    value: float = Field(gt=-1, le=10)


class EventStudyRequest(BaseModel):
    asset_returns: list[DailyReturn]
    benchmark_returns: list[DailyReturn]
    event_dates: list[date]
    window_before: int = Field(default=5, ge=0, le=60)
    window_after: int = Field(default=5, ge=0, le=60)


@app.get("/health")
def health():
    return {"status": "ok", "service": get_settings().app_name}


@app.get("/health/ready")
def readiness():
    try:
        with get_engine().connect() as connection:
            connection.exec_driver_sql("SELECT 1")
        return {"status": "ready", "database": "connected"}
    except Exception as exc:
        logger.exception("Readiness check failed")
        raise HTTPException(status_code=503, detail=f"Database unavailable: {type(exc).__name__}") from exc


@app.post("/v1/sentiment")
def sentiment(payload: AnalyseRequest):
    try:
        result = analyse_text(payload.text, get_settings().finbert_model)
        return {
            "label": result.label,
            "score": result.score,
            "confidence": result.confidence,
            "probabilities": result.probabilities,
            "model": get_settings().finbert_model,
        }
    except Exception as exc:
        logger.exception("Sentiment inference failed")
        raise HTTPException(status_code=503, detail=f"Sentiment model unavailable: {type(exc).__name__}") from exc


@app.get("/v1/news")
def news(company: str = Query(min_length=1, max_length=120), limit: int = Query(50, ge=1, le=100)):
    try:
        articles, scored = fetch_company_news(company, limit), []
        for article in articles:
            result = analyse_text(article["analysis_text"], get_settings().finbert_model)
            scored.append({
                **article,
                "sentiment": result.label,
                "score": result.score,
                "confidence": result.confidence,
            })
        summary = aggregate_sentiment(scored)
        persistence = "not_attempted"
        if scored:
            try:
                stored = persist_article_results(scored, get_settings().finbert_model)
                persistence = f"stored_{stored}_articles"
            except Exception:
                persistence = "unavailable"
                logger.exception("Article persistence failed; returning provider results without storage")
        return {
            "company_query": company,
            "articles": scored,
            "summary": summary,
            "data_status": "news_provider_response_not_exchange_complete",
            "persistence_status": persistence,
            "model": get_settings().finbert_model,
        }
    except (RuntimeError, LookupError, ValueError) as exc:
        logger.warning("News request rejected: %s", str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("News analysis failed")
        raise HTTPException(status_code=502, detail=f"News analysis failed: {type(exc).__name__}") from exc


@app.get("/v1/market/{ticker}")
def market(ticker: str, period: str = Query("1y", pattern="^(1mo|3mo|6mo|1y|2y|5y)$")):
    try:
        return get_daily_history(ticker, period)
    except (ValueError, LookupError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Historical market data request failed")
        raise HTTPException(status_code=502, detail=f"Market data provider failed: {type(exc).__name__}") from exc


@app.post("/v1/analytics/event-study")
def event_study(payload: EventStudyRequest):
    asset = pd.Series(
        [item.value for item in payload.asset_returns],
        index=pd.to_datetime([item.date for item in payload.asset_returns]),
    )
    benchmark = pd.Series(
        [item.value for item in payload.benchmark_returns],
        index=pd.to_datetime([item.date for item in payload.benchmark_returns]),
    )
    try:
        frame = calculate_event_study(
            asset, benchmark, [pd.Timestamp(day) for day in payload.event_dates],
            payload.window_before, payload.window_after,
        )
        return {
            "method": "market_adjusted_abnormal_return",
            "event_count": len(payload.event_dates),
            "observations": frame.to_dict("records"),
            "interpretation": "Descriptive historical event analysis; not a forecast or causal estimate.",
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
