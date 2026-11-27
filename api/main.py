"""HTTP API for Bharat Stock Intelligence."""
import logging
from datetime import date

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from bharat_stock_intelligence.config import get_settings
from bharat_stock_intelligence.database import (
    get_announcements, get_earnings_surprises, get_engine, get_instruments,
    get_sector_sentiment, persist_article_results, persist_event_study,
)
from bharat_stock_intelligence.entity_matching import match_companies
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
    instrument_symbol: str | None = Field(default=None, min_length=1, max_length=30)
    event_type: str = Field(default="corporate_announcement", min_length=2, max_length=80)
    benchmark_symbol: str = Field(default="NIFTY 500", min_length=1, max_length=30)


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
        # Universe-backed matching; unmatched stories remain unmatched, not force-assigned.
        instrument_rows = get_instruments(limit=5000)
        for article in articles:
            result = analyse_text(article["analysis_text"], get_settings().finbert_model)
            matches = match_companies(article["analysis_text"], instrument_rows)
            scored.append({
                **article,
                "company_matches": matches,
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
            "matched_company_count": len({m["symbol"] for a in scored for m in a.get("company_matches", [])}),
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
        observations = frame.to_dict("records")
        run_id = None
        if payload.instrument_symbol:
            run_id = persist_event_study(
                payload.instrument_symbol.upper(), payload.event_type, payload.event_dates[0],
                payload.benchmark_symbol, payload.window_before, payload.window_after,
                observations, "Market-adjusted abnormal return: asset daily return minus benchmark daily return; CAR is cumulative sum.",
            )
        return {
            "run_id": run_id,
            "method": "market_adjusted_abnormal_return",
            "event_count": len(payload.event_dates),
            "observations": observations,
            "interpretation": "Descriptive historical event analysis; not a forecast or causal estimate.",
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc



@app.get("/v1/instruments")
def instruments(q: str | None = Query(default=None, max_length=120), limit: int = Query(100, ge=1, le=500)):
    try:
        rows = get_instruments(query=q, limit=limit)
        return {"count": len(rows), "instruments": rows}
    except Exception as exc:
        logger.exception("Instrument lookup failed")
        raise HTTPException(status_code=503, detail="Instrument database unavailable") from exc


@app.get("/v1/announcements")
def announcements(symbol: str | None = Query(default=None, max_length=30), limit: int = Query(100, ge=1, le=500)):
    try:
        return {"announcements": get_announcements(symbol.upper() if symbol else None, limit)}
    except Exception as exc:
        logger.exception("Announcement lookup failed")
        raise HTTPException(status_code=503, detail="Announcement database unavailable") from exc


@app.get("/v1/earnings/surprises")
def earnings_surprises(limit: int = Query(100, ge=1, le=500)):
    try:
        return {"surprises": get_earnings_surprises(limit)}
    except Exception as exc:
        logger.exception("Earnings surprise lookup failed")
        raise HTTPException(status_code=503, detail="Earnings database unavailable") from exc


@app.get("/v1/sectors/sentiment")
def sector_sentiment(days: int = Query(7, ge=1, le=90)):
    try:
        rows = get_sector_sentiment(get_settings().finbert_model, days)
        return {"lookback_days": days, "sectors": rows, "coverage": "no_data" if not rows else "scored"}
    except Exception as exc:
        logger.exception("Sector sentiment query failed")
        raise HTTPException(status_code=503, detail="Sector sentiment database unavailable") from exc
