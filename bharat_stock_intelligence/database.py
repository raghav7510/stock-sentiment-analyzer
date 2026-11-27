"""PostgreSQL persistence and read models for Indian equity intelligence."""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from functools import lru_cache

from sqlalchemy import create_engine, text

from .config import get_settings


@lru_cache(maxsize=1)
def get_engine():
    return create_engine(get_settings().database_url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def _published_at(value):
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def persist_universe(instruments: list[dict], constituents: list[dict]) -> dict:
    """Upsert current NSE instruments and retain a dated NIFTY 500 snapshot."""
    instrument_sql = text("""
        INSERT INTO instruments(symbol, company_name, exchange, sector, isin, series, source, updated_at)
        VALUES (:symbol, :company_name, :exchange, :sector, :isin, :series, :source, NOW())
        ON CONFLICT (exchange, symbol) DO UPDATE SET
            company_name = EXCLUDED.company_name,
            isin = COALESCE(EXCLUDED.isin, instruments.isin),
            series = COALESCE(EXCLUDED.series, instruments.series),
            source = EXCLUDED.source,
            updated_at = NOW()
        RETURNING id
    """)
    snapshot_sql = text("""
        INSERT INTO universe_snapshots(index_name, captured_on, source_as_of_date, symbol, exchange, company_name, sector, source)
        VALUES ('NIFTY 500', :captured_on, :source_as_of_date, :symbol, 'NSE', :company_name, :sector, :source)
        ON CONFLICT (index_name, captured_on, exchange, symbol) DO UPDATE SET
            company_name = EXCLUDED.company_name, sector = EXCLUDED.sector, source = EXCLUDED.source
    """)
    sector_sql = text("""
        UPDATE instruments SET sector = :sector, company_name = COALESCE(:company_name, company_name),
            updated_at = NOW()
        WHERE exchange = 'NSE' AND symbol = :symbol
    """)
    with get_engine().begin() as connection:
        for row in instruments:
            connection.execute(instrument_sql, {
                "symbol": row["symbol"], "company_name": row.get("company_name"),
                "exchange": row.get("exchange", "NSE"), "sector": row.get("sector"),
                "isin": row.get("isin"), "series": row.get("series"),
                "source": row.get("source", "NSE official equity master"),
            })
        for row in constituents:
            captured_on = row.get("captured_on") or date.today()
            connection.execute(sector_sql, {
                "symbol": row["symbol"], "sector": row.get("sector"),
                "company_name": row.get("company_name"),
            })
            connection.execute(snapshot_sql, {
                "captured_on": captured_on, "source_as_of_date": row.get("source_as_of_date"),
                "symbol": row["symbol"], "company_name": row.get("company_name"), "sector": row.get("sector"),
                "source": row.get("source", "NSE Indices NIFTY 500 constituent CSV"),
            })
    return {"instruments_upserted": len(instruments), "nifty500_membership_rows": len(constituents)}


def persist_announcements(rows: list[dict]) -> int:
    statement = text("""
        INSERT INTO corporate_announcements(
            instrument_id, exchange, symbol, company_name, subject, details,
            published_at, source_url, source_hash, source_format, retrieved_at
        )
        VALUES (:instrument_id, :exchange, :symbol, :company_name, :subject, :details,
            :published_at, :source_url, :source_hash, :source_format, :retrieved_at)
        ON CONFLICT (source_hash) DO NOTHING
    """)
    with get_engine().begin() as connection:
        count = 0
        for row in rows:
            instrument_id = None
            if row.get("symbol"):
                instrument_id = connection.execute(text(
                    "SELECT id FROM instruments WHERE exchange=:exchange AND symbol=:symbol LIMIT 1"
                ), {"exchange": row["exchange"], "symbol": row["symbol"]}).scalar()
            if instrument_id is None and row.get("company_name"):
                instrument_id = connection.execute(text(
                    "SELECT id FROM instruments WHERE exchange=:exchange AND LOWER(company_name)=LOWER(:name) LIMIT 1"
                ), {"exchange": row["exchange"], "name": row["company_name"]}).scalar()
            result = connection.execute(statement, {
                **row, "instrument_id": instrument_id,
                "published_at": _published_at(row.get("published_at")),
            })
            count += result.rowcount
    return count


def persist_article_results(articles: list[dict], model_name: str) -> int:
    """Upsert articles, sentiment, and deterministic instrument associations."""
    stored = 0
    insert_article = text("""
        INSERT INTO news_articles (url, title, source, published_at, content_hash)
        VALUES (:url, :title, :source, CAST(:published_at AS timestamptz), :content_hash)
        ON CONFLICT DO NOTHING
    """)
    lookup_article = text("SELECT id FROM news_articles WHERE url=:url LIMIT 1")
    sentiment_statement = text("""
        INSERT INTO article_sentiment (article_id, model_name, label, score, confidence)
        VALUES (:article_id, :model_name, :label, :score, :confidence)
        ON CONFLICT (article_id, model_name) DO UPDATE SET
            label=EXCLUDED.label, score=EXCLUDED.score, confidence=EXCLUDED.confidence, created_at=NOW()
    """)
    association = text("""
        INSERT INTO article_instruments(article_id, instrument_id, match_type, matched_text, match_confidence)
        VALUES (:article_id, :instrument_id, :match_type, :matched_text, :match_confidence)
        ON CONFLICT (article_id, instrument_id) DO UPDATE SET
            match_type=EXCLUDED.match_type, matched_text=EXCLUDED.matched_text
    """)
    with get_engine().begin() as connection:
        for article in articles:
            identity = article.get("analysis_text") or article["title"]
            params = {
                "url": article["url"], "title": article["title"], "source": article.get("source"),
                "published_at": _published_at(article.get("published_at")),
                "content_hash": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            }
            connection.execute(insert_article, params)
            article_id = connection.execute(lookup_article, {"url": article["url"]}).scalar()
            if article_id is None:
                continue
            connection.execute(sentiment_statement, {
                "article_id": article_id, "model_name": model_name,
                "label": article["sentiment"], "score": float(article["score"]),
                "confidence": float(article["confidence"]),
            })
            for match in article.get("company_matches", []):
                instrument_id = connection.execute(text(
                    "SELECT id FROM instruments WHERE exchange=:exchange AND symbol=:symbol LIMIT 1"
                ), {"symbol": match["symbol"], "exchange": match.get("exchange", "NSE")}).scalar()
                if instrument_id:
                    connection.execute(association, {
                        "article_id": article_id, "instrument_id": instrument_id,
                        "match_type": match["match_type"], "matched_text": match["matched_text"],
                        "match_confidence": 1.0,
                    })
            stored += 1
    return stored


def get_instruments(query: str | None = None, limit: int = 100) -> list[dict]:
    statement = text("""
        SELECT i.symbol, i.company_name, i.exchange, i.sector, i.isin, i.series,
               ARRAY(SELECT a.alias FROM instrument_aliases a WHERE a.instrument_id=i.id) AS aliases,
               EXISTS(SELECT 1 FROM universe_snapshots u
                      WHERE u.symbol=i.symbol AND u.exchange=i.exchange AND u.index_name='NIFTY 500'
                        AND u.captured_on=(SELECT MAX(captured_on) FROM universe_snapshots WHERE index_name='NIFTY 500')) AS in_nifty500
        FROM instruments i
        WHERE (:query IS NULL OR i.symbol ILIKE :pattern OR i.company_name ILIKE :pattern)
        ORDER BY i.company_name LIMIT :limit
    """)
    with get_engine().connect() as connection:
        rows = connection.execute(statement, {
            "query": query, "pattern": f"%{query}%" if query else None, "limit": limit,
        }).mappings().all()
    return [dict(row) for row in rows]


def get_sector_sentiment(model_name: str, days: int = 7) -> list[dict]:
    statement = text("""
        SELECT COALESCE(i.sector, 'Unclassified') AS sector,
               ROUND(AVG(s.score)::numeric, 4) AS sentiment_score,
               COUNT(DISTINCT a.id) AS article_count,
               COUNT(DISTINCT i.id) AS company_count,
               MAX(a.published_at) AS latest_article_at
        FROM article_sentiment s
        JOIN news_articles a ON a.id=s.article_id
        JOIN article_instruments ai ON ai.article_id=a.id
        JOIN instruments i ON i.id=ai.instrument_id
        WHERE s.model_name=:model_name AND a.published_at >= NOW() - (:days * INTERVAL '1 day')
        GROUP BY COALESCE(i.sector, 'Unclassified')
        ORDER BY sentiment_score DESC
    """)
    with get_engine().connect() as connection:
        return [dict(row) for row in connection.execute(statement, {"model_name": model_name, "days": days}).mappings().all()]


def persist_earnings_surprises(rows: list[dict]) -> int:
    statement = text("""
        INSERT INTO earnings_surprises(
            instrument_id, fiscal_period, metric, actual_value, consensus_value,
            absolute_surprise, surprise_pct, direction, actual_source, consensus_source, published_at
        )
        SELECT i.id, :fiscal_period, :metric, :actual_value, :consensus_value,
               :absolute_surprise, :surprise_pct, :direction, :actual_source, :consensus_source,
               :published_at
        FROM instruments i
        WHERE i.exchange=:exchange AND i.symbol=:symbol
        ON CONFLICT(instrument_id, fiscal_period, metric, actual_source, consensus_source)
        DO UPDATE SET actual_value=EXCLUDED.actual_value, consensus_value=EXCLUDED.consensus_value,
          absolute_surprise=EXCLUDED.absolute_surprise, surprise_pct=EXCLUDED.surprise_pct,
          direction=EXCLUDED.direction, published_at=EXCLUDED.published_at, imported_at=NOW()
    """)
    with get_engine().begin() as connection:
        count = 0
        for row in rows:
            result = connection.execute(statement, {
                **row, "published_at": _published_at(row.get("published_at")),
                "exchange": row.get("exchange", "NSE"),
            })
            count += result.rowcount
    return count


def get_announcements(symbol: str | None = None, limit: int = 100) -> list[dict]:
    statement = text("""
        SELECT exchange, symbol, company_name, subject, details, published_at, source_url
        FROM corporate_announcements
        WHERE (:symbol IS NULL OR symbol=:symbol)
        ORDER BY published_at DESC NULLS LAST, retrieved_at DESC LIMIT :limit
    """)
    with get_engine().connect() as connection:
        return [dict(row) for row in connection.execute(statement, {"symbol": symbol, "limit": limit}).mappings().all()]


def get_earnings_surprises(limit: int = 100) -> list[dict]:
    statement = text("""
        SELECT i.symbol, i.company_name, i.sector, e.fiscal_period, e.metric,
               e.actual_value, e.consensus_value, e.absolute_surprise, e.surprise_pct,
               e.direction, e.actual_source, e.consensus_source, e.published_at
        FROM earnings_surprises e JOIN instruments i ON i.id=e.instrument_id
        ORDER BY e.published_at DESC NULLS LAST, e.imported_at DESC LIMIT :limit
    """)
    with get_engine().connect() as connection:
        return [dict(row) for row in connection.execute(statement, {"limit": limit}).mappings().all()]


def persist_event_study(instrument_symbol: str, event_type: str, event_date: date,
                        benchmark_symbol: str, window_before: int, window_after: int,
                        observations: list[dict], methodology: str, exchange: str = "NSE") -> int:
    import json
    statement = text("""
        INSERT INTO event_study_runs(instrument_id, event_type, event_date, benchmark_symbol,
            window_before, window_after, observations, methodology)
        SELECT id, :event_type, :event_date, :benchmark_symbol, :window_before,
            :window_after, CAST(:observations AS jsonb), :methodology
        FROM instruments WHERE symbol=:symbol AND exchange=:exchange
        RETURNING id
    """)
    with get_engine().begin() as connection:
        value = connection.execute(statement, {
            "symbol": instrument_symbol, "event_type": event_type, "event_date": event_date,
            "benchmark_symbol": benchmark_symbol, "window_before": window_before,
            "window_after": window_after, "observations": json.dumps(observations, default=str),
            "methodology": methodology, "exchange": exchange,
        }).scalar()
    if value is None:
        raise LookupError(f"Instrument {instrument_symbol} is not in the instrument master.")
    return int(value)



def persist_daily_prices(symbol: str, rows: list[dict], provider: str, exchange: str = "NSE") -> int:
    statement = text("""
        INSERT INTO daily_prices(instrument_id, observed_at, open, high, low, close, volume, provider, data_status)
        SELECT i.id, CAST(:observed_at AS timestamptz), :open, :high, :low, :close, :volume, :provider, 'historical_daily'
        FROM instruments i WHERE i.exchange=:exchange AND i.symbol=:symbol
        ON CONFLICT(instrument_id, observed_at) DO UPDATE SET
            open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close,
            volume=EXCLUDED.volume, provider=EXCLUDED.provider, data_status=EXCLUDED.data_status
    """)
    written = 0
    with get_engine().begin() as connection:
        for row in rows:
            result = connection.execute(statement, {
                "symbol": symbol.upper(), "exchange": exchange, "observed_at": row["Date"],
                "open": row.get("Open"), "high": row.get("High"), "low": row.get("Low"),
                "close": row.get("Close"), "volume": row.get("Volume"), "provider": provider,
            })
            written += result.rowcount
    return written


def get_daily_close_returns(symbol: str, exchange: str = "NSE", limit: int = 1500) -> list[dict]:
    statement = text("""
        SELECT p.observed_at::date AS trade_date, p.close
        FROM daily_prices p JOIN instruments i ON i.id=p.instrument_id
        WHERE i.symbol=:symbol AND i.exchange=:exchange AND p.close IS NOT NULL
        ORDER BY p.observed_at DESC LIMIT :limit
    """)
    with get_engine().connect() as connection:
        rows = connection.execute(statement, {
            "symbol": symbol.upper(), "exchange": exchange, "limit": limit,
        }).mappings().all()
    ordered = list(reversed(rows))
    output = []
    previous = None
    for row in ordered:
        close = float(row["close"])
        daily_return = None if previous is None or previous == 0 else close / previous - 1
        output.append({"date": row["trade_date"], "close": close, "return": daily_return})
        previous = close
    return output


def ensure_instrument(symbol: str, exchange: str = "NSE", company_name: str | None = None) -> None:
    statement = text("""
        INSERT INTO instruments(symbol, company_name, exchange, source)
        VALUES (:symbol, :company_name, :exchange, 'market_history_adapter')
        ON CONFLICT(exchange, symbol) DO NOTHING
    """)
    with get_engine().begin() as connection:
        connection.execute(statement, {
            "symbol": symbol.upper(), "exchange": exchange,
            "company_name": company_name or symbol.upper(),
        })
