CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS instruments (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL UNIQUE,
    company_name TEXT,
    exchange TEXT NOT NULL DEFAULT 'NSE',
    sector TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS news_articles (
    id BIGSERIAL PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    source TEXT,
    published_at TIMESTAMPTZ,
    retrieved_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    content_hash TEXT NOT NULL,
    UNIQUE (content_hash, published_at)
);

CREATE TABLE IF NOT EXISTS article_sentiment (
    article_id BIGINT NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('positive','neutral','negative')),
    score DOUBLE PRECISION NOT NULL CHECK (score BETWEEN -1 AND 1),
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (article_id, model_name)
);

CREATE TABLE IF NOT EXISTS daily_prices (
    instrument_id BIGINT NOT NULL REFERENCES instruments(id),
    observed_at TIMESTAMPTZ NOT NULL,
    open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC,
    volume BIGINT,
    provider TEXT NOT NULL,
    data_status TEXT NOT NULL DEFAULT 'historical_daily',
    PRIMARY KEY (instrument_id, observed_at)
);
SELECT create_hypertable('daily_prices', 'observed_at', if_not_exists => TRUE);
