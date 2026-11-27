-- Indian Equity Intelligence Engine schema extension.
-- Run after 001_initial_schema.sql. All source data must retain provenance.

ALTER TABLE instruments ADD COLUMN IF NOT EXISTS isin TEXT;
ALTER TABLE instruments ADD COLUMN IF NOT EXISTS series TEXT;
ALTER TABLE instruments ADD COLUMN IF NOT EXISTS source TEXT;
ALTER TABLE instruments ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE instruments DROP CONSTRAINT IF EXISTS instruments_symbol_key;
CREATE UNIQUE INDEX IF NOT EXISTS uq_instruments_exchange_symbol ON instruments(exchange, symbol);
CREATE INDEX IF NOT EXISTS ix_instruments_isin ON instruments(isin);

CREATE TABLE IF NOT EXISTS universe_snapshots (
    id BIGSERIAL PRIMARY KEY,
    index_name TEXT NOT NULL,
    captured_on DATE NOT NULL,
    source_as_of_date DATE,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL DEFAULT 'NSE',
    company_name TEXT,
    sector TEXT,
    source TEXT NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(index_name, captured_on, exchange, symbol)
);
CREATE INDEX IF NOT EXISTS ix_universe_snapshot_lookup ON universe_snapshots(index_name, captured_on, sector);

CREATE TABLE IF NOT EXISTS instrument_aliases (
    id BIGSERIAL PRIMARY KEY,
    instrument_id BIGINT NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    alias TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual_or_exchange',
    UNIQUE(instrument_id, alias)
);
CREATE INDEX IF NOT EXISTS ix_instrument_alias_text ON instrument_aliases(LOWER(alias));

CREATE TABLE IF NOT EXISTS corporate_announcements (
    id BIGSERIAL PRIMARY KEY,
    instrument_id BIGINT REFERENCES instruments(id),
    exchange TEXT NOT NULL,
    symbol TEXT,
    company_name TEXT,
    subject TEXT NOT NULL,
    details TEXT,
    published_at TIMESTAMPTZ,
    source_url TEXT,
    source_hash TEXT NOT NULL UNIQUE,
    source_format TEXT NOT NULL,
    retrieved_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_announcements_published ON corporate_announcements(published_at DESC);
CREATE INDEX IF NOT EXISTS ix_announcements_instrument ON corporate_announcements(instrument_id, published_at DESC);

ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS canonical_url TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS uq_news_articles_canonical_url ON news_articles(canonical_url) WHERE canonical_url IS NOT NULL;

CREATE TABLE IF NOT EXISTS article_instruments (
    article_id BIGINT NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    instrument_id BIGINT NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    match_type TEXT NOT NULL,
    matched_text TEXT,
    match_confidence DOUBLE PRECISION NOT NULL DEFAULT 1 CHECK (match_confidence BETWEEN 0 AND 1),
    PRIMARY KEY(article_id, instrument_id)
);

CREATE TABLE IF NOT EXISTS earnings_surprises (
    id BIGSERIAL PRIMARY KEY,
    instrument_id BIGINT NOT NULL REFERENCES instruments(id),
    fiscal_period TEXT NOT NULL,
    metric TEXT NOT NULL CHECK(metric IN ('EPS','REVENUE')),
    actual_value NUMERIC NOT NULL,
    consensus_value NUMERIC NOT NULL,
    absolute_surprise NUMERIC NOT NULL,
    surprise_pct NUMERIC NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('beat','miss','inline')),
    actual_source TEXT NOT NULL,
    consensus_source TEXT NOT NULL,
    published_at TIMESTAMPTZ,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(instrument_id, fiscal_period, metric, actual_source, consensus_source)
);
CREATE INDEX IF NOT EXISTS ix_earnings_surprise_published ON earnings_surprises(published_at DESC);

CREATE TABLE IF NOT EXISTS event_study_runs (
    id BIGSERIAL PRIMARY KEY,
    instrument_id BIGINT REFERENCES instruments(id),
    event_type TEXT NOT NULL,
    event_date DATE NOT NULL,
    benchmark_symbol TEXT NOT NULL,
    window_before INTEGER NOT NULL,
    window_after INTEGER NOT NULL,
    observations JSONB NOT NULL,
    methodology TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    dataset TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('success','partial','failed')),
    rows_seen INTEGER NOT NULL DEFAULT 0,
    rows_written INTEGER NOT NULL DEFAULT 0,
    source_url TEXT,
    checksum TEXT,
    error_message TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);
