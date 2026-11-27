# Bharat Stock Intelligence

An India-focused financial research platform. The new implementation is developed alongside the legacy Streamlit prototype so existing behaviour remains available while the new foundation is validated.

## Indian Equity Intelligence Engine (foundation v0.3)

1. **NSE/BSE instrument master:** parser for the official NSE equity-segment security CSV and BSE standardized scrip-master CSV. Instrument identifiers, ISIN, exchange, series and source provenance are stored in PostgreSQL.
2. **NIFTY 500 universe:** imports the official NSE Indices constituent CSV and stores a dated membership snapshot with industry/sector. It does not fabricate constituents or infer historical membership.
3. **Corporate announcements:** importer for CSV exports downloaded from NSE/BSE corporate-announcement pages. Stores issuer, subject, details, timestamp, attachment URL, exchange and deduplication hash. It does not scrape private/undocumented endpoints.
4. **Earnings surprises:** calculates EPS/revenue beat, miss or inline versus a supplied consensus estimate. Actual and consensus source labels are mandatory; no consensus data is invented.
5. **Sector sentiment:** joins FinBERT-scored news to matched instruments and calculates sector aggregates with article/company coverage.
6. **News-to-company matching:** deterministic exact issuer-name, alias and ticker matching against the loaded instrument master; ambiguous/fuzzy guesses are avoided.
7. **Persistent PostgreSQL storage:** instruments, point-in-time NIFTY membership snapshots, aliases, articles, sentiment, article-issuer links, announcements, earnings surprises, daily prices, event-study runs and ingestion audit schema.
8. **Historical event impact:** market-adjusted abnormal return and cumulative abnormal return from stored daily prices. The market-history adapter is currently Yahoo Finance and is explicitly historical/prototype data.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-bsi.txt
```

Copy `.env.example` to `.env` and add your NewsAPI key. Never commit credentials.

Start TimescaleDB: `docker compose -f docker-compose.bsi.yml up -d`.

Apply both schema files in order:

```bash
psql postgresql://bsi:bsi@localhost:5432/bsi -f sql/001_initial_schema.sql
psql postgresql://bsi:bsi@localhost:5432/bsi -f sql/002_indian_equity_intelligence.sql
```

Start API: `uvicorn api.main:app --reload`. API docs: `http://127.0.0.1:8000/docs`.

Start dashboard in another terminal: `streamlit run dashboard_bsi.py`.

Run tests: `python -m pytest tests/ -q`.

## Ingestion workflow

Refresh NSE equity master and current NIFTY 500 constituent snapshot from the official downloadable CSV sources:

```bash
python -m scripts.ingest_indian_equities --refresh-nse
```

Download the BSE standardized scrip master CSV from BSE's official download page and import it:

```bash
python -m scripts.ingest_indian_equities --bse-master path/to/BSE_EQ_SCRIP_DDMMYYYY.csv
```

Download announcement CSV exports from the official NSE/BSE corporate filings pages, then import them:

```bash
python -m scripts.ingest_indian_equities --nse-announcements path/to/nse_announcements.csv
python -m scripts.ingest_indian_equities --bse-announcements path/to/bse_announcements.csv
```

Prepare a consensus CSV from a licensed/identified estimate source. Required columns:

`symbol,fiscal_period,metric,actual_value,consensus_value,actual_source,consensus_source,published_at`

Metric must be `EPS` or `REVENUE`. Import it with:

```bash
python -m scripts.ingest_indian_equities --earnings-csv path/to/earnings_consensus.csv
```

Fetch and persist historical daily prices by calling `GET /v1/market/INFY?period=5y` and the corresponding benchmark ticker. For the NIFTY 500 index, verify the correct provider symbol before use; index symbols are accepted with the leading `^`.

## API

- `GET /health` — process health.
- `GET /health/ready` — database readiness.
- `POST /v1/sentiment` — single-text FinBERT classification.
- `GET /v1/news?company=Infosys&limit=50` — fetch, classify, match and persist available news.
- `GET /v1/market/INFY?period=5y` — historical daily data and persistence.
- `GET /v1/universe/nifty500` — latest imported NIFTY 500 constituent snapshot.
- `GET /v1/instruments?q=Infosys` — issuer and instrument lookup.
- `GET /v1/announcements?symbol=INFY` — imported corporate announcements.
- `GET /v1/earnings/surprises` — actual-versus-consensus surprise records.
- `GET /v1/sectors/sentiment?days=7` — sector sentiment with sample coverage.
- `POST /v1/analytics/event-study` — event study from supplied return series; optionally persists the run.
- `POST /v1/analytics/event-study/from-stored-prices` — event study from previously stored daily prices.

## Data and research limitations

- NSE publishes downloadable equity security lists and a NIFTY 500 constituent CSV; BSE publishes a standardized scrip-master format. The importer uses those downloadable files and stores source provenance. Check source terms and permitted use before redistribution or commercial deployment.
- NIFTY 500 membership is captured as a dated snapshot on each import. The capture date is not represented as the official effective date; source_as_of_date remains null unless the source supplies an explicit effective date. Full historical constituent membership requires archived snapshots from prior dates; current constituents must not be backfilled into the past.
- Exchange announcement ingestion currently consumes official CSV exports downloaded by the operator. It does not claim automated real-time exchange polling. Any automation must use an authorized exchange feed or documented interface and comply with exchange terms.
- NewsAPI search is not a complete NSE/BSE corporate-announcement feed; plan and licensing restrictions apply.
- Earnings surprise calculation is implemented, but actual quarterly financial figures and consensus estimates must be supplied from identified sources. PDF/XBRL extraction and a licensed consensus provider are not connected.
- yfinance is for prototyping and historical research, not guaranteed real-time NSE/BSE quotes. Historical event studies should be rerun with adjusted, survivorship-aware, corporate-action-consistent price data before research conclusions are relied on.
- Sentiment scores classify text; they are not calibrated probabilities of returns or investment recommendations.
- Event studies are descriptive and sensitive to event-date quality, overlapping events, confounders, benchmark selection and look-ahead bias. They do not establish causality.
- Docker Compose credentials are development-only. Do not expose this database publicly or deploy with these defaults.
- Authentication, role-based access, scheduled workers, retries, production migrations, observability and deployment hardening remain future work.

## Next engineering milestones

1. Secure licensed NSE/BSE corporate announcement and historical market-data feeds.
2. Automate point-in-time instrument/universe snapshots with change detection and ingestion audits.
3. Add issuer-specific alias governance and review queue for ambiguous entity matches.
4. Connect structured quarterly financials and licensed consensus estimates; add EPS/revenue surprise history.
5. Validate sector sentiment against issuer and sector returns with time-aware leakage controls.
6. Build event studies with adjusted returns, benchmark controls, estimation windows, overlapping-event handling and statistical inference.
7. Add walk-forward model evaluation, transaction costs, model cards, access control and production deployment.
