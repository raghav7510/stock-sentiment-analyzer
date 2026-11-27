# Bharat Stock Intelligence

An India-focused financial research platform. The new implementation is developed alongside the legacy Streamlit prototype so existing behaviour remains available while the new foundation is validated.

## Foundation (v0.2)

- FastAPI service with health/readiness, FinBERT sentiment, news-analysis, historical market-data and event-study endpoints.
- Pretrained ProsusAI/finbert inference with model-metadata class ordering.
- Dependency-light weighted sentiment aggregation; missing coverage is never labelled neutral.
- News ingestion with canonical-URL deduplication and title-plus-description model input. Missing credentials never trigger fabricated news.
- Idempotent PostgreSQL persistence for articles and model-specific sentiment outputs.
- TimescaleDB schema, Docker Compose database service, structured JSON logging and pytest coverage.
- Market-adjusted event-study calculations using aligned daily asset and benchmark returns.
- Yahoo Finance daily history adapter explicitly labelled historical, not live exchange data.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
# Install CPU PyTorch first (for CUDA, use the official PyTorch selector)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-bsi.txt
```

Copy `.env.example` to `.env` and add your own NewsAPI key. Never commit credentials.

Start TimescaleDB: `docker compose -f docker-compose.bsi.yml up -d`.

Apply schema: `psql postgresql://bsi:bsi@localhost:5432/bsi -f sql/001_initial_schema.sql`.

Start API: `uvicorn api.main:app --reload`. API docs: `http://127.0.0.1:8000/docs`.

Start dashboard in another terminal: `streamlit run dashboard_bsi.py`.

Run tests: `python -m pytest tests/ -q`.

## API

- `GET /health` — process health.
- `GET /health/ready` — database readiness.
- `POST /v1/sentiment` — single-text FinBERT classification.
- `GET /v1/news?company=Infosys&limit=50` — fetch, classify and persist available news.
- `GET /v1/market/INFY?period=1y` — daily historical data.
- `POST /v1/analytics/event-study` — market-adjusted event-window analysis.

## Data and research limitations

- NewsAPI search is not a complete NSE/BSE corporate-announcement feed; licensing and plan limits apply.
- yfinance is for prototyping and historical research, not guaranteed real-time exchange quotes.
- No licensed live NSE/BSE feed is configured. Do not use this dashboard for executable prices.
- Sentiment scores classify text; they are not calibrated probabilities of returns or investment recommendations.
- Event studies are descriptive and sensitive to event-date quality, overlapping events, confounders and benchmark selection. They do not establish causality.
- Database persistence currently covers news and sentiment; instrument and price ingestion are not yet connected to TimescaleDB.
- No NIFTY 500 master, official exchange announcement connector, event classifier, earnings-surprise engine, production backtest or predictive model is claimed complete.
- Docker Compose credentials are development-only. Do not expose this database publicly or deploy with these defaults.

## Next engineering milestones

1. Verified instrument master with source provenance and point-in-time membership history.
2. Compliant NSE/BSE corporate-announcement ingestion and issuer/entity resolution.
3. Licensed market-data adapter, historical price persistence and market/sector benchmarks.
4. Sector analytics, earnings-surprise extraction and announcement/event taxonomy.
5. Walk-forward evaluation, leakage controls, transaction-cost-aware backtesting and model cards.
6. Production authentication, secrets management, migrations, observability and deployment hardening.
