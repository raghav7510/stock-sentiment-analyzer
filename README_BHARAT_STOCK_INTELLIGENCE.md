# Bharat Stock Intelligence

An India-focused financial research platform. This implementation is introduced alongside the legacy Streamlit prototype so existing behaviour remains available while the new foundation is validated.

## Foundation (v0.1)

- FastAPI service with health, FinBERT sentiment, news-analysis and daily-market-history endpoints.
- Pretrained ProsusAI/finbert inference and validated weighted sentiment aggregation.
- News ingestion that fails clearly when NEWS_API_KEY is absent; it never fabricates demo headlines.
- Yahoo Finance daily historical data adapter, explicitly labelled as historical and not live exchange data.
- PostgreSQL/TimescaleDB schema for instruments, articles, sentiment and daily prices.
- Docker Compose database service and initial unit tests.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-bsi.txt
```

Create an uncommitted `.env` file:

```env
NEWS_API_KEY=your_newsapi_key
DATABASE_URL=postgresql+psycopg://bsi:bsi@localhost:5432/bsi
FINBERT_MODEL=ProsusAI/finbert
```

Start the API: `uvicorn api.main:app --reload`. Open `http://127.0.0.1:8000/docs`.

Start TimescaleDB: `docker compose -f docker-compose.bsi.yml up -d`.

Apply schema: `psql postgresql://bsi:bsi@localhost:5432/bsi -f sql/001_initial_schema.sql`.

Run tests: `pytest -q`.

## API

- `GET /health`
- `POST /v1/sentiment` with `{"text":"Company reports stronger quarterly revenue"}`
- `GET /v1/news?company=Infosys&limit=50`
- `GET /v1/market/INFY?period=1y`

## Limitations

- NewsAPI access and licensing/plan restrictions apply; search results are not a complete exchange-announcement feed.
- yfinance is for prototyping and historical research, not guaranteed real-time NSE/BSE quotes. Responses state this explicitly.
- No BUY/SELL recommendations, price forecasts or order execution are implemented.
- Sentiment scores classify text; they are not calibrated probabilities of returns. Do not trade from them.
- The database schema is an initial foundation; API persistence and migrations are not yet wired in.
- Model training is not enabled until datasets, labels, time-aware validation and baseline metrics are audited.

## Roadmap

1. Instrument master and NSE/BSE corporate-announcement ingestion.
2. Persistent news ingestion, entity resolution, deduplication and audit trail.
3. Background workers and incremental market-data ingestion from a licensed provider.
4. Streamlit/Plotly research dashboard with source timestamps and coverage indicators.
5. Event classification, sector/market-relative analytics and event studies.
6. Walk-forward backtesting with transaction costs, slippage and out-of-sample reporting.
7. Monitoring, access control, deployment hardening and operational runbooks.
