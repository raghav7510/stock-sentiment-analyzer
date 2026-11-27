"""Streamlit research dashboard for Bharat Stock Intelligence."""
from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("BSI_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = (5, 180)

st.set_page_config(page_title="Bharat Stock Intelligence", page_icon="📊", layout="wide")
st.title("Bharat Stock Intelligence")
st.caption("India-focused equity research | Evidence-led analytics, not investment advice")


def api_get(path: str, params: dict | None = None):
    response = requests.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict):
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def show_error(label: str, exc: Exception):
    detail = getattr(getattr(exc, "response", None), "text", "")
    st.error(f"{label}: {detail[:500] or str(exc)}")


def show_table(rows, columns=None):
    frame = pd.DataFrame(rows or [])
    if frame.empty:
        st.info("No records are available yet. Import the relevant source data first.")
        return
    if columns:
        frame = frame[[column for column in columns if column in frame.columns]]
    st.dataframe(frame, use_container_width=True, hide_index=True)


with st.sidebar:
    st.subheader("Research controls")
    company = st.text_input("Company / news query", "Infosys")
    ticker = st.text_input("NSE ticker", "INFY").strip().upper()
    news_limit = st.slider("News articles", 10, 100, 50, 10)
    price_period = st.selectbox("Historical price window", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    run_research = st.button("Run company research", type="primary", use_container_width=True)
    st.caption(f"API: {API_URL}")

if run_research:
    try:
        with st.spinner("Running news classification and loading historical prices..."):
            st.session_state["bsi_news"] = api_get("/v1/news", {"company": company, "limit": news_limit})
            st.session_state["bsi_market"] = api_get(f"/v1/market/{ticker}", {"period": price_period})
    except requests.RequestException as exc:
        show_error("Company research failed", exc)

try:
    health = api_get("/health/ready")
    st.sidebar.success(f"API {health.get('status')} · DB {health.get('database')}")
except requests.RequestException:
    st.sidebar.error("API/database unavailable")

overview, universe, announcements, earnings, sectors, event_study, company_tab, operations = st.tabs([
    "Overview", "NIFTY 500", "Announcements", "Earnings", "Sector intelligence",
    "Event study", "Company research", "Ingestion operations"
])

with overview:
    st.subheader("Platform overview")
    col1, col2, col3 = st.columns(3)
    try:
        u = api_get("/v1/universe/nifty500")
        col1.metric("NIFTY 500 constituents", u.get("count", 0))
        col1.caption(f"Snapshot captured: {u.get('capture_date') or 'unknown'}")
    except requests.RequestException:
        col1.metric("NIFTY 500 constituents", "Unavailable")
    try:
        a = api_get("/v1/announcements", {"limit": 1})
        col2.metric("Announcement records", len(a.get("announcements", [])) if a else 0)
        col2.caption("Latest record preview only; use Announcements tab for full list")
    except requests.RequestException:
        col2.metric("Announcements", "Unavailable")
    try:
        e = api_get("/v1/earnings/surprises", {"limit": 1})
        col3.metric("Earnings surprise records", len(e.get("surprises", [])) if e else 0)
        col3.caption("Latest record preview only; use Earnings tab for full list")
    except requests.RequestException:
        col3.metric("Earnings records", "Unavailable")
    st.info("Coverage depends on the files and providers actually ingested. Empty datasets are not interpreted as neutral signals.")

with universe:
    st.subheader("NIFTY 500 universe")
    if st.button("Load latest universe snapshot"):
        try:
            st.session_state["bsi_universe"] = api_get("/v1/universe/nifty500")
        except requests.RequestException as exc:
            show_error("Universe lookup failed", exc)
    data = st.session_state.get("bsi_universe")
    if data:
        st.caption(f"{data.get('count', 0)} constituents · captured {data.get('capture_date')} · source effective date {data.get('source_effective_date') or 'not supplied'}")
        frame = pd.DataFrame(data.get("members", []))
        if not frame.empty:
            search = st.text_input("Filter company, symbol or sector", key="universe_filter").strip().lower()
            if search:
                frame = frame[frame.astype(str).apply(lambda col: col.str.lower().str.contains(search, regex=False)).any(axis=1)]
            st.dataframe(frame, use_container_width=True, hide_index=True)

with announcements:
    st.subheader("Corporate announcements")
    ann_symbol = st.text_input("Filter by symbol (optional)", key="ann_symbol").strip().upper()
    ann_limit = st.slider("Maximum announcements", 20, 500, 100, 20)
    if st.button("Load announcements"):
        try:
            st.session_state["bsi_announcements"] = api_get("/v1/announcements", {"symbol": ann_symbol or None, "limit": ann_limit}).get("announcements", [])
        except requests.RequestException as exc:
            show_error("Announcement lookup failed", exc)
    show_table(st.session_state.get("bsi_announcements", []),
               ["published_at", "exchange", "symbol", "company_name", "category", "priority", "subject", "source_url"])

with earnings:
    st.subheader("Earnings surprises")
    earnings_limit = st.slider("Maximum earnings records", 20, 500, 100, 20)
    if st.button("Load earnings surprises"):
        try:
            st.session_state["bsi_earnings"] = api_get("/v1/earnings/surprises", {"limit": earnings_limit}).get("surprises", [])
        except requests.RequestException as exc:
            show_error("Earnings lookup failed", exc)
    earnings_rows = st.session_state.get("bsi_earnings", [])
    show_table(earnings_rows, ["published_at", "symbol", "company_name", "sector", "fiscal_period", "metric",
                               "actual_value", "consensus_value", "surprise_pct", "direction", "actual_source", "consensus_source"])
    if earnings_rows:
        ef = pd.DataFrame(earnings_rows)
        if "direction" in ef:
            counts = ef["direction"].value_counts().rename_axis("Outcome").reset_index(name="Records")
            st.plotly_chart(px.bar(counts, x="Outcome", y="Records", title="Reported earnings outcomes"), use_container_width=True)

with sectors:
    st.subheader("Sector sentiment")
    lookback = st.slider("Lookback (days)", 1, 90, 7)
    if st.button("Calculate sector sentiment"):
        try:
            st.session_state["bsi_sectors"] = api_get("/v1/sectors/sentiment", {"days": lookback})
        except requests.RequestException as exc:
            show_error("Sector analytics failed", exc)
    sector_data = st.session_state.get("bsi_sectors")
    if sector_data:
        st.caption(f"Coverage: {sector_data.get('coverage')} · lookback: {sector_data.get('lookback_days')} days")
        sector_rows = sector_data.get("sectors", [])
        show_table(sector_rows, ["sector", "sentiment_score", "article_count", "company_count", "latest_article_at"])
        if sector_rows:
            sf = pd.DataFrame(sector_rows)
            st.plotly_chart(px.bar(sf.sort_values("sentiment_score"), x="sentiment_score", y="sector",
                                   orientation="h", color="sentiment_score", title="Average FinBERT sentiment by sector"),
                            use_container_width=True)

with event_study:
    st.subheader("Historical event impact")
    st.warning("Descriptive market-adjusted event study only. Historical Yahoo Finance data is not live, and abnormal returns do not establish causality.")
    with st.form("event_study_form"):
        event_ticker = st.text_input("Instrument ticker", ticker or "INFY").strip().upper()
        benchmark = st.text_input("Benchmark provider ticker", "^CRSLDX")
        event_day = st.date_input("Event date", value=date.today() - timedelta(days=7))
        before = st.slider("Trading sessions before", 0, 30, 5)
        after = st.slider("Trading sessions after", 0, 30, 5)
        event_kind = st.selectbox("Event type", ["corporate_announcement", "earnings", "order_win", "management_change", "other"])
        submit_event = st.form_submit_button("Fetch history and run event study")
    if submit_event:
        try:
            with st.spinner("Fetching historical prices and calculating abnormal returns..."):
                api_get(f"/v1/market/{event_ticker}", {"period": "5y"})
                api_get(f"/v1/market/{benchmark}", {"period": "5y"})
                result = api_post("/v1/analytics/event-study/from-stored-prices", {
                    "instrument_symbol": event_ticker,
                    "instrument_exchange": "NSE",
                    "benchmark_symbol": benchmark,
                    "event_dates": [event_day.isoformat()],
                    "event_type": event_kind,
                    "window_before": before,
                    "window_after": after,
                })
            st.session_state["bsi_event_result"] = result
        except requests.RequestException as exc:
            show_error("Event study failed", exc)
    event_result = st.session_state.get("bsi_event_result")
    if event_result:
        st.caption(f"Method: {event_result.get('method')} · Data: {event_result.get('data_status')}")
        observations = event_result.get("observations", [])
        show_table(observations)
        if observations:
            of = pd.DataFrame(observations)
            if {"relative_day", "abnormal_return"}.issubset(of.columns):
                st.plotly_chart(px.bar(of, x="relative_day", y="abnormal_return", title="Market-adjusted abnormal return by event time"), use_container_width=True)
        st.caption(event_result.get("interpretation", ""))

with company_tab:
    st.subheader("Company news and price research")
    news_data = st.session_state.get("bsi_news")
    market_data = st.session_state.get("bsi_market")
    if news_data:
        summary = news_data.get("summary", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Aggregate sentiment", f'{summary.get("score", 0):+.3f}' if summary.get("coverage") != "no_data" else "No data")
        c2.metric("Articles analysed", summary.get("article_count", 0))
        c3.metric("Matched issuers", news_data.get("matched_company_count", 0))
        st.caption(f"News provider status: {news_data.get('data_status')} · persistence: {news_data.get('persistence_status')}")
        rows = news_data.get("articles", [])
        if rows:
            nf = pd.DataFrame(rows)
            if "sentiment" in nf:
                counts = nf["sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Articles")
                st.plotly_chart(px.bar(counts, x="Sentiment", y="Articles", title="News sentiment distribution"), use_container_width=True)
            show_table(rows, ["published_at", "source", "sentiment", "score", "confidence", "title", "url"])
    if market_data:
        st.subheader(f"{market_data.get('ticker')} · daily historical prices")
        st.caption(f"Provider: {market_data.get('provider')} · status: {market_data.get('data_status')}")
        prices = pd.DataFrame(market_data.get("rows", []))
        if not prices.empty and {"Date", "Close"}.issubset(prices.columns):
            prices["Date"] = pd.to_datetime(prices["Date"])
            st.plotly_chart(px.line(prices, x="Date", y="Close", title="Adjusted daily close"), use_container_width=True)
            show_table(prices.tail(20).to_dict("records"))
    if not news_data and not market_data:
        st.info("Enter a company and ticker in the sidebar, then select Run company research.")

with operations:
    st.subheader("Ingestion operations")
    audit_limit = st.slider("Recent ingestion attempts", 5, 100, 20, 5)
    if st.button("Refresh ingestion audit"):
        try:
            st.session_state["bsi_ingestion"] = api_get("/v1/ingestion/status", {"limit": audit_limit}).get("runs", [])
        except requests.RequestException as exc:
            show_error("Ingestion audit lookup failed", exc)
    runs = st.session_state.get("bsi_ingestion", [])
    show_table(runs, ["started_at", "source", "dataset", "status", "rows_seen", "rows_written", "checksum", "error_message"])
    if runs:
        st.metric("Successful attempts in view", sum(row.get("status") == "success" for row in runs))
        st.metric("Failed attempts in view", sum(row.get("status") == "failed" for row in runs))
