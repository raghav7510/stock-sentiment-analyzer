"""Streamlit research interface for Bharat Stock Intelligence."""
import os
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

API_URL = os.getenv("BSI_API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="Bharat Stock Intelligence", page_icon="📊", layout="wide")
st.title("Bharat Stock Intelligence")
st.caption("AI-powered Indian equity research | Research analytics, not investment advice")

with st.sidebar:
    st.subheader("Research controls")
    company = st.text_input("Company name / ticker", "Infosys")
    ticker = st.text_input("NSE ticker", "INFY").strip().upper()
    limit = st.slider("Maximum news articles", 10, 100, 50, 10)
    period = st.selectbox("Price history", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    run = st.button("Run research", type="primary", use_container_width=True)

if run:
    try:
        with st.spinner("Retrieving news and running FinBERT..."):
            response = requests.get(f"{API_URL}/v1/news", params={"company": company, "limit": limit}, timeout=180)
            response.raise_for_status()
            news_data = response.json()
        with st.spinner("Loading daily historical market data..."):
            market_response = requests.get(f"{API_URL}/v1/market/{ticker}", params={"period": period}, timeout=45)
            market_response.raise_for_status()
            market_data = market_response.json()
        st.session_state["bsi_news"] = news_data
        st.session_state["bsi_market"] = market_data
    except requests.RequestException as exc:
        st.error(f"Research request failed: {exc}")

news_data = st.session_state.get("bsi_news")
market_data = st.session_state.get("bsi_market")
if news_data:
    summary = news_data["summary"]
    a, b, c = st.columns(3)
    a.metric("Aggregate sentiment", f'{summary["score"]:+.3f}' if summary["coverage"] != "no_data" else "No data")
    b.metric("Articles analysed", summary["article_count"])
    c.metric("Sentiment classification", "Positive" if summary["score"] > 0.1 else "Negative" if summary["score"] < -0.1 else "Neutral / mixed")
    st.caption("Sentiment score is a text-classification aggregate, not a return forecast. Review source coverage before interpreting it.")
    st.caption(f"News source status: {news_data.get('data_status', 'unknown')} | Database persistence: {news_data.get('persistence_status', 'unknown')}")
    frame = pd.DataFrame(news_data["articles"])
    if not frame.empty:
        st.subheader("News sentiment distribution")
        counts = frame["sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Articles")
        st.plotly_chart(px.bar(counts, x="Sentiment", y="Articles", color="Sentiment"), use_container_width=True)
        st.subheader("Analysed headlines")
        cols = [c for c in ["published_at", "source", "sentiment", "score", "confidence", "title", "url"] if c in frame.columns]
        st.dataframe(frame[cols], use_container_width=True, hide_index=True)
if market_data:
    st.subheader(f'{market_data["ticker"]} — daily historical prices')
    st.warning("Historical daily data from Yahoo Finance. This is not a live NSE/BSE quote or executable price.")
    prices = pd.DataFrame(market_data["rows"])
    if not prices.empty:
        prices["Date"] = pd.to_datetime(prices["Date"])
        st.plotly_chart(px.line(prices, x="Date", y="Close", title="Adjusted daily close"), use_container_width=True)
        st.dataframe(prices.tail(20), use_container_width=True, hide_index=True)
else:
    st.info("Enter a company and ticker, then run research. Start the API before using this dashboard.")
