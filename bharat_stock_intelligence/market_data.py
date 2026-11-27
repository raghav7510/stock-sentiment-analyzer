"""Yahoo Finance adapter for prototyping; not an exchange-grade live feed."""
import pandas as pd
import yfinance as yf


def resolve_indian_ticker(ticker: str) -> str:
    value = ticker.strip().upper()
    if not value:
        raise ValueError("Ticker is required.")
    if value.startswith("^") or value.endswith((".NS", ".BO")):
        return value
    return f"{value}.NS"


def get_daily_history(ticker: str, period: str = "1y") -> dict:
    symbol = resolve_indian_ticker(ticker)
    history = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
    if history.empty:
        raise LookupError(f"No daily market history returned for {symbol}.")
    history = history.reset_index()
    history["Date"] = pd.to_datetime(history["Date"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    selected = history[["Date", "Open", "High", "Low", "Close", "Volume"]]
    rows = selected.astype(object).where(pd.notna(selected), None).to_dict("records")
    return {
        "ticker": symbol,
        "data_status": "historical_daily_not_live",
        "provider": "Yahoo Finance via yfinance",
        "rows": rows,
    }
