"""Descriptive market-adjusted event studies; not a predictive trading model."""
from __future__ import annotations

import pandas as pd


def calculate_event_study(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    event_dates: list[str | pd.Timestamp],
    window_before: int = 5,
    window_after: int = 5,
) -> pd.DataFrame:
    """Calculate market-adjusted returns and CAR around dated events.

    Inputs must be daily simple returns indexed by trading date. Dates absent
    from the return index are skipped rather than silently shifted to another day.
    """
    if window_before < 0 or window_after < 0:
        raise ValueError("Event windows must be non-negative.")
    asset = pd.to_numeric(asset_returns, errors="coerce").dropna().sort_index()
    benchmark = pd.to_numeric(benchmark_returns, errors="coerce").dropna().sort_index()
    if asset.index.has_duplicates or benchmark.index.has_duplicates:
        raise ValueError("Return indexes must not contain duplicate dates.")

    aligned = pd.concat(
        [asset.rename("asset_return"), benchmark.rename("benchmark_return")],
        axis=1,
        join="inner",
    ).dropna()
    aligned["abnormal_return"] = aligned["asset_return"] - aligned["benchmark_return"]
    rows = []
    for raw_date in event_dates:
        event_date = pd.Timestamp(raw_date)
        if event_date not in aligned.index:
            continue
        event_pos = aligned.index.get_loc(event_date)
        start, end = max(0, event_pos - window_before), min(len(aligned), event_pos + window_after + 1)
        window = aligned.iloc[start:end].copy()
        window["event_day"] = range(start - event_pos, end - event_pos)
        window["car"] = window["abnormal_return"].cumsum()
        for date, record in window.iterrows():
            rows.append({
                "event_date": event_date,
                "trade_date": date,
                "event_day": int(record["event_day"]),
                "asset_return": float(record["asset_return"]),
                "benchmark_return": float(record["benchmark_return"]),
                "abnormal_return": float(record["abnormal_return"]),
                "car": float(record["car"]),
            })
    return pd.DataFrame(rows, columns=[
        "event_date", "trade_date", "event_day", "asset_return",
        "benchmark_return", "abnormal_return", "car",
    ])
