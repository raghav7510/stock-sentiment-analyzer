"""Import actual-versus-consensus earnings data from licensed/provider CSV exports."""
from __future__ import annotations

from datetime import datetime
from io import BytesIO

import pandas as pd

from .earnings import calculate_surprise


def parse_earnings_csv(csv_bytes: bytes) -> list[dict]:
    frame = pd.read_csv(BytesIO(csv_bytes), dtype=str).fillna("")
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    required = {
        "symbol", "fiscal_period", "metric", "actual_value", "consensus_value",
        "actual_source", "consensus_source",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Earnings CSV missing columns: {sorted(missing)}")
    rows = []
    for row in frame.to_dict("records"):
        metric = row["metric"].strip().upper()
        if metric not in {"EPS", "REVENUE"}:
            raise ValueError(f"Unsupported earnings metric: {metric}")
        actual, estimate = float(row["actual_value"]), float(row["consensus_value"])
        result = calculate_surprise(actual, estimate)
        published_at = None
        if row.get("published_at"):
            published_at = pd.to_datetime(row["published_at"], errors="coerce", utc=True)
            if pd.isna(published_at):
                raise ValueError(f"Invalid published_at for {row['symbol']}")
            published_at = published_at.to_pydatetime()
        rows.append({
            "symbol": row["symbol"].strip().upper(),
            "exchange": row.get("exchange", "NSE").strip().upper() or "NSE",
            "fiscal_period": row["fiscal_period"].strip(),
            "metric": metric,
            "actual_value": actual,
            "consensus_value": estimate,
            "absolute_surprise": result["surprise"],
            "surprise_pct": result["surprise_pct"],
            "direction": result["direction"],
            "actual_source": row["actual_source"].strip(),
            "consensus_source": row["consensus_source"].strip(),
            "published_at": published_at,
        })
    return rows
