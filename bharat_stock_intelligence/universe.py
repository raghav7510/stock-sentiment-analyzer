"""Official-source instrument and NIFTY 500 CSV normalization."""
from __future__ import annotations

from datetime import date
from io import BytesIO, StringIO
import re

import pandas as pd
import requests

NSE_EQUITY_MASTER_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
NIFTY_500_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 BharatStockIntelligence/0.3", "Accept": "text/csv,*/*"}


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in frame.columns]
    return frame


def _column(frame: pd.DataFrame, *names: str, required: bool = False) -> str | None:
    for name in names:
        key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        if key in frame.columns:
            return key
    if required:
        raise ValueError(f"Required CSV column missing; expected one of {names}, got {list(frame.columns)}")
    return None


def parse_nse_equity_master(csv_bytes: bytes) -> pd.DataFrame:
    frame = _clean_columns(pd.read_csv(BytesIO(csv_bytes), dtype=str))
    symbol = _column(frame, "symbol", required=True)
    name = _column(frame, "name_of_company", "company_name", "name")
    isin = _column(frame, "isin_number", "isin")
    series = _column(frame, "series")
    out = pd.DataFrame({
        "symbol": frame[symbol].str.strip().str.upper(),
        "company_name": frame[name].str.strip() if name else frame[symbol].str.strip(),
        "isin": frame[isin].str.strip() if isin else None,
        "series": frame[series].str.strip().str.upper() if series else "EQ",
        "exchange": "NSE",
        "sector": None,
        "source": "NSE securities available for equity segment",
    })
    out = out.dropna(subset=["symbol"]).drop_duplicates(["exchange", "symbol"])
    return out[out["series"].eq("EQ")].reset_index(drop=True)


def parse_nifty500_constituents(csv_bytes: bytes) -> pd.DataFrame:
    frame = _clean_columns(pd.read_csv(BytesIO(csv_bytes), dtype=str))
    symbol = _column(frame, "symbol", required=True)
    name = _column(frame, "company_name", "company")
    sector = _column(frame, "industry", "sector")
    isin = _column(frame, "isin_code", "isin")
    out = pd.DataFrame({
        "symbol": frame[symbol].str.strip().str.upper(),
        "company_name": frame[name].str.strip() if name else frame[symbol].str.strip(),
        "sector": frame[sector].str.strip() if sector else None,
        "isin": frame[isin].str.strip() if isin else None,
        "exchange": "NSE",
        "source": "NSE Indices NIFTY 500 constituent CSV",
        "captured_on": date.today(),
        "source_as_of_date": None,
    })
    out = out.dropna(subset=["symbol"]).drop_duplicates(["symbol"])
    if len(out) < 450 or len(out) > 550:
        raise ValueError(f"NIFTY 500 file sanity check failed: expected about 500 rows, received {len(out)}")
    return out.reset_index(drop=True)


def download_csv(url: str, timeout: int = 25) -> bytes:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    if not response.content:
        raise ValueError(f"Empty CSV response from {url}")
    return response.content


def fetch_official_nse_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download NSE equity master and current NIFTY 500 snapshot."""
    instruments = parse_nse_equity_master(download_csv(NSE_EQUITY_MASTER_URL))
    constituents = parse_nifty500_constituents(download_csv(NIFTY_500_URL))
    return instruments, constituents
