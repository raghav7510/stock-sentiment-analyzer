"""Parser for the BSE standardized scrip-master CSV format."""
from __future__ import annotations

from io import BytesIO

import pandas as pd


def parse_bse_scrip_master(csv_bytes: bytes) -> pd.DataFrame:
    frame = pd.read_csv(BytesIO(csv_bytes), dtype=str).fillna("")
    normalized = {str(c).strip().lower().replace(" ", "_"): c for c in frame.columns}
    def col(*names):
        for name in names:
            if name in normalized:
                return normalized[name]
        return None
    code, symbol, name, isin = (
        col("scrip_code", "security_code"),
        col("instrument_code", "symbol"),
        col("scrip_name", "security_name"),
        col("isin_code", "isin"),
    )
    if not code or not name:
        raise ValueError(f"BSE scrip master missing required columns; got {list(frame.columns)}")
    out = pd.DataFrame({
        "symbol": frame[symbol].str.strip().str.upper() if symbol else frame[code].str.strip(),
        "company_name": frame[name].str.strip(),
        "isin": frame[isin].str.strip() if isin else None,
        "series": "EQ",
        "exchange": "BSE",
        "sector": None,
        "source": "BSE standardized scrip master CSV",
    })
    out = out[out["symbol"].ne("") & out["company_name"].ne("")]
    return out.drop_duplicates(["exchange", "symbol"]).reset_index(drop=True)
