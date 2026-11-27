"""Normalize exchange announcement CSV exports without scraping private endpoints."""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import hashlib
import re

import pandas as pd


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def parse_announcement_csv(csv_bytes: bytes, exchange: str = "NSE") -> list[dict]:
    frame = pd.read_csv(BytesIO(csv_bytes), dtype=str).fillna("")
    frame.columns = [_key(c) for c in frame.columns]
    aliases = {
        "symbol": ("symbol", "security_symbol", "scrip_code"),
        "company_name": ("company_name", "company", "security_name"),
        "subject": ("subject", "announcement_subject", "headline"),
        "details": ("details", "description", "announcement_text"),
        "broadcast_at": ("broadcast_date_time", "broadcast_datetime", "date", "announcement_date"),
        "url": ("attachment", "attachment_url", "link", "url"),
    }
    selected = {}
    for field, names in aliases.items():
        selected[field] = next((n for n in names if n in frame.columns), None)
    if not selected["subject"] or not (selected["symbol"] or selected["company_name"]):
        raise ValueError(f"Announcement CSV needs subject and company/symbol columns; got {list(frame.columns)}")
    rows = []
    for raw in frame.to_dict("records"):
        subject = raw.get(selected["subject"], "").strip()
        if not subject:
            continue
        symbol = raw.get(selected["symbol"], "").strip().upper() if selected["symbol"] else ""
        company = raw.get(selected["company_name"], "").strip() if selected["company_name"] else ""
        details = raw.get(selected["details"], "").strip() if selected["details"] else ""
        timestamp_raw = raw.get(selected["broadcast_at"], "").strip() if selected["broadcast_at"] else ""
        published_at = None
        if timestamp_raw:
            parsed = pd.to_datetime(timestamp_raw, errors="coerce", dayfirst=True, utc=True)
            if not pd.isna(parsed):
                published_at = parsed.to_pydatetime()
        url = raw.get(selected["url"], "").strip() if selected["url"] else ""
        identity = "|".join([exchange.upper(), symbol, company, subject, str(published_at), url])
        rows.append({
            "exchange": exchange.upper(),
            "symbol": symbol or None,
            "company_name": company or None,
            "subject": subject,
            "details": details or None,
            "published_at": published_at,
            "source_url": url or None,
            "source_hash": hashlib.sha256(identity.encode()).hexdigest(),
            "retrieved_at": datetime.now(timezone.utc),
            "source_format": "exchange_csv_export",
        })
    return rows
