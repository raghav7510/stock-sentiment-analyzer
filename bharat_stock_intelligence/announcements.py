"""Normalize and classify official NSE/BSE announcement CSV exports.

This module deliberately does not scrape exchange endpoints. Import operator-downloaded
exchange exports and retain the original source URL and a deterministic deduplication key.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import hashlib
import re

import pandas as pd


CATEGORY_RULES = (
    ("financial_results", re.compile(r"\b(results?|financial results?|quarterly|annual results?|audited financials?|unaudited financials?|financial statements?)\b", re.I)),
    ("merger_acquisition", re.compile(r"\b(merger|amalgamation|acquisition|acquire[sd]?|scheme of arrangement|demerger|joint venture|takeover)\b", re.I)),
    ("fundraising", re.compile(r"\b(fund rais|fundraising|preferential issue|qip\b|qualified institutional|rights issue|public issue|private placement|debenture|bond issue|ncd\b|allotment of shares)\b", re.I)),
    ("regulatory_legal", re.compile(r"\b(sebi|regulatory|regulator|show cause|adjudication|penalty|fine imposed|court order|tribunal|litigation|insolvency|nclt|investigation|search and seizure)\b", re.I)),
    ("order_win", re.compile(r"\b(order (?:received|win|won|book)|letter of award|loa\b|contract (?:received|awarded)|work order|project award|purchase order)\b", re.I)),
    ("management_change", re.compile(r"\b(resignation|appointment|appointed|cessation|director|chief executive|ceo|cfo|managing director|key managerial|management change|auditor)\b", re.I)),
    ("dividend", re.compile(r"\b(dividend|interim dividend|final dividend|special dividend|record date|ex-dividend)\b", re.I)),
    ("credit_rating", re.compile(r"\b(credit rating|rating rationale|rating upgraded|rating downgraded|rating reaffirmed|crisil|icra|care ratings|india ratings)\b", re.I)),
    ("board_meeting", re.compile(r"\b(board meeting|meeting of board|outcome of board|board of directors)\b", re.I)),
    ("shareholding", re.compile(r"\b(shareholding pattern|substantial acquisition|disclosure under|pledge|encumbrance|insider trading|trading window)\b", re.I)),
    ("investor_presentation", re.compile(r"\b(investor presentation|analyst meet|investor meet|earnings call|conference call|transcript|corporate presentation)\b", re.I)),
)

HIGH_PRIORITY = re.compile(
    r"\b(financial results?|merger|amalgamation|acquisition|insolvency|nclt|"
    r"fraud|default|resignation of|cease operations|plant shutdown|重大)\b", re.I
)
MEDIUM_PRIORITY_CATEGORIES = {
    "order_win", "fundraising", "regulatory_legal", "management_change",
    "dividend", "credit_rating", "board_meeting",
}


def classify_announcement(subject: str, details: str | None = None) -> dict:
    """Return a transparent rule-based category and priority; this is not an ML prediction."""
    text = " ".join(part for part in (subject or "", details or "") if part)
    category = "other"
    for candidate, pattern in CATEGORY_RULES:
        if pattern.search(text):
            category = candidate
            break

    if HIGH_PRIORITY.search(subject or "") or category in {"financial_results", "merger_acquisition"}:
        priority = "high"
    elif category in MEDIUM_PRIORITY_CATEGORIES:
        priority = "medium"
    else:
        priority = "low"

    return {
        "category": category,
        "priority": priority,
        "classification_method": "keyword_rules_v1",
    }


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _parse_exchange_timestamp(value: str):
    """Exchange timestamps without offsets are interpreted as India Standard Time."""
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return None
    timestamp = pd.Timestamp(parsed)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Kolkata")
    else:
        timestamp = timestamp.tz_convert("Asia/Kolkata")
    return timestamp.tz_convert("UTC").to_pydatetime()


def parse_announcement_csv(csv_bytes: bytes, exchange: str = "NSE") -> list[dict]:
    """Parse common NSE/BSE export headers into a normalized, provenance-aware schema."""
    normalized_exchange = exchange.strip().upper()
    if normalized_exchange not in {"NSE", "BSE"}:
        raise ValueError("exchange must be NSE or BSE")
    if not csv_bytes:
        raise ValueError("Announcement CSV is empty")

    try:
        frame = pd.read_csv(BytesIO(csv_bytes), dtype=str, encoding="utf-8-sig", sep=None, engine="python").fillna("")
    except Exception as exc:
        raise ValueError(f"Could not read announcement CSV: {type(exc).__name__}") from exc

    frame.columns = [_key(c) for c in frame.columns]
    aliases = {
        "symbol": (
            "symbol", "security_symbol", "scrip_code", "scrip_cd", "security_code",
            "scripcode", "security_id",
        ),
        "company_name": (
            "company_name", "company", "security_name", "long_name", "slongname",
            "issuer_name", "companyname",
        ),
        "subject": (
            "subject", "announcement_subject", "headline", "news_sub", "news_subject",
            "subject_of_announcement",
        ),
        "details": (
            "details", "description", "announcement_text", "news_dtls", "news_details",
            "brief_description",
        ),
        "broadcast_at": (
            "broadcast_date_time", "broadcast_datetime", "date", "announcement_date",
            "news_dt", "news_date", "dissemination_date_time", "dissemination_datetime",
        ),
        "url": (
            "attachment", "attachment_url", "link", "url", "NSURL".lower(),
            "news_url", "pdf_url", "attchmntfile",
        ),
    }
    selected = {
        field: next((name for name in names if name in frame.columns), None)
        for field, names in aliases.items()
    }
    if not selected["subject"] or not (selected["symbol"] or selected["company_name"]):
        raise ValueError(
            "Announcement CSV needs a recognized subject and company/symbol column; "
            f"found columns: {list(frame.columns)}"
        )

    rows = []
    for raw in frame.to_dict("records"):
        subject = raw.get(selected["subject"], "").strip()
        if not subject:
            continue
        symbol = raw.get(selected["symbol"], "").strip().upper() if selected["symbol"] else ""
        company = raw.get(selected["company_name"], "").strip() if selected["company_name"] else ""
        details = raw.get(selected["details"], "").strip() if selected["details"] else ""
        timestamp_raw = raw.get(selected["broadcast_at"], "").strip() if selected["broadcast_at"] else ""
        published_at = _parse_exchange_timestamp(timestamp_raw)
        url = raw.get(selected["url"], "").strip() if selected["url"] else ""
        classification = classify_announcement(subject, details)

        # Exclude retrieval time from identity so re-importing the same source row is idempotent.
        identity = "|".join([
            normalized_exchange, symbol, company.casefold(), subject.casefold(),
            published_at.isoformat() if published_at else timestamp_raw, url,
        ])
        rows.append({
            "exchange": normalized_exchange,
            "symbol": symbol or None,
            "company_name": company or None,
            "subject": subject,
            "details": details or None,
            "published_at": published_at,
            "source_url": url or None,
            "source_hash": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            "retrieved_at": datetime.now(timezone.utc),
            "source_format": "exchange_csv_export",
            **classification,
        })
    return rows
