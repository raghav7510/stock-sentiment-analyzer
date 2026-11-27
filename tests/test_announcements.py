from datetime import timezone

import pytest

from bharat_stock_intelligence.announcements import (
    classify_announcement,
    parse_announcement_csv,
)


def test_parse_nse_style_announcement_export():
    raw = (
        b"SYMBOL,COMPANY NAME,SUBJECT,DETAILS,BROADCAST DATE/TIME,ATTACHMENT\n"
        b"INFY,Infosys Limited,Outcome of Board Meeting,Quarterly results,"
        b"27-Nov-2026 18:00:00,https://example.com/filing.pdf\n"
    )
    rows = parse_announcement_csv(raw, "NSE")
    assert rows[0]["symbol"] == "INFY"
    assert rows[0]["subject"] == "Outcome of Board Meeting"
    assert rows[0]["published_at"] is not None
    assert rows[0]["published_at"].tzinfo == timezone.utc
    assert rows[0]["published_at"].hour == 12  # 18:00 IST converted to 12:30 UTC
    assert rows[0]["source_url"].startswith("https://")
    assert rows[0]["category"] == "financial_results"
    assert rows[0]["priority"] == "high"
    assert rows[0]["classification_method"] == "keyword_rules_v1"


def test_parse_bse_style_export_headers():
    raw = (
        b"SCRIP_CD,SLONGNAME,NEWS_SUB,NEWS_DT,NSURL\n"
        b"500209,Infosys Limited,Order received,27-Nov-2026 18:00:00,"
        b"https://example.com/bse.pdf\n"
    )
    rows = parse_announcement_csv(raw, "BSE")
    assert rows[0]["symbol"] == "500209"
    assert rows[0]["company_name"] == "Infosys Limited"
    assert rows[0]["category"] == "order_win"
    assert rows[0]["priority"] == "medium"
    assert rows[0]["published_at"].hour == 12


def test_reimport_identity_is_stable():
    raw = (
        b"SYMBOL,COMPANY NAME,SUBJECT,BROADCAST DATE/TIME\n"
        b"INFY,Infosys Limited,Dividend declared,27-Nov-2026 18:00:00\n"
    )
    first = parse_announcement_csv(raw, "NSE")
    second = parse_announcement_csv(raw, "NSE")
    assert first[0]["source_hash"] == second[0]["source_hash"]


def test_classification_is_rule_based_and_transparent():
    result = classify_announcement("Outcome of quarterly financial results")
    assert result == {
        "category": "financial_results",
        "priority": "high",
        "classification_method": "keyword_rules_v1",
    }


def test_announcement_export_requires_issuer_and_subject():
    with pytest.raises(ValueError):
        parse_announcement_csv(b"SUBJECT,DETAILS\nDividend,Declared\n")


def test_announcement_exchange_is_validated():
    with pytest.raises(ValueError, match="exchange must be NSE or BSE"):
        parse_announcement_csv(b"SYMBOL,SUBJECT\nINFY,Dividend\n", "NYSE")


def test_empty_csv_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        parse_announcement_csv(b"", "NSE")
