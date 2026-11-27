from bharat_stock_intelligence.announcements import parse_announcement_csv


def test_parse_nse_style_announcement_export():
    raw = b"SYMBOL,COMPANY NAME,SUBJECT,DETAILS,BROADCAST DATE/TIME,ATTACHMENT\nINFY,Infosys Limited,Outcome of Board Meeting,Quarterly results,27-Nov-2026 18:00:00,https://example.com/filing.pdf\n"
    rows = parse_announcement_csv(raw, "NSE")
    assert rows[0]["symbol"] == "INFY"
    assert rows[0]["subject"] == "Outcome of Board Meeting"
    assert rows[0]["published_at"] is not None
    assert rows[0]["source_url"].startswith("https://")


def test_announcement_export_requires_issuer_and_subject():
    import pytest
    with pytest.raises(ValueError):
        parse_announcement_csv(b"SUBJECT,DETAILS\nDividend,Declared\n")
