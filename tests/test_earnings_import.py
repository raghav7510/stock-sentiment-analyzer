import pytest

from bharat_stock_intelligence.earnings_import import parse_earnings_csv


def test_parse_earnings_provider_export():
    raw = b"symbol,fiscal_period,metric,actual_value,consensus_value,actual_source,consensus_source\nINFY,Q2 FY27,EPS,18,16,Company filing,Licensed consensus\n"
    rows = parse_earnings_csv(raw)
    assert rows[0]["direction"] == "beat"
    assert rows[0]["surprise_pct"] == 12.5


def test_earnings_import_requires_provenance():
    raw = b"symbol,fiscal_period,metric,actual_value,consensus_value\nINFY,Q2,EPS,18,16\n"
    with pytest.raises(ValueError, match="missing columns"):
        parse_earnings_csv(raw)
