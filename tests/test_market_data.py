import pytest

from bharat_stock_intelligence.market_data import resolve_indian_ticker


@pytest.mark.parametrize(("raw", "expected"), [
    ("INFY", "INFY.NS"),
    ("infy", "INFY.NS"),
    ("TCS.NS", "TCS.NS"),
    ("500209.BO", "500209.BO"),
    ("^CRSLDX", "^CRSLDX"),
])
def test_resolve_indian_ticker(raw, expected):
    assert resolve_indian_ticker(raw) == expected


def test_empty_ticker_rejected():
    with pytest.raises(ValueError):
        resolve_indian_ticker(" ")
