import pandas as pd
import pytest

from bharat_stock_intelligence.universe import parse_nifty500_constituents, parse_nse_equity_master


def test_parse_nse_equity_master_keeps_eq_series():
    raw = b"SYMBOL,NAME OF COMPANY,SERIES,ISIN NUMBER\nINFY,Infosys Limited,EQ,INE009A01021\nABC,ABC Limited,BE,INE000A01000\n"
    frame = parse_nse_equity_master(raw)
    assert frame["symbol"].tolist() == ["INFY"]
    assert frame.iloc[0]["isin"] == "INE009A01021"


def test_nifty500_sanity_check_rejects_partial_snapshot():
    raw = b"Company Name,Industry,Symbol,Series,ISIN Code\nInfosys,IT,INFY,EQ,INE009A01021\n"
    with pytest.raises(ValueError, match="sanity check"):
        parse_nifty500_constituents(raw)
