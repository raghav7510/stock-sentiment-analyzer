import pandas as pd
import pytest

from bharat_stock_intelligence.event_study import calculate_event_study


def test_market_adjusted_car_on_event_window():
    idx = pd.bdate_range("2026-01-01", periods=5)
    asset = pd.Series([0.01, 0.03, 0.00, -0.01, 0.02], index=idx)
    benchmark = pd.Series([0.01, 0.01, 0.00, 0.00, 0.01], index=idx)
    result = calculate_event_study(asset, benchmark, [idx[2]], 1, 2)
    assert result["event_day"].tolist() == [-1, 0, 1, 2]
    assert result.iloc[1]["abnormal_return"] == pytest.approx(0.0)
    assert result.iloc[-1]["car"] == pytest.approx(0.02)


def test_missing_event_date_is_skipped():
    idx = pd.bdate_range("2026-01-01", periods=3)
    returns = pd.Series([0.01, 0.02, 0.03], index=idx)
    assert calculate_event_study(returns, returns, ["2026-02-01"]).empty


def test_negative_window_rejected():
    with pytest.raises(ValueError):
        calculate_event_study(pd.Series(dtype=float), pd.Series(dtype=float), [], -1, 2)
