import pytest

from bharat_stock_intelligence.earnings import calculate_surprise, classify_earnings_surprise


def test_positive_eps_surprise():
    result = calculate_surprise(12, 10)
    assert result["surprise_pct"] == 20
    assert result["direction"] == "beat"


def test_negative_estimate_uses_absolute_denominator():
    assert calculate_surprise(-8, -10)["surprise_pct"] == 20


def test_zero_consensus_rejected():
    with pytest.raises(ValueError):
        calculate_surprise(1, 0)


def test_partial_metric_pair_supported():
    result = classify_earnings_surprise(12, 10)
    assert result["eps"]["direction"] == "beat"
    assert "revenue" not in result
