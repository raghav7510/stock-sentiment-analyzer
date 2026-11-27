from bharat_stock_intelligence.sector_analytics import aggregate_sector_sentiment


def test_sector_weighted_sentiment_and_coverage():
    rows = [
        {"sector": "IT", "score": 0.5, "weight": 2, "symbol": "INFY"},
        {"sector": "IT", "score": -0.5, "weight": 1, "symbol": "TCS"},
        {"sector": None, "score": 0.2, "symbol": "ABC"},
    ]
    result = aggregate_sector_sentiment(rows)
    it = next(item for item in result if item["sector"] == "IT")
    assert it["sentiment_score"] == pytest.approx(0.1667)
    assert it["article_count"] == 2
    assert it["company_count"] == 2


import pytest
