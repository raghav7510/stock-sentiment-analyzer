import pytest
from bharat_stock_intelligence.sentiment import aggregate_sentiment

def test_empty_data_is_not_neutral_signal():
    assert aggregate_sentiment([]) == {"score": 0.0, "article_count": 0, "coverage": "no_data"}

def test_weighted_score():
    result = aggregate_sentiment([{"score": 1, "weight": 3}, {"score": -1, "weight": 1}])
    assert result["score"] == 0.5
    assert result["article_count"] == 2

def test_invalid_score_rejected():
    with pytest.raises(ValueError):
        aggregate_sentiment([{"score": 2}])

def test_non_positive_weight_rejected():
    with pytest.raises(ValueError):
        aggregate_sentiment([{"score": 0.2, "weight": 0}])
