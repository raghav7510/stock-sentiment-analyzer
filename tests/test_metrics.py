import pytest

from bharat_stock_intelligence.metrics import aggregate_sentiment


def test_empty_data_is_not_neutral():
    assert aggregate_sentiment([]) == {"score": 0.0, "article_count": 0, "coverage": "no_data"}


def test_weighted_score():
    result = aggregate_sentiment([{"score": 1, "weight": 3}, {"score": -1, "weight": 1}])
    assert result["score"] == 0.5
    assert result["article_count"] == 2


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1.01, 1.01])
def test_invalid_scores_rejected(score):
    with pytest.raises(ValueError):
        aggregate_sentiment([{"score": score}])


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf")])
def test_invalid_weights_rejected(weight):
    with pytest.raises(ValueError):
        aggregate_sentiment([{"score": 0.2, "weight": weight}])


def test_missing_score_rejected():
    with pytest.raises(ValueError):
        aggregate_sentiment([{"weight": 1}])
