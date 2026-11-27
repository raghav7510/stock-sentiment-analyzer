from bharat_stock_intelligence.news import _canonical_url


def test_canonical_url_removes_tracking_and_fragment():
    assert _canonical_url("HTTPS://Example.com/story/?utm_source=x#section") == "https://example.com/story"
