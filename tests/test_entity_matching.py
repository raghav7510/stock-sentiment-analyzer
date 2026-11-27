from bharat_stock_intelligence.entity_matching import match_companies


def test_matches_company_name_and_long_ticker():
    universe = [
        {"symbol": "INFY", "company_name": "Infosys Limited", "sector": "IT"},
        {"symbol": "IT", "company_name": "ITC Limited", "sector": "FMCG"},
    ]
    matches = match_companies("Infosys Limited reported results; INFY shares reacted.", universe)
    assert [m["symbol"] for m in matches] == ["INFY"]


def test_does_not_match_short_symbol_as_common_word():
    universe = [{"symbol": "IT", "company_name": "ITC Limited", "sector": "FMCG"}]
    assert match_companies("IT services are growing", universe) == []
