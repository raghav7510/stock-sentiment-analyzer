from bharat_stock_intelligence.bse_master import parse_bse_scrip_master


def test_parse_bse_standardized_scrip_master():
    raw = b"Scrip Code,Instrument Code,Scrip Name,ISIN CODE\n500209,INFY,Infosys Limited,INE009A01021\n"
    result = parse_bse_scrip_master(raw)
    assert result.iloc[0]["symbol"] == "INFY"
    assert result.iloc[0]["exchange"] == "BSE"
    assert result.iloc[0]["isin"] == "INE009A01021"
