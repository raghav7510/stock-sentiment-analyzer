import pytest

from scripts import ingest_indian_equities


def test_audited_import_records_success(monkeypatch):
    recorded = []
    monkeypatch.setattr(ingest_indian_equities, "record_ingestion_run", lambda **kwargs: recorded.append(kwargs))

    result = ingest_indian_equities._audited_import(
        "NSE_official_csv", "corporate_announcements", "abc123", lambda: (12, 8)
    )

    assert result["rows_seen"] == 12
    assert result["rows_written"] == 8
    assert recorded[0]["status"] == "success"
    assert recorded[0]["checksum"] == "abc123"


def test_audited_import_records_failure_and_reraises(monkeypatch):
    recorded = []
    monkeypatch.setattr(ingestion_indian_equities, "record_ingestion_run", lambda **kwargs: recorded.append(kwargs))

    def fail():
        raise ValueError("bad source file")

    with pytest.raises(ValueError, match="bad source file"):
        ingestion_indian_equities._audited_import("BSE_official_csv", "instrument_master", None, fail)

    assert recorded[0]["status"] == "failed"
    assert "bad source file" in recorded[0]["error_message"]
