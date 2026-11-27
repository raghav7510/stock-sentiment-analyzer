"""One-shot importer for official NSE/BSE CSV exports and provider earnings data.

Usage:
  python -m scripts.ingest_indian_equities --refresh-nse
  python -m scripts.ingest_indian_equities --bse-master path/to/BSE_EQ_SCRIP_DDMMYYYY.csv
  python -m scripts.ingest_indian_equities --nse-announcements path/to/nse_export.csv
  python -m scripts.ingest_indian_equities --bse-announcements path/to/bse_export.csv
  python -m scripts.ingest_indian_equities --earnings-csv path/to/earnings_consensus.csv

NIFTY 500 membership is stored as a dated snapshot. Earnings estimates must
come from a licensed/identified consensus source; the importer does not invent them.
"""
import argparse
import hashlib
from pathlib import Path
from typing import Callable

from bharat_stock_intelligence.announcements import parse_announcement_csv
from bharat_stock_intelligence.bse_master import parse_bse_scrip_master
from bharat_stock_intelligence.database import (
    persist_announcements, persist_earnings_surprises, persist_universe, record_ingestion_run,
)
from bharat_stock_intelligence.earnings_import import parse_earnings_csv
from bharat_stock_intelligence.universe import fetch_official_nse_sources


def _audited_import(source: str, dataset: str, checksum: str | None,
                    importer: Callable[[], tuple[int, int]]) -> dict:
    """Run one importer and persist a success/failure audit record."""
    try:
        rows_seen, rows_written = importer()
        record_ingestion_run(
            source=source, dataset=dataset, status="success",
            rows_seen=rows_seen, rows_written=rows_written, checksum=checksum,
        )
        return {"source": source, "dataset": dataset, "rows_seen": rows_seen, "rows_written": rows_written}
    except Exception as exc:
        try:
            record_ingestion_run(
                source=source, dataset=dataset, status="failed",
                checksum=checksum, error_message=f"{type(exc).__name__}: {exc}"[:2000],
            )
        except Exception:
            pass
        raise


def _file_checksum(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-nse", action="store_true")
    parser.add_argument("--bse-master")
    parser.add_argument("--nse-announcements")
    parser.add_argument("--bse-announcements")
    parser.add_argument("--earnings-csv")
    args = parser.parse_args()

    if args.refresh_nse:
        def import_nse_universe():
            instruments, constituents = fetch_official_nse_sources()
            result = persist_universe(instruments.to_dict("records"), constituents.to_dict("records"))
            return len(instruments) + len(constituents), result["instruments_upserted"] + result["nifty500_membership_rows"]
        print(_audited_import("NSE_official_download", "instrument_master_and_nifty500", None, import_nse_universe))

    if args.bse_master:
        checksum = _file_checksum(args.bse_master)
        def import_bse_master():
            bse = parse_bse_scrip_master(Path(args.bse_master).read_bytes())
            result = persist_universe(bse.to_dict("records"), [])
            return len(bse), result["instruments_upserted"]
        print(_audited_import("BSE_official_csv", "instrument_master", checksum, import_bse_master))

    for exchange, path in (("NSE", args.nse_announcements), ("BSE", args.bse_announcements)):
        if path:
            checksum = _file_checksum(path)
            def import_announcements(exchange=exchange, path=path):
                records = parse_announcement_csv(Path(path).read_bytes(), exchange)
                return len(records), persist_announcements(records)
            print(_audited_import(f"{exchange}_official_csv", "corporate_announcements", checksum, import_announcements))

    if args.earnings_csv:
        checksum = _file_checksum(args.earnings_csv)
        def import_earnings():
            rows = parse_earnings_csv(Path(args.earnings_csv).read_bytes())
            return len(rows), persist_earnings_surprises(rows)
        print(_audited_import("identified_consensus_provider", "earnings_surprises", checksum, import_earnings))


if __name__ == "__main__":
    main()
