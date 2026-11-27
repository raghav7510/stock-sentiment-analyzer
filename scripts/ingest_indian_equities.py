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
from pathlib import Path

from bharat_stock_intelligence.announcements import parse_announcement_csv
from bharat_stock_intelligence.bse_master import parse_bse_scrip_master
from bharat_stock_intelligence.database import (
    persist_announcements, persist_earnings_surprises, persist_universe,
)
from bharat_stock_intelligence.earnings_import import parse_earnings_csv
from bharat_stock_intelligence.universe import fetch_official_nse_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-nse", action="store_true")
    parser.add_argument("--bse-master")
    parser.add_argument("--nse-announcements")
    parser.add_argument("--bse-announcements")
    parser.add_argument("--earnings-csv")
    args = parser.parse_args()

    if args.refresh_nse:
        instruments, constituents = fetch_official_nse_sources()
        print(persist_universe(instruments.to_dict("records"), constituents.to_dict("records")))

    if args.bse_master:
        bse = parse_bse_scrip_master(Path(args.bse_master).read_bytes())
        print(persist_universe(bse.to_dict("records"), []))

    for exchange, path in (("NSE", args.nse_announcements), ("BSE", args.bse_announcements)):
        if path:
            records = parse_announcement_csv(Path(path).read_bytes(), exchange)
            print({"exchange": exchange, "announcements_written": persist_announcements(records)})

    if args.earnings_csv:
        rows = parse_earnings_csv(Path(args.earnings_csv).read_bytes())
        print({"earnings_surprises_written": persist_earnings_surprises(rows)})


if __name__ == "__main__":
    main()
