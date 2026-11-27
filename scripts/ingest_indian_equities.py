"""One-shot importer for official NSE universe files and exchange announcement CSV exports.

Usage:
  python -m scripts.ingest_indian_equities --refresh-nse
  python -m scripts.ingest_indian_equities --nse-announcements path/to/nse_export.csv
  python -m scripts.ingest_indian_equities --bse-announcements path/to/bse_export.csv

The NIFTY 500 membership file is a point-in-time snapshot, not a historical archive.
"""
import argparse
from pathlib import Path

from bharat_stock_intelligence.announcements import parse_announcement_csv
from bharat_stock_intelligence.database import persist_announcements, persist_universe
from bharat_stock_intelligence.universe import fetch_official_nse_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-nse", action="store_true")
    parser.add_argument("--nse-announcements")
    parser.add_argument("--bse-announcements")
    args = parser.parse_args()
    if args.refresh_nse:
        instruments, constituents = fetch_official_nse_sources()
        print(persist_universe(instruments.to_dict("records"), constituents.to_dict("records")))
    for exchange, path in (("NSE", args.nse_announcements), ("BSE", args.bse_announcements)):
        if path:
            records = parse_announcement_csv(Path(path).read_bytes(), exchange)
            print({"exchange": exchange, "announcements": persist_announcements(records)})


if __name__ == "__main__":
    main()
