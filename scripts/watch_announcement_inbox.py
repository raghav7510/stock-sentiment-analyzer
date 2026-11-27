"""Watch a local inbox and ingest operator-downloaded NSE/BSE announcement CSVs.

This does not scrape or poll exchange websites. Place permitted official CSV exports
in data/inbox/NSE or data/inbox/BSE; successfully handled files are archived.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from bharat_stock_intelligence.announcements import parse_announcement_csv
from bharat_stock_intelligence.database import persist_announcements, record_ingestion_run

logger = logging.getLogger("bsi.announcement_watcher")


def iter_ready_files(inbox: Path, settle_seconds: int = 10, now: float | None = None):
    """Yield CSV files whose modification time is older than the settle window."""
    current = time.time() if now is None else now
    for exchange in ("NSE", "BSE"):
        folder = inbox / exchange
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.csv")):
            try:
                if current - path.stat().st_mtime >= settle_seconds:
                    yield exchange, path
            except FileNotFoundError:
                # Another process may have moved the file between glob and stat.
                continue


def _archive(path: Path, inbox: Path, bucket: str) -> Path:
    destination_dir = inbox / bucket
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / path.name
    if destination.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = destination_dir / f"{path.stem}_{stamp}{path.suffix}"
    return Path(shutil.move(str(path), str(destination)))


def process_file(exchange: str, path: Path, inbox: Path) -> bool:
    """Import and archive one file. Return False if it should be retried."""
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        records = parse_announcement_csv(path.read_bytes(), exchange)
        written = persist_announcements(records)
    except Exception as exc:
        # Do not discard the file if the database is unavailable and the failed
        # attempt cannot be audited. It remains in the inbox for a later retry.
        try:
            record_ingestion_run(
                source=f"{exchange}_official_csv", dataset="corporate_announcements",
                status="failed", checksum=checksum,
                error_message=f"{type(exc).__name__}: {exc}"[:2000],
            )
        except Exception:
            logger.exception("Could not persist failed ingestion audit; leaving %s in inbox", path)
            return False
        archived = _archive(path, inbox, "failed")
        logger.error("Announcement import failed for %s; moved to %s: %s", path, archived, exc)
        return True

    record_ingestion_run(
        source=f"{exchange}_official_csv", dataset="corporate_announcements",
        status="success", rows_seen=len(records), rows_written=written,
        checksum=checksum,
    )
    archived = _archive(path, inbox, "processed")
    logger.info(
        "Imported %s: exchange=%s rows_seen=%d rows_written=%d archived=%s",
        path.name, exchange, len(records), written, archived,
    )
    return True


def run(inbox: Path, poll_seconds: int, settle_seconds: int, once: bool = False):
    inbox.mkdir(parents=True, exist_ok=True)
    for exchange in ("NSE", "BSE"):
        (inbox / exchange).mkdir(parents=True, exist_ok=True)

    logger.info("Watching %s (poll=%ss, settle=%ss)", inbox.resolve(), poll_seconds, settle_seconds)
    while True:
        for exchange, path in iter_ready_files(inbox, settle_seconds):
            process_file(exchange, path, inbox)
        if once:
            return
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description="Automatically import CSVs placed in a local NSE/BSE inbox.")
    parser.add_argument("--inbox", default="data/inbox")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--settle-seconds", type=int, default=10)
    parser.add_argument("--once", action="store_true", help="scan once and exit")
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.settle_seconds < 0:
        parser.error("poll-seconds must be >= 1 and settle-seconds must be >= 0")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    run(Path(args.inbox), args.poll_seconds, args.settle_seconds, args.once)


if __name__ == "__main__":
    main()
