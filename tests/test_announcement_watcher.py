import os
import time
from pathlib import Path

from scripts.watch_announcement_inbox import iter_ready_files


def test_iter_ready_files_only_yields_stable_exchange_csvs(tmp_path):
    nse = tmp_path / "NSE"
    bse = tmp_path / "BSE"
    nse.mkdir()
    bse.mkdir()
    ready = nse / "ready.csv"
    recent = bse / "still_downloading.csv"
    ignored = nse / "notes.txt"
    ready.write_text("SYMBOL,SUBJECT\nINFY,Dividend\n", encoding="utf-8")
    recent.write_text("SYMBOL,SUBJECT\nTCS,Results\n", encoding="utf-8")
    ignored.write_text("not a csv", encoding="utf-8")
    now = time.time()
    os.utime(ready, (now - 30, now - 30))
    os.utime(recent, (now, now))

    found = list(iter_ready_files(tmp_path, settle_seconds=10, now=now))
    assert found == [("NSE", ready)]


def test_iter_ready_files_handles_missing_exchange_folders(tmp_path):
    (tmp_path / "NSE").mkdir()
    assert list(iter_ready_files(tmp_path, settle_seconds=0)) == []


def test_scan_inbox_reports_import_duplicate_failure_and_retry(tmp_path, monkeypatch):
    from scripts import watch_announcement_inbox as watcher

    nse = tmp_path / "NSE"
    nse.mkdir()
    names = ["a.csv", "b.csv", "c.csv", "d.csv"]
    now = time.time()
    for name in names:
        file = nse / name
        file.write_text("x", encoding="utf-8")
        os.utime(file, (now - 30, now - 30))

    outcomes = iter(["imported", "duplicate_only", "failed", "retry"])
    monkeypatch.setattr(watcher, "process_file", lambda exchange, path, inbox: next(outcomes))
    counts = watcher.scan_inbox(tmp_path, settle_seconds=10)

    assert counts == {
        "files_ready": 4, "imported": 1, "duplicate_only": 1,
        "failed": 1, "retry": 1,
    }
