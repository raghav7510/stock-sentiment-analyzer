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
