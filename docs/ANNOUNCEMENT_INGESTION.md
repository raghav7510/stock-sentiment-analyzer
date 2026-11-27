# Announcement ingestion operations

## Important source boundary

The watcher does **not** scrape NSE or BSE websites. NSE's published terms prohibit systematic or automated collection without express written consent. Use only exchange exports that you are permitted to download, or connect an appropriately licensed feed after confirming its terms.

## Local inbox workflow

1. Download an official NSE or BSE announcement CSV using the permitted exchange interface or your licensed provider.
2. Place NSE files in `data/inbox/NSE/` and BSE files in `data/inbox/BSE/`.
3. Start the watcher:

   ```powershell
   python -m scripts.watch_announcement_inbox
   ```

4. The watcher scans every 60 seconds. It waits until a file has stopped changing for at least 10 seconds, parses it, persists announcements with the existing deterministic source-hash deduplication, records an ingestion audit row, then moves the file to `data/inbox/processed/`.
5. Files that fail validation are recorded as failed and moved to `data/inbox/failed/`. If the database is unavailable and the failure cannot be audited, the file stays in the inbox so it can be retried.

For a one-time scan:

```powershell
python -m scripts.watch_announcement_inbox --once
```

## Monitoring

With the API running, inspect recent ingestion attempts:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/v1/ingestion/status?limit=20"
```

This automates local file processing, not exchange data retrieval. Automated exchange delivery requires an authorized API/feed or written permission. Do not interpret this watcher as a real-time NSE/BSE feed.
