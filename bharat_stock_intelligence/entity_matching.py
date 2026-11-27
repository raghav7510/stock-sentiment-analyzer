"""Deterministic, auditable article-to-issuer matching."""
from __future__ import annotations

import re
import unicodedata


def _norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def match_companies(text: str, instruments: list[dict]) -> list[dict]:
    """Match known symbol/company/alias phrases; avoid fuzzy guesses.

    Short symbols are matched only as uppercase ticker tokens to reduce
    false positives (e.g. IT, ON, or ONE in ordinary prose).
    """
    raw = str(text or "")
    normalized = f" {_norm(raw)} "
    matches = []
    for instrument in instruments:
        candidates = [instrument.get("company_name", ""), *(instrument.get("aliases") or [])]
        company_hit = next((c for c in candidates if len(_norm(c)) >= 5 and f" {_norm(c)} " in normalized), None)
        symbol = str(instrument.get("symbol") or "").strip().upper()
        symbol_hit = bool(symbol and len(symbol) >= 4 and re.search(rf"(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])", raw))
        if company_hit or symbol_hit:
            matches.append({
                "symbol": symbol,
                "company_name": instrument.get("company_name"),
                "match_type": "company_name_or_alias" if company_hit else "ticker_symbol",
                "matched_text": company_hit or symbol,
                "sector": instrument.get("sector"),
            })
    return matches
