"""Fetch SEC EDGAR 8-K filings for public portfolio companies.

Uses two free EDGAR APIs — no key required:
  • EFTS  (efts.sec.gov/hits.json)   — full-text search across all filings
  • EDGAR data API (data.sec.gov)     — filing index + CIK lookup

Targets 8-K Item 5.02 filings specifically, which cover:
  "Departure of Directors or Certain Officers; Election of Directors;
   Appointment of Certain Officers; Compensatory Arrangements"

Rate limiting: SEC requests ≤ 10 req/s. We use a 0.11 s floor between calls
and back off on 429/503. User-Agent header is required (SEC policy).

Deduplication: accession_no is UNIQUE in edgar_filings.

CIK resolution:
    companies.json may include a "cik" field per company (10-digit zero-padded
    SEC Central Index Key). When present, it is used directly for the EFTS
    "ciks" filter param, which is more reliable than ticker/entity-name
    matching. The stored CIK is still cross-checked against SEC's live
    ticker→CIK map (company_tickers.json) on every run; on a mismatch the
    live-resolved CIK wins and a warning is logged, so a stale or fat-fingered
    value in companies.json can never silently produce wrong results.
    If no CIK is stored, we fall back fully to ticker-based resolution.

Usage:
    python ingestion/fetch_edgar.py                    # all public companies, 90-day window
    python ingestion/fetch_edgar.py --days 180         # wider window
    python ingestion/fetch_edgar.py --ticker SNOW      # single ticker
    python ingestion/fetch_edgar.py --dry-run          # no DB writes
    python ingestion/fetch_edgar.py --debug            # verbose logging
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
import requests
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.init_db import init_db

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SEC requires a descriptive User-Agent: "Company Name contact@email.com"
# Update this to your own contact if you deploy this in production.
USER_AGENT = "PortfolioIntelligenceTool research@example.com"

EFTS_BASE         = "https://efts.sec.gov/LATEST/search-index"
COMPANY_TICKERS   = "https://www.sec.gov/files/company_tickers.json"
EDGAR_ARCHIVES    = "https://www.sec.gov/Archives/edgar/data"

REQUEST_INTERVAL = 0.15   # 10 req/s max → 0.1 s floor; we use 0.15 for safety
MAX_RETRIES      = 4
LOOKBACK_DAYS    = 90     # widened from 30 — exec-move 8-Ks are infrequent per company,
                          # so a longer window is needed to reliably find any in a given run

# 8-K Item 5.02 keywords that indicate executive movement
EXEC_KEYWORDS = [
    "appointment", "appointed", "appoints",
    "departure", "departed", "resigns", "resigned", "resignation",
    "Chief Executive Officer", "Chief Financial Officer", "Chief Operating Officer",
    "Chief Technology Officer", "Chief Revenue Officer",
    "President", "General Counsel", "Board of Directors",
    "named", "elected", "stepped down", "transition",
]

# Regex patterns to classify the signal from extracted text
_DEPARTURE_RE = re.compile(
    r"\b(resign|depart|step(?:ped)? down|terminat|no longer|leave|left)\b",
    re.I,
)
_APPOINT_RE = re.compile(
    r"\b(appoint|elect|nam(?:ed|es)|promot|hire[sd]?|join(?:ed|s)?|assume[sd]?)\b",
    re.I,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, interval: float = REQUEST_INTERVAL) -> None:
        self._interval = interval
        self._last: float = 0.0

    def wait(self) -> None:
        gap = self._interval - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


_limiter = RateLimiter()


def _get(url: str, params: dict | None = None) -> dict:
    """GET with rate limiting, retry on 429/5xx, required User-Agent."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    backoff = 2.0

    for attempt in range(MAX_RETRIES):
        _limiter.wait()
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)
        except requests.RequestException as exc:
            log.warning("Network error (attempt %d): %s", attempt + 1, exc)
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(backoff)
            backoff *= 2
            continue

        if resp.status_code in (429, 503):
            wait = int(resp.headers.get("Retry-After", backoff))
            log.warning("SEC rate-limited (HTTP %d) — sleeping %ds", resp.status_code, wait)
            time.sleep(wait)
            backoff = max(backoff * 2, wait)
            continue

        if resp.status_code == 404:
            return {}

        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError as exc:
            log.warning("SEC returned non-JSON from %s: %s", url, exc)
            return {}

    raise RuntimeError(f"Exceeded {MAX_RETRIES} retries for {url}")


# ---------------------------------------------------------------------------
# CIK resolution  (ticker → 10-digit zero-padded CIK)
# ---------------------------------------------------------------------------

_cik_cache: dict[str, str] = {}
_ticker_map_cache: dict | None = None


def _normalize_cik(raw: str) -> str:
    """Zero-pad a CIK to 10 digits, stripping any non-digit characters."""
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    return digits.zfill(10)


def _live_lookup_cik_by_ticker(ticker: str) -> str | None:
    """Resolve a ticker → CIK using SEC's live company_tickers.json map."""
    global _ticker_map_cache
    if _ticker_map_cache is None:
        _ticker_map_cache = _get(COMPANY_TICKERS) or {}

    ticker_upper = ticker.upper()
    for entry in _ticker_map_cache.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            return _normalize_cik(entry["cik_str"])
    return None


def resolve_cik(company: dict) -> str | None:
    """Return the zero-padded CIK for a company.

    Resolution order:
      1. If companies.json/DB has a non-null `cik`, normalize it and verify
         it against SEC's live ticker→CIK map. On a match, use it directly
         (skips a network round-trip on cache hits). On a mismatch, log a
         warning and prefer the live-resolved value — a hand-entered CIK
         should never silently override what SEC's own data says.
      2. If no stored CIK (or verification couldn't run), fall back to a
         live ticker-based lookup.
    """
    ticker = company.get("ticker")
    stored_cik = company.get("cik")
    cache_key = ticker or stored_cik
    if cache_key and cache_key in _cik_cache:
        return _cik_cache[cache_key]

    if not ticker and not stored_cik:
        return None

    live_cik = _live_lookup_cik_by_ticker(ticker) if ticker else None

    if stored_cik:
        normalized = _normalize_cik(stored_cik)
        if live_cik and live_cik != normalized:
            log.warning(
                "  CIK mismatch for %s: companies.json has %s but SEC live map "
                "resolves %s to %s — using the live-resolved value.",
                ticker, normalized, ticker, live_cik,
            )
            resolved = live_cik
        else:
            resolved = normalized
    elif live_cik:
        resolved = live_cik
    else:
        log.warning("  Could not resolve CIK for ticker %s (no stored CIK, live lookup failed)", ticker)
        return None

    if cache_key:
        _cik_cache[cache_key] = resolved
    log.debug("  Resolved %s → CIK %s", ticker or stored_cik, resolved)
    return resolved


# ---------------------------------------------------------------------------
# EFTS full-text search for 8-K filings
# ---------------------------------------------------------------------------

def search_8k_filings(
    ticker: str,
    cik: str,
    start_date: str,
    end_date: str,
) -> Iterator[dict]:
    """
    Yield raw EFTS hit dicts for 8-K filings from this company in the date range.

    EFTS endpoint: https://efts.sec.gov/LATEST/search-index
    Key parameters:
      q         — full-text query (supports phrase quotes and OR)
      forms     — comma-separated form types
      dateRange — "custom" enables startdt/enddt
      ciks      — comma-separated list of 10-digit zero-padded CIKs; scopes
                  results to specific filer(s). Far more reliable than the
                  "entity" name/ticker filter, which does fuzzy text matching
                  against company names and can miss or misattribute filings
                  for tickers that overlap with other share classes/aliases.
      from      — pagination offset
      size      — page size

    We target Item 5.02 language. EFTS returns highlighted fragments which we
    use as snippets when the full document fetch is unavailable.
    """
    exec_query = (
        '"Item 5.02" OR "appointed" OR "resigned" OR "departure" '
        'OR "Chief Executive Officer" OR "Chief Financial Officer"'
    )

    page_size = 20
    offset    = 0

    while True:
        params = {
            "q":         exec_query,
            "dateRange": "custom",
            "startdt":   start_date,
            "enddt":     end_date,
            "forms":     "8-K",
            "ciks":      cik,   # CIK-based filer filter — more reliable than entity/ticker matching
            "from":      offset,
            "size":      page_size,
        }

        data = _get(EFTS_BASE, params=params)
        hits_wrapper = data.get("hits", {})
        hits = hits_wrapper.get("hits", [])

        if not hits:
            break

        for hit in hits:
            yield hit

        total   = hits_wrapper.get("total", {}).get("value", 0)
        offset += len(hits)
        if offset >= total or offset >= 200:
            break


# ---------------------------------------------------------------------------
# Filing document retrieval  (accession → index → 8-K HTM text)
# ---------------------------------------------------------------------------

def _accession_to_index_url(cik: str, accession_no: str) -> str:
    acc_clean = accession_no.replace("-", "")
    return f"{EDGAR_ARCHIVES}/{int(cik)}/{acc_clean}/{accession_no}-index.json"


def fetch_filing_index(cik: str, accession_no: str) -> dict:
    url = _accession_to_index_url(cik, accession_no)
    return _get(url)


def _find_primary_document(index: dict) -> str | None:
    """Return the URL of the primary 8-K HTM document from a filing index."""
    docs = index.get("documents", [])
    # Prefer the file explicitly typed as the 8-K body
    for doc in docs:
        doc_type = (doc.get("type") or "").upper()
        name = (doc.get("document") or "").lower()
        if doc_type == "8-K" or (doc_type == "" and name.endswith((".htm", ".html"))):
            return doc.get("documentUrl") or doc.get("url")
    # Fall back to first .htm
    for doc in docs:
        if (doc.get("document") or "").lower().endswith((".htm", ".html")):
            return doc.get("documentUrl") or doc.get("url")
    return None


def fetch_document_text(url: str) -> str:
    """Fetch raw HTML/text of an EDGAR document and strip tags."""
    _limiter.wait()
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("  Could not fetch document %s: %s", url, exc)
        return ""

    text = resp.text
    # Strip HTML tags (good enough for snippet extraction)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Snippet + signal extraction from raw filing text
# ---------------------------------------------------------------------------

_ITEM_502_RE = re.compile(
    r"Item\s+5\.02[.\s\-–]*([^I]{50,2000})",
    re.I | re.S,
)

# Sentence splitter (rough)
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def extract_snippet(text: str, max_chars: int = 600) -> str:
    """Pull the Item 5.02 section, or fall back to any exec-keyword sentence."""
    # Try Item 5.02 section first
    m = _ITEM_502_RE.search(text)
    if m:
        chunk = m.group(1).strip()
        return chunk[:max_chars]

    # Fall back: find first sentence containing an exec keyword
    sentences = _SENT_RE.split(text)
    for sent in sentences:
        if any(kw.lower() in sent.lower() for kw in EXEC_KEYWORDS):
            return sent[:max_chars]

    return ""


def classify_signal(snippet: str) -> str:
    """Return 'appointment', 'departure', 'both', or 'other'."""
    has_departure  = bool(_DEPARTURE_RE.search(snippet))
    has_appointment = bool(_APPOINT_RE.search(snippet))
    if has_departure and has_appointment:
        return "both"
    if has_departure:
        return "departure"
    if has_appointment:
        return "appointment"
    return "other"


# ---------------------------------------------------------------------------
# Per-company result tracking
# ---------------------------------------------------------------------------

@dataclass
class EdgarResult:
    company_id: str
    ticker: str
    stored: int = 0
    skipped_duplicate: int = 0
    skipped_no_exec: int = 0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        parts = [f"{self.stored} new filings"]
        if self.skipped_duplicate:
            parts.append(f"{self.skipped_duplicate} duplicate")
        if self.skipped_no_exec:
            parts.append(f"{self.skipped_no_exec} no exec signal")
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        return f"{self.ticker}: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Per-company pipeline
# ---------------------------------------------------------------------------

def process_company(
    company: dict,
    conn,
    start_date: str,
    end_date: str,
    dry_run: bool = False,
) -> EdgarResult:
    ticker = company["ticker"]
    result = EdgarResult(company_id=company["id"], ticker=ticker)

    cik = resolve_cik(dict(company))
    if not cik:
        result.errors.append("CIK not found")
        return result

    log.debug("  [%s] CIK=%s, searching %s → %s", ticker, cik, start_date, end_date)

    try:
        hits = list(search_8k_filings(ticker, cik, start_date, end_date))
    except Exception as exc:
        result.errors.append(f"EFTS search failed: {exc}")
        log.error("  [%s] EFTS error: %s", ticker, exc)
        return result

    log.debug("  [%s] %d EFTS hits", ticker, len(hits))

    for hit in hits:
        src = hit.get("_source", {})
        accession_no = (
            src.get("accession_no")
            or hit.get("_id", "")
        )

        # Normalise accession number format (XXXXXXXXXX-YY-ZZZZZZ)
        if accession_no and "-" not in accession_no and len(accession_no) == 18:
            accession_no = f"{accession_no[:10]}-{accession_no[10:12]}-{accession_no[12:]}"

        if not accession_no:
            result.skipped_no_exec += 1
            continue

        # Dedup check before expensive HTTP calls
        exists = conn.execute(
            "SELECT 1 FROM edgar_filings WHERE accession_no = ?", (accession_no,)
        ).fetchone()
        if exists:
            result.skipped_duplicate += 1
            continue

        filing_date      = src.get("file_date") or src.get("period_of_report") or ""
        period_of_report = src.get("period_of_report") or ""

        # Build filing index URL and fetch document
        filing_url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={cik}&type=8-K&dateb=&owner=include&count=40"
        )
        index_url = _accession_to_index_url(cik, accession_no)

        doc_url  = None
        snippet  = ""
        item_no  = "5.02"

        try:
            index = _get(index_url)
            if index:
                doc_url = _find_primary_document(index)
                if doc_url:
                    text    = fetch_document_text(doc_url)
                    snippet = extract_snippet(text)
        except Exception as exc:
            log.warning("  [%s] Could not fetch document for %s: %s", ticker, accession_no, exc)

        # Fall back to EFTS highlight fragments if document fetch yielded nothing
        if not snippet:
            for fragments in hit.get("highlight", {}).values():
                snippet = " … ".join(str(f) for f in fragments)[:600]
                break

        exec_signal = classify_signal(snippet) if snippet else "other"

        if not snippet:
            result.skipped_no_exec += 1
            continue

        if dry_run:
            log.info(
                "  [DRY-RUN] %s | %s | %s | signal=%s | %.80s",
                ticker, accession_no, filing_date, exec_signal, snippet,
            )
            result.stored += 1
            continue

        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO edgar_filings
                    (company_id, ticker, cik, accession_no, form_type,
                     filing_date, period_of_report, filing_url, document_url,
                     item_no, snippet, exec_signal)
                VALUES
                    (?, ?, ?, ?, '8-K', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    company["id"], ticker, cik, accession_no,
                    filing_date, period_of_report,
                    filing_url, doc_url,
                    item_no, snippet[:1000], exec_signal,
                ),
            )
            if conn.execute("SELECT changes()").fetchone()[0]:
                result.stored += 1
            else:
                result.skipped_duplicate += 1
        except Exception as exc:
            result.errors.append(str(exc))
            log.warning("  [%s] DB insert failed for %s: %s", ticker, accession_no, exc)

    if not dry_run:
        conn.commit()

    return result


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run(
    days: int = LOOKBACK_DAYS,
    ticker_filter: str | None = None,
    dry_run: bool = False,
) -> list[EdgarResult]:
    conn = init_db()

    end_date   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    log.info("EDGAR 8-K search: %s → %s (%d-day window)", start_date, end_date, days)
    if dry_run:
        log.info("DRY-RUN — no database writes")

    # Only process companies with tickers (i.e. public)
    if ticker_filter:
        companies = conn.execute(
            "SELECT * FROM companies WHERE ticker = ?", (ticker_filter.upper(),)
        ).fetchall()
        if not companies:
            raise ValueError(f"No public company found with ticker '{ticker_filter}'")
    else:
        companies = conn.execute(
            "SELECT * FROM companies WHERE ticker IS NOT NULL AND ticker != '' ORDER BY name"
        ).fetchall()

    log.info("Processing %d public company/companies...\n", len(companies))

    results: list[EdgarResult] = []
    for company in companies:
        log.info("  → %s (%s)", company["name"], company["ticker"])
        result = process_company(company, conn, start_date, end_date, dry_run=dry_run)
        results.append(result)
        log.info("     %s", result)

    conn.close()

    total_stored = sum(r.stored for r in results)
    total_dupes  = sum(r.skipped_duplicate for r in results)
    total_errors = sum(len(r.errors) for r in results)
    log.info(
        "\nDone — %d filings stored, %d duplicates skipped, %d errors",
        total_stored, total_dupes, total_errors,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch EDGAR 8-K executive-move filings for public portfolio companies"
    )
    parser.add_argument(
        "--days", type=int, default=LOOKBACK_DAYS,
        help=f"Lookback window in days (default: {LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--ticker", type=str, default=None,
        help="Process a single ticker (e.g. SNOW)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be stored without writing to DB",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    run(days=args.days, ticker_filter=args.ticker, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
