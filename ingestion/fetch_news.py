"""Fetch recent news for each portfolio company via NewsAPI.

Free-tier constraints respected:
  - 100 requests/day total
  - 1 request/second max (enforced via inter-request sleep)
  - Results limited to last 30 days (free tier cap, we use 7)
  - Max 100 articles per endpoint call (we page up to PAGE_LIMIT pages)

Deduplication: url column is UNIQUE in news_items — INSERT OR IGNORE
silently drops any article already stored.

Usage:
    python ingestion/fetch_news.py               # all companies, 7-day window
    python ingestion/fetch_news.py --days 3      # shorter window
    python ingestion/fetch_news.py --company snowflake  # single company by id
    python ingestion/fetch_news.py --dry-run     # print what would be fetched, no DB writes
"""
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.init_db import init_db

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOOKBACK_DAYS: int = int(os.getenv("LOOKBACK_DAYS", "7"))
# NEWSAPI_KEY is read lazily in run() so Streamlit Cloud secret injection
# (which happens inside init_db) has a chance to populate os.environ first.

# Free tier: 100 req/day. With 12 companies × up to 2 pages each = 24 req.
# We stay well under the cap; page limit is a safety rail.
PAGE_LIMIT: int = 2
PAGE_SIZE: int = 100  # max per NewsAPI call

# Minimum seconds between any two NewsAPI requests (free tier: ~1 req/sec)
REQUEST_INTERVAL: float = 1.2

NEWSAPI_BASE = "https://newsapi.org/v2/everything"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Ensures at least `interval` seconds between successive calls."""

    def __init__(self, interval: float = REQUEST_INTERVAL) -> None:
        self._interval = interval
        self._last_call: float = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        gap = self._interval - elapsed
        if gap > 0:
            time.sleep(gap)
        self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# NewsAPI thin wrapper (uses `requests` directly — no extra SDK dependency)
# ---------------------------------------------------------------------------

def _newsapi_request(params: dict, limiter: RateLimiter) -> dict:
    """Make a single authenticated GET to /v2/everything with retry on 429."""
    import requests  # local import — already in requirements via newsapi-python

    params = {"apiKey": os.environ.get("NEWSAPI_KEY", ""), "language": "en", **params}
    max_retries = 3

    for attempt in range(max_retries):
        limiter.wait()
        try:
            resp = requests.get(NEWSAPI_BASE, params=params, timeout=15)
        except requests.RequestException as exc:
            log.warning("Network error (attempt %d/%d): %s", attempt + 1, max_retries, exc)
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
            continue

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 60))
            log.warning("Rate limited — waiting %ds before retry", retry_after)
            time.sleep(retry_after)
            continue

        if resp.status_code == 401:
            raise RuntimeError("NewsAPI returned 401 — check your NEWSAPI_KEY in .env")

        if resp.status_code != 200:
            log.error("NewsAPI HTTP %d: %s", resp.status_code, resp.text[:200])
            resp.raise_for_status()

        data = resp.json()
        if data.get("status") != "ok":
            raise RuntimeError(f"NewsAPI error: {data.get('message', data)}")

        return data

    raise RuntimeError("Exceeded maximum retries for NewsAPI request")


# ---------------------------------------------------------------------------
# Article iteration with pagination
# ---------------------------------------------------------------------------

def iter_articles(
    query: str,
    from_date: str,
    limiter: RateLimiter,
) -> Iterator[dict]:
    """Yield articles across pages, stopping at PAGE_LIMIT or no more results."""
    for page in range(1, PAGE_LIMIT + 1):
        data = _newsapi_request(
            {
                "q": query,
                "from": from_date,
                "sortBy": "relevancy",
                "pageSize": PAGE_SIZE,
                "page": page,
            },
            limiter,
        )
        articles = data.get("articles", [])
        total = data.get("totalResults", 0)

        if not articles:
            break

        yield from articles

        # No point fetching further pages when we have all results
        if page * PAGE_SIZE >= total:
            break


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def build_query(company: dict) -> str:
    """
    Build a NewsAPI query string for a company.

    Strategy: use the 2-3 most specific keywords from the keywords field.
    Falls back to the company name. OR-joins with quotes for phrase matching.
    NewsAPI free tier supports up to 500 chars in q.
    """
    raw = company["keywords"]
    keywords: list[str] = json.loads(raw) if raw else []

    if not keywords:
        keywords = [company["name"]]

    # Use at most 3 terms; longer lists risk diluting relevancy
    terms = keywords[:3]
    query = " OR ".join(f'"{t}"' for t in terms)

    # If name isn't already in keywords, prepend it as the primary signal
    name_lower = company["name"].lower()
    if not any(name_lower in t.lower() for t in terms):
        query = f'"{company["name"]}" OR {query}'

    return query[:500]


# ---------------------------------------------------------------------------
# Per-company result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    company_id: str
    company_name: str
    fetched: int = 0
    skipped_duplicate: int = 0
    skipped_no_url: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_seen(self) -> int:
        return self.fetched + self.skipped_duplicate + self.skipped_no_url

    def __str__(self) -> str:
        parts = [f"{self.fetched} new"]
        if self.skipped_duplicate:
            parts.append(f"{self.skipped_duplicate} duplicate")
        if self.skipped_no_url:
            parts.append(f"{self.skipped_no_url} no-url")
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        return f"{self.company_name}: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Core fetch-and-store logic
# ---------------------------------------------------------------------------

def fetch_company(
    company: dict,
    conn,
    limiter: RateLimiter,
    from_date: str,
    dry_run: bool = False,
    max_articles: int = 0,        # 0 = no cap
) -> FetchResult:
    result = FetchResult(company_id=company["id"], company_name=company["name"])
    query = build_query(company)
    log.debug("  Query: %s", query)

    try:
        articles = list(iter_articles(query, from_date, limiter))
    except Exception as exc:
        result.errors.append(f"API fetch failed: {exc}")
        log.error("  [%s] fetch failed: %s", company["id"], exc)
        return result

    if max_articles > 0:
        articles = articles[:max_articles]

    for art in articles:
        url = (art.get("url") or "").strip()
        if not url or url == "https://removed.com":
            result.skipped_no_url += 1
            continue

        title = (art.get("title") or "").strip()
        if not title or title == "[Removed]":
            result.skipped_no_url += 1
            continue

        # Combine description + truncated content as raw_text
        description = art.get("description") or ""
        content_raw = art.get("content") or ""
        # NewsAPI free tier truncates content at ~200 chars with "[+N chars]"
        raw_text = "\n\n".join(filter(None, [description, content_raw]))

        source = (art.get("source") or {}).get("name") or ""
        published_at = art.get("publishedAt") or ""

        if dry_run:
            log.info("  [DRY-RUN] Would insert: %s", title[:80])
            result.fetched += 1
            continue

        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO news_items
                    (company_id, title, source, url, published_at, content, fetched_at)
                VALUES
                    (?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (company["id"], title[:500], source[:200], url, published_at, raw_text),
            )
            rows_changed = conn.execute("SELECT changes()").fetchone()[0]
            if rows_changed:
                result.fetched += 1
            else:
                result.skipped_duplicate += 1
        except Exception as exc:
            result.errors.append(str(exc))
            log.warning("  DB insert failed for '%s': %s", title[:60], exc)

    if not dry_run:
        conn.commit()

    return result


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run(
    days: int = LOOKBACK_DAYS,
    company_filter: str | None = None,
    dry_run: bool = False,
    max_articles_per_company: int = 0,   # 0 = no cap; set e.g. 10 for fast dashboard runs
) -> list[FetchResult]:
    newsapi_key = os.environ.get("NEWSAPI_KEY", "")
    if not newsapi_key:
        raise RuntimeError(
            "NEWSAPI_KEY is not set. "
            "Add it to .env for local dev, or to Streamlit Cloud secrets."
        )

    conn = init_db()
    limiter = RateLimiter()

    from_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    log.info("Fetching news from %s to today (window: %d days)", from_date, days)
    if dry_run:
        log.info("DRY-RUN mode — no database writes")

    if company_filter:
        companies = conn.execute(
            "SELECT * FROM companies WHERE id = ?", (company_filter,)
        ).fetchall()
        if not companies:
            raise ValueError(f"No company found with id '{company_filter}'")
    else:
        companies = conn.execute("SELECT * FROM companies ORDER BY name").fetchall()

    log.info("Processing %d company/companies...\n", len(companies))

    results: list[FetchResult] = []
    for company in companies:
        result = fetch_company(
            company, conn, limiter, from_date,
            dry_run=dry_run,
            max_articles=max_articles_per_company,
        )
        results.append(result)
        log.info("  %s", result)

    conn.close()

    total_new = sum(r.fetched for r in results)
    total_dupes = sum(r.skipped_duplicate for r in results)
    total_errors = sum(len(r.errors) for r in results)
    log.info(
        "\nDone — %d new articles stored, %d duplicates skipped, %d errors",
        total_new, total_dupes, total_errors,
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch portfolio company news via NewsAPI")
    parser.add_argument("--days", type=int, default=LOOKBACK_DAYS, help="Lookback window in days (default: 7)")
    parser.add_argument("--company", type=str, default=None, help="Fetch a single company by id (e.g. snowflake)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be fetched without writing to DB")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    run(days=args.days, company_filter=args.company, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
