"""Extract structured signals from unprocessed news items using Claude.

For each unprocessed news_item this script:
  1. Sends title + content to claude-sonnet-4-6 with a strict JSON system prompt
  2. Parses the response into the signal schema
  3. Stores each signal in extracted_signals
  4. Marks the news_item as processed via processed_at timestamp

Signal schema (per article, may contain 0–N signals):
  signal_type      : leadership_change | funding_partnership | notable_hire |
                     product_launch | other
  summary          : 1-2 sentence description of the signal
  people_mentioned : list of {name, role, company} objects
  relevance_score  : 1-5 integer (5 = highest significance for ecosystem tracking)

Usage:
    python processing/extract_signals.py               # process all pending, batch of 50
    python processing/extract_signals.py --limit 10    # smaller batch
    python processing/extract_signals.py --company snowflake
    python processing/extract_signals.py --dry-run     # call Claude, print results, no DB writes
    python processing/extract_signals.py --debug       # verbose logging
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.init_db import init_db

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL          = "claude-sonnet-4-6"
MAX_TOKENS     = 1024
BATCH_SIZE     = 50   # articles per run (keeps costs predictable)
CONTENT_LIMIT  = 2000 # chars of article content sent to Claude (free-tier NewsAPI truncates anyway)

VALID_SIGNAL_TYPES = frozenset({
    "leadership_change",
    "funding_partnership",
    "notable_hire",
    "product_launch",
    "other",
})

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a signal-extraction engine for a venture capital portfolio monitoring system.

Your ONLY job is to read a news article about a portfolio company and return a single \
valid JSON object. You must return ONLY the JSON — no preamble, no explanation, \
no markdown code fences, no trailing text.

The JSON must conform exactly to this schema:
{
  "signals": [
    {
      "signal_type": "<one of: leadership_change, funding_partnership, notable_hire, product_launch, other>",
      "summary": "<1-2 sentences describing what happened and why it matters>",
      "people_mentioned": [
        {"name": "<full name>", "role": "<title or role>", "company": "<employer or affiliation>"}
      ],
      "relevance_score": <integer 1-5>
    }
  ]
}

Relevance score guide:
  5 — Major event: CEO/CFO change, large funding round, acquisition, IPO
  4 — Significant: VP-level hire, strategic partnership, major product launch
  3 — Noteworthy: director-level hire, product update, market expansion
  2 — Minor: team changes, small partnerships, incremental updates
  1 — Noise: PR fluff, awards, minor mentions

Rules:
- Return an empty signals array [] if the article contains no meaningful signal.
- A single article may produce multiple signals (e.g. a funding round that also names a new board member).
- people_mentioned may be an empty list if no named individuals appear.
- signal_type must be exactly one of the five allowed values — choose the closest fit.
- Do NOT include commentary outside the JSON object.\
"""

USER_TEMPLATE = """\
Company: {company_name}
Sector: {sector}
Stage: {stage}

Article title: {title}
Article content:
{content}\
"""

# ---------------------------------------------------------------------------
# JSON parsing with a repair fallback
# ---------------------------------------------------------------------------

# Strip markdown fences if Claude adds them despite instructions
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _parse_response(raw: str) -> dict[str, Any] | None:
    """Parse Claude's response to a dict, tolerating minor formatting violations."""
    text = raw.strip()

    # Strip code fences if present
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()

    # Find the outermost {...} if there's leading/trailing noise
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        log.warning("JSON parse failed: %s | raw: %.120s", exc, raw)
        return None


def _validate_signal(sig: dict) -> dict:
    """Coerce a raw signal dict into the expected shape, filling defaults."""
    signal_type = sig.get("signal_type", "other")
    if signal_type not in VALID_SIGNAL_TYPES:
        signal_type = "other"

    score = sig.get("relevance_score", 3)
    try:
        score = max(1, min(5, int(score)))
    except (TypeError, ValueError):
        score = 3

    people = sig.get("people_mentioned", [])
    if not isinstance(people, list):
        people = []

    return {
        "signal_type":      signal_type,
        "summary":          str(sig.get("summary", "")).strip()[:500],
        "people_mentioned": people,
        "relevance_score":  score,
    }


# ---------------------------------------------------------------------------
# Claude call
# ---------------------------------------------------------------------------

_RETRY_DELAYS = [5, 30, 90]   # seconds between retries on transient failures


def call_claude(client: anthropic.Anthropic, article: dict, company: dict) -> dict[str, Any] | None:
    """Call Claude and return the parsed JSON dict, or None on unrecoverable failure.

    Retry policy:
      - RateLimitError  → wait and retry (up to 3×, with increasing back-off)
      - APITimeoutError → retry immediately up to 3×
      - APIConnectionError → retry up to 3×
      - APIStatusError 5xx → retry up to 3×
      - APIStatusError 4xx (not 429) → fatal, do not retry
    """
    content_text = (article["content"] or "").strip()[:CONTENT_LIMIT]

    if not article["title"] and not content_text:
        return None

    user_message = USER_TEMPLATE.format(
        company_name=company["name"],
        sector=company["sector"] or "Technology",
        stage=company["stage"] or "Unknown",
        title=article["title"] or "(no title)",
        content=content_text or "(no content available)",
    )

    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            log.info("  Retrying Claude call in %ds (attempt %d)...", delay, attempt + 1)
            time.sleep(delay)
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = next((b.text for b in response.content if b.type == "text"), "").strip()
            log.debug("Raw response: %.200s", raw)
            return _parse_response(raw)

        except anthropic.RateLimitError as exc:
            log.warning("Claude rate limit hit: %s", exc)
            if attempt == len(_RETRY_DELAYS):
                raise
        except anthropic.APITimeoutError as exc:
            log.warning("Claude timeout: %s", exc)
            if attempt == len(_RETRY_DELAYS):
                raise
        except anthropic.APIConnectionError as exc:
            log.warning("Claude connection error: %s", exc)
            if attempt == len(_RETRY_DELAYS):
                raise
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                log.warning("Claude server error (HTTP %d): %s", exc.status_code, exc.message)
                if attempt == len(_RETRY_DELAYS):
                    raise
            else:
                # 4xx errors (bad request, auth) — don't retry
                log.error("Claude API error (HTTP %d): %s", exc.status_code, exc.message)
                raise

    return None   # unreachable but satisfies type checker


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------

def store_signals(conn, article: dict, company: dict, signals: list[dict]) -> int:
    """Insert validated signals, return count inserted."""
    inserted = 0
    for sig in signals:
        s = _validate_signal(sig)
        conn.execute(
            """
            INSERT INTO extracted_signals
                (news_item_id, company_id, signal_type, summary, people_mentioned, relevance_score)
            VALUES
                (?, ?, ?, ?, ?, ?)
            """,
            (
                article["id"],
                company["id"],
                s["signal_type"],
                s["summary"],
                json.dumps(s["people_mentioned"]),
                s["relevance_score"],
            ),
        )
        inserted += 1
    return inserted


def mark_processed(conn, article_id: int) -> None:
    conn.execute(
        "UPDATE news_items SET processed_at = datetime('now') WHERE id = ?",
        (article_id,),
    )


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    processed: int = 0
    signals_stored: int = 0
    skipped_empty: int = 0
    skipped_parse_error: int = 0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{self.processed} articles processed → "
            f"{self.signals_stored} signals stored "
            f"({self.skipped_empty} no-signal, "
            f"{self.skipped_parse_error} parse errors, "
            f"{len(self.errors)} API errors)"
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    limit: int = BATCH_SIZE,
    company_filter: str | None = None,
    dry_run: bool = False,
) -> ExtractionResult:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set — add it to your .env file")

    client = anthropic.Anthropic(api_key=api_key)
    conn   = init_db()
    result = ExtractionResult()

    # Fetch unprocessed articles — those without a processed_at timestamp
    company_clause = "AND ni.company_id = ?" if company_filter else ""
    params: list[Any] = []
    if company_filter:
        params.append(company_filter)
    params.append(limit)

    articles = conn.execute(
        f"""
        SELECT
            ni.id, ni.company_id, ni.title, ni.content,
            ni.source, ni.url, ni.published_at,
            c.name  AS company_name,
            c.sector, c.stage
        FROM news_items ni
        JOIN companies c ON c.id = ni.company_id
        WHERE ni.processed_at IS NULL
        {company_clause}
        ORDER BY ni.published_at DESC
        LIMIT ?
        """,
        params,
    ).fetchall()

    if not articles:
        log.info("No unprocessed articles found.")
        conn.close()
        return result

    log.info(
        "Extracting signals from %d article(s) using %s%s...",
        len(articles), MODEL, " [DRY-RUN]" if dry_run else "",
    )

    for article in articles:
        company = {
            "id":     article["company_id"],
            "name":   article["company_name"],
            "sector": article["sector"],
            "stage":  article["stage"],
        }

        log.info(
            "  [%s] %s",
            article["company_id"],
            (article["title"] or "")[:70],
        )

        try:
            parsed = call_claude(client, article, company)
        except Exception as exc:
            result.errors.append(str(exc))
            log.warning("    Skipping due to API error: %s", exc)
            # Don't mark as processed — will retry next run
            continue

        if parsed is None:
            result.skipped_parse_error += 1
            log.warning("    Could not parse Claude response — marking processed to avoid retry loop")
            if not dry_run:
                mark_processed(conn, article["id"])
                conn.commit()
            continue

        signals = parsed.get("signals", [])
        if not isinstance(signals, list):
            signals = []

        if not signals:
            log.debug("    No signals extracted.")
            result.skipped_empty += 1
        else:
            log.debug("    %d signal(s):", len(signals))
            for s in signals:
                sv = _validate_signal(s)
                log.debug(
                    "      [%s] score=%d  %s",
                    sv["signal_type"], sv["relevance_score"], sv["summary"][:80],
                )
                if sv["people_mentioned"]:
                    names = ", ".join(
                        f"{p.get('name','?')} ({p.get('role','?')})"
                        for p in sv["people_mentioned"]
                    )
                    log.debug("      People: %s", names)

        if not dry_run:
            n = store_signals(conn, article, company, signals)
            result.signals_stored += n
            mark_processed(conn, article["id"])
            conn.commit()
        else:
            result.signals_stored += len(signals)

        result.processed += 1

    conn.close()
    log.info("\n%s", result)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured signals from news items using Claude"
    )
    parser.add_argument(
        "--limit", type=int, default=BATCH_SIZE,
        help=f"Max articles to process per run (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--company", type=str, default=None,
        help="Process only articles for this company id (e.g. snowflake)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Call Claude and print results but don't write to DB",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging (shows raw Claude responses and parsed signals)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    run(limit=args.limit, company_filter=args.company, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
