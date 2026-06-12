"""Generate a Weekly Ecosystem Brief from portfolio signals and executive moves.

Pulls this week's extracted_signals and executive_moves, sends a richly
structured context payload to Claude (claude-opus-4-8, adaptive thinking,
streaming), and writes the result as:

  1. A Markdown file  →  briefs/brief_YYYY-MM-DD.md
  2. A DB row         →  weekly_briefs table

Brief structure (enforced via system prompt):
  1. Executive Summary          — 3-4 sentence memo-style opener
  2. Per-Company Highlights     — only companies with notable signals
  3. Ecosystem Movement         — cross-portfolio executive moves with context
  4. People to Watch            — named individuals worth tracking

Usage:
    python processing/generate_brief.py               # current week
    python processing/generate_brief.py --days 14     # two-week window
    python processing/generate_brief.py --dry-run     # stream to terminal, no saves
    python processing/generate_brief.py --no-stream   # silent generation, saves at end
    python processing/generate_brief.py --rerun       # overwrite today's existing brief
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.init_db import init_db

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL      = "claude-opus-4-8"
MAX_TOKENS = 4096
BRIEFS_DIR = Path(__file__).parent.parent / "briefs"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior portfolio operations analyst at ICONIQ Growth, a top-tier growth \
equity firm. You write the internal Weekly Ecosystem Brief — a concise, opinionated \
memo distributed to partners every Monday morning.

Your writing style:
- Authoritative, direct, no filler. Partners have 5 minutes, not 20.
- Opinionated: flag what matters and why, not just what happened.
- Forward-looking: close each section with a "so what" for the portfolio.
- Internally candid: this is not a press release. Name risks plainly.

The brief MUST follow this exact Markdown structure with these exact section headers:

---

# Weekly Ecosystem Brief — {week_label}

## Executive Summary

[3-4 sentences. Lead with the single most important development. Name companies \
and people directly. End with an implication for the portfolio as a whole.]

## Portfolio Company Highlights

[Only include companies with meaningful signal this week. Skip companies with \
no activity — do not write placeholder entries. For each company, use this format:]

### {Company Name}

**Signal type · Relevance {score}/5**

[2-3 sentences of insight. What happened, why it matters, what to watch next.]

## Ecosystem Movement

[Executive moves, leadership transitions, and cross-portfolio people flow. \
Include context on why a particular move is significant — competitive implication, \
talent signal, relationship map. If no moves this week, write a single sentence \
noting the absence and what that calm might indicate.]

## People to Watch

[A bulleted list of named individuals who appeared in signals this week and are \
worth tracking in future weeks. Include: name, current or recent role, why they \
matter to the portfolio. 3-6 people max. Omit if signals contain no named people.]

---

Return only the Markdown document. No preamble, no sign-off, no commentary \
outside the structure above.\
"""

USER_TEMPLATE = """\
Today's date: {today}
Brief covers: {start_date} through {end_date}
Portfolio companies tracked: {company_count}

---

## Signals This Week ({signal_count} total)

{signals_block}

---

## Executive Moves This Week ({move_count} total)

{moves_block}

---

## Cross-Portfolio Bridges Detected

{bridges_block}

---

Write the Weekly Ecosystem Brief now.\
"""

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _fmt_date(dt_str: str) -> str:
    """Format a stored datetime string to a readable short form."""
    try:
        dt = datetime.strptime(dt_str[:10], "%Y-%m-%d")
        return f"{dt.strftime('%b')} {dt.day}"   # e.g. "Jun 9" — avoids %-d (Linux-only)
    except Exception:
        return dt_str[:10] if dt_str else "unknown date"


def collect_signals(conn, since: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT
            s.id,
            s.signal_type,
            s.summary,
            s.people_mentioned,
            s.relevance_score,
            s.extracted_at,
            c.name    AS company_name,
            c.sector,
            c.stage,
            ni.title  AS article_title,
            ni.source AS article_source,
            ni.url    AS article_url,
            ni.published_at
        FROM extracted_signals s
        JOIN companies  c  ON c.id  = s.company_id
        JOIN news_items ni ON ni.id = s.news_item_id
        WHERE s.extracted_at >= ?
        ORDER BY s.relevance_score DESC, s.extracted_at DESC
        """,
        (since,),
    ).fetchall()
    return [dict(r) for r in rows]


def collect_moves(conn, since: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT
            em.id,
            em.person_name,
            em.old_role,
            em.new_role,
            em.move_type,
            em.move_date,
            em.from_company,
            em.to_company,
            em.confidence_note,
            em.detected_by,
            em.recorded_at,
            c.name AS company_name,
            c.sector
        FROM executive_moves em
        LEFT JOIN companies c ON c.id = em.company_id
        WHERE em.recorded_at >= ?
        ORDER BY em.recorded_at DESC
        """,
        (since,),
    ).fetchall()
    return [dict(r) for r in rows]


def collect_bridges(conn, since: str) -> list[dict]:
    """Cross-portfolio bridges only (detected by cross_reference.py)."""
    rows = conn.execute(
        """
        SELECT
            em.person_name,
            em.old_role,
            em.new_role,
            em.from_company,
            em.to_company,
            em.confidence_note,
            em.move_date,
            cf.name  AS from_company_name,
            ct.name  AS to_company_name
        FROM executive_moves em
        LEFT JOIN companies cf ON cf.id = em.from_company
        LEFT JOIN companies ct ON ct.id = em.to_company
        WHERE em.detected_by = 'cross_ref'
          AND em.recorded_at >= ?
        ORDER BY em.move_date DESC
        """,
        (since,),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Context builders  (turn raw DB rows into readable prompt blocks)
# ---------------------------------------------------------------------------

def _signals_block(signals: list[dict]) -> str:
    if not signals:
        return "_No signals extracted this week._"

    # Group by company
    by_company: dict[str, list[dict]] = {}
    for s in signals:
        by_company.setdefault(s["company_name"], []).append(s)

    lines: list[str] = []
    for company, sigs in sorted(by_company.items(), key=lambda kv: -max(s["relevance_score"] or 0 for s in kv[1])):
        lines.append(f"### {company} ({sigs[0]['sector']})")
        for s in sigs:
            people_raw = s.get("people_mentioned") or "[]"
            try:
                people = json.loads(people_raw)
            except (json.JSONDecodeError, TypeError):
                people = []

            people_str = ""
            if people:
                names = ", ".join(
                    f"{p.get('name','?')} ({p.get('role','?')})" for p in people[:4]
                )
                people_str = f"\n  People: {names}"

            score = s.get("relevance_score") or "?"
            lines.append(
                f"- [{s['signal_type']}] Score {score}/5 — {s['summary']}"
                f"{people_str}"
                f"\n  Source: {s.get('article_title','')[:80]} ({s.get('article_source','')})"
                f"\n  Published: {_fmt_date(s.get('published_at',''))}"
            )
        lines.append("")

    return "\n".join(lines)


def _moves_block(moves: list[dict]) -> str:
    if not moves:
        return "_No executive moves recorded this week._"

    lines: list[str] = []
    for m in moves:
        company = m.get("company_name") or m.get("to_company") or "Unknown"
        old_r   = m.get("old_role")  or "—"
        new_r   = m.get("new_role")  or "—"
        mtype   = (m.get("move_type") or "move").replace("_", " ")
        date    = _fmt_date(m.get("move_date") or m.get("recorded_at") or "")
        lines.append(
            f"- **{m['person_name']}** at {company} | {mtype} | {date}\n"
            f"  {old_r} → {new_r}"
        )
    return "\n".join(lines)


def _bridges_block(bridges: list[dict]) -> str:
    if not bridges:
        return "_No cross-portfolio bridges detected this week._"

    lines: list[str] = []
    for b in bridges:
        frm  = b.get("from_company_name") or b.get("from_company") or "?"
        to   = b.get("to_company_name")   or b.get("to_company")   or "?"
        note = b.get("confidence_note")   or ""
        lines.append(
            f"- **{b['person_name']}**: {frm} → {to}\n"
            f"  {b.get('old_role','?')} → {b.get('new_role','?')}\n"
            f"  Confidence note: {note}"
        )
    return "\n".join(lines)


def build_user_message(
    signals: list[dict],
    moves: list[dict],
    bridges: list[dict],
    start_date: str,
    end_date: str,
    week_label: str,
    company_count: int,
) -> str:
    return USER_TEMPLATE.format(
        today       = end_date,
        start_date  = start_date,
        end_date    = end_date,
        week_label  = week_label,
        company_count = company_count,
        signal_count  = len(signals),
        move_count    = len(moves),
        signals_block = _signals_block(signals),
        moves_block   = _moves_block(moves),
        bridges_block = _bridges_block(bridges),
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    client: anthropic.Anthropic,
    user_message: str,
    week_label: str,
    stream_to_terminal: bool = True,
) -> str:
    """Call Claude with streaming. Returns the full brief text.

    On rate-limit or transient server errors, retries up to 2× with back-off.
    On mid-stream disconnection, returns whatever text was received so far with
    a warning appended — partial briefs are still useful.
    """
    import anthropic as _anthropic

    system = SYSTEM_PROMPT.replace("{week_label}", week_label)
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        chunks: list[str] = []
        try:
            with client.messages.stream(
                model         = MODEL,
                max_tokens    = MAX_TOKENS,
                thinking      = {"type": "adaptive"},
                output_config = {"effort": "high"},
                system        = system,
                messages      = [{"role": "user", "content": user_message}],
            ) as stream:
                for chunk in stream.text_stream:
                    chunks.append(chunk)
                    if stream_to_terminal:
                        print(chunk, end="", flush=True)

            if stream_to_terminal:
                print()
            return "".join(chunks)

        except _anthropic.RateLimitError as exc:
            wait = 60 * attempt
            log.warning("Rate limit on attempt %d — waiting %ds: %s", attempt, wait, exc)
            if attempt == max_attempts:
                raise
            time.sleep(wait)

        except _anthropic.APITimeoutError as exc:
            log.warning("Timeout on attempt %d: %s", attempt, exc)
            if attempt == max_attempts:
                raise
            time.sleep(10 * attempt)

        except _anthropic.APIConnectionError as exc:
            if chunks:
                # Mid-stream disconnection — salvage what we have
                log.warning("Stream disconnected after %d chunks: %s", len(chunks), exc)
                if stream_to_terminal:
                    print()
                partial = "".join(chunks)
                return partial + "\n\n---\n*Brief truncated due to connection error.*\n"
            log.warning("Connection error on attempt %d: %s", attempt, exc)
            if attempt == max_attempts:
                raise
            time.sleep(5 * attempt)

        except _anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                log.warning("Server error %d on attempt %d", exc.status_code, attempt)
                if attempt == max_attempts:
                    raise
                time.sleep(15 * attempt)
            else:
                raise   # 4xx — don't retry

    return ""  # unreachable


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_to_file(brief_text: str, date_str: str) -> Path:
    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    path = BRIEFS_DIR / f"brief_{date_str}.md"
    path.write_text(brief_text, encoding="utf-8")
    return path


def save_to_db(
    conn,
    brief_text: str,
    brief_date: str,
    week_label: str,
    file_path: Path | None,
    signal_count: int,
    move_count: int,
    companies_covered: int,
    rerun: bool = False,
) -> int:
    if rerun:
        conn.execute("DELETE FROM weekly_briefs WHERE brief_date = ?", (brief_date,))

    conn.execute(
        """
        INSERT OR IGNORE INTO weekly_briefs
            (brief_date, week_label, file_path, body,
             signal_count, move_count, companies_covered, model_used)
        VALUES
            (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            brief_date,
            week_label,
            str(file_path) if file_path else None,
            brief_text,
            signal_count,
            move_count,
            companies_covered,
            MODEL,
        ),
    )
    conn.commit()
    return conn.execute(
        "SELECT id FROM weekly_briefs WHERE brief_date = ?", (brief_date,)
    ).fetchone()["id"]


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run(
    days: int = 7,
    dry_run: bool = False,
    stream_to_terminal: bool = True,
    rerun: bool = False,
) -> Path | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set — add it to your .env file")

    conn   = init_db()
    client = anthropic.Anthropic(api_key=api_key)

    now        = datetime.now(timezone.utc)
    end_dt     = now
    start_dt   = now - timedelta(days=days)
    since      = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
    brief_date = now.strftime("%Y-%m-%d")
    week_label = f"Week of {now.strftime('%B')} {now.day}, {now.year}"

    # Guard: don't regenerate today's brief unless --rerun
    if not rerun and not dry_run:
        existing = conn.execute(
            "SELECT id, file_path FROM weekly_briefs WHERE brief_date = ?", (brief_date,)
        ).fetchone()
        if existing:
            log.warning(
                "Brief for %s already exists (id=%d, path=%s). Use --rerun to overwrite.",
                brief_date, existing["id"], existing["file_path"],
            )
            conn.close()
            return Path(existing["file_path"]) if existing["file_path"] else None

    # Collect data
    signals = collect_signals(conn, since)
    moves   = collect_moves(conn, since)
    bridges = collect_bridges(conn, since)

    companies_with_signals = len({s["company_name"] for s in signals})
    total_companies = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]

    if not signals and not moves:
        log.warning(
            "No signals or moves found for the past %d days. "
            "Run ingestion → extract_signals → cross_reference first.",
            days,
        )
        conn.close()
        return None

    log.info(
        "Generating %s: %d signals, %d moves, %d bridges across %d/%d companies",
        week_label, len(signals), len(moves), len(bridges),
        companies_with_signals, total_companies,
    )

    user_message = build_user_message(
        signals          = signals,
        moves            = moves,
        bridges          = bridges,
        start_date       = start_dt.strftime("%Y-%m-%d"),
        end_date         = end_dt.strftime("%Y-%m-%d"),
        week_label       = week_label,
        company_count    = total_companies,
    )

    if not stream_to_terminal:
        log.info("Generating brief (silent)...")
    else:
        print(f"\n{'─'*70}")
        print(f"  ICONIQ Growth — {week_label}")
        print(f"{'─'*70}\n")

    brief_text = generate(client, user_message, week_label, stream_to_terminal)

    if dry_run:
        log.info("\n[DRY-RUN] Brief generated but not saved to file or database.")
        conn.close()
        return None

    # Save file
    file_path = save_to_file(brief_text, brief_date)
    log.info("Saved to %s", file_path)

    # Save to DB
    brief_id = save_to_db(
        conn              = conn,
        brief_text        = brief_text,
        brief_date        = brief_date,
        week_label        = week_label,
        file_path         = file_path,
        signal_count      = len(signals),
        move_count        = len(moves),
        companies_covered = companies_with_signals,
        rerun             = rerun,
    )
    log.info("Stored in weekly_briefs (id=%d)", brief_id)

    conn.close()
    return file_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Weekly Ecosystem Brief from portfolio signals"
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Lookback window in days (default: 7)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Stream brief to terminal without saving to file or database",
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Generate silently (no terminal output until complete)",
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help="Overwrite today's existing brief if one already exists",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level   = logging.DEBUG if args.debug else logging.INFO,
        format  = "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt = "%H:%M:%S",
    )

    path = run(
        days               = args.days,
        dry_run            = args.dry_run,
        stream_to_terminal = not args.no_stream,
        rerun              = args.rerun,
    )

    if path:
        print(f"\nBrief saved → {path}")


if __name__ == "__main__":
    main()
