"""Ecosystem bridge detection: find people moving between portfolio companies.

Algorithm
---------
1. Load every extracted_signal where signal_type = 'leadership_change' that
   has a non-empty people_mentioned JSON list.
2. Build a name index: canonical_name → list of (signal, company, date, role).
3. For each name that appears in signals from two or more *distinct* companies,
   or appears in one portfolio signal and also in the text of a different signal
   with a different company attribution, emit a bridge candidate.
4. Rank candidates by a confidence heuristic:
     HIGH   — same name, 2+ portfolio companies, signals within 180 days
     MEDIUM — same name, portfolio + non-portfolio mention, or >180-day gap
     LOW    — fuzzy/partial name match across companies
5. Insert qualifying candidates into executive_moves (detected_by='cross_ref'),
   skipping rows where the same (person_name, from_company, to_company) already
   exists to stay idempotent.

Name matching
-------------
Exact canonical match (lowercase, whitespace-normalised) is HIGH confidence.
Fuzzy match (last-name-only or first+last transposition) is LOW confidence and
requires relevance_score >= 3 on both signals to be included.

Usage
-----
    python processing/cross_reference.py               # full scan
    python processing/cross_reference.py --days 180    # limit signal window
    python processing/cross_reference.py --min-score 3 # minimum relevance_score
    python processing/cross_reference.py --dry-run     # print only, no DB writes
    python processing/cross_reference.py --debug       # verbose name-matching detail
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.init_db import init_db

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DAYS      = 365   # scan signals from the last N days
DEFAULT_MIN_SCORE = 2     # ignore signals with relevance_score below this
FUZZY_MIN_SCORE   = 3     # floor for including LOW-confidence fuzzy matches
MAX_BRIDGE_GAP    = 180   # days between two signals to qualify as HIGH confidence

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SignalRecord:
    signal_id:   int
    company_id:  str
    company_name: str
    summary:     str
    person_name: str        # as extracted by Claude
    person_role: str
    signal_date: str        # extracted_at ISO string
    relevance:   int


@dataclass
class BridgeCandidate:
    person_name:       str
    from_signal:       SignalRecord
    to_signal:         SignalRecord
    confidence:        str          # HIGH / MEDIUM / LOW
    confidence_note:   str
    match_type:        str          # exact / fuzzy_last / fuzzy_partial

    @property
    def from_company(self) -> str:
        return self.from_signal.company_id

    @property
    def to_company(self) -> str:
        return self.to_signal.company_id

    @property
    def move_date(self) -> str:
        # Use the later signal's date as the effective move date
        return max(self.from_signal.signal_date, self.to_signal.signal_date)


# ---------------------------------------------------------------------------
# Name canonicalisation & matching
# ---------------------------------------------------------------------------

def _canon(name: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def _tokens(name: str) -> frozenset[str]:
    """Return non-trivial word tokens from a canonical name."""
    stop = {"the", "of", "and", "jr", "sr", "ii", "iii", "iv", "dr", "mr", "ms", "mrs"}
    return frozenset(t for t in _canon(name).split() if t not in stop and len(t) > 1)


def _last_name(name: str) -> str:
    parts = _canon(name).split()
    return parts[-1] if parts else ""


def match_names(a: str, b: str) -> tuple[str, str] | None:
    """
    Compare two name strings. Returns (match_type, note) or None if no match.
      exact         — canonical strings are identical
      fuzzy_last    — last names match and at least one first-name token overlaps
      fuzzy_partial — token sets share >= 2 tokens (catches middle-name variations)
    """
    ca, cb = _canon(a), _canon(b)
    if ca == cb:
        return "exact", f'Exact match: "{a}"'

    ta, tb = _tokens(a), _tokens(b)
    la, lb = _last_name(a), _last_name(b)

    if la and la == lb:
        # Last names match — require at least one other token in common
        first_overlap = (ta - {la}) & (tb - {lb})
        if first_overlap:
            return "fuzzy_last", f'Last-name + first-token match: "{a}" ↔ "{b}"'

    shared = ta & tb
    if len(shared) >= 2:
        return "fuzzy_partial", f'Partial token match ({", ".join(sorted(shared))}): "{a}" ↔ "{b}"'

    return None


# ---------------------------------------------------------------------------
# Signal loading
# ---------------------------------------------------------------------------

def load_leadership_signals(conn, since: str, min_score: int) -> list[SignalRecord]:
    """Return all leadership_change signals with at least one named person."""
    rows = conn.execute(
        """
        SELECT
            s.id            AS signal_id,
            s.company_id,
            s.summary,
            s.people_mentioned,
            s.relevance_score,
            s.extracted_at,
            c.name          AS company_name
        FROM extracted_signals s
        JOIN companies c ON c.id = s.company_id
        WHERE s.signal_type    = 'leadership_change'
          AND s.relevance_score >= ?
          AND s.extracted_at   >= ?
          AND s.people_mentioned IS NOT NULL
          AND s.people_mentioned != '[]'
          AND s.people_mentioned != 'null'
        ORDER BY s.extracted_at DESC
        """,
        (min_score, since),
    ).fetchall()

    records: list[SignalRecord] = []
    for row in rows:
        try:
            people = json.loads(row["people_mentioned"] or "[]")
        except json.JSONDecodeError:
            log.warning("Bad people_mentioned JSON for signal %d — skipping", row["signal_id"])
            continue

        if not isinstance(people, list):
            continue

        for person in people:
            name = (person.get("name") or "").strip()
            if not name or len(name) < 3:
                continue
            records.append(SignalRecord(
                signal_id    = row["signal_id"],
                company_id   = row["company_id"],
                company_name = row["company_name"],
                summary      = row["summary"] or "",
                person_name  = name,
                person_role  = (person.get("role") or "").strip(),
                signal_date  = row["extracted_at"] or "",
                relevance    = row["relevance_score"] or 0,
            ))

    log.info("Loaded %d person-signal records from %d DB rows", len(records), len(rows))
    return records


# ---------------------------------------------------------------------------
# Bridge detection
# ---------------------------------------------------------------------------

def _days_apart(date_a: str, date_b: str) -> int | None:
    """Return absolute day difference between two ISO datetime strings, or None."""
    fmt = "%Y-%m-%d"
    for a_str, b_str in [(date_a[:10], date_b[:10])]:
        try:
            da = datetime.strptime(a_str, fmt)
            db = datetime.strptime(b_str, fmt)
            return abs((da - db).days)
        except ValueError:
            return None


def _confidence(
    match_type: str,
    days: int | None,
    rel_a: int,
    rel_b: int,
) -> tuple[str, str]:
    """
    Return (confidence_level, note) for a candidate bridge.

    HIGH   exact match, signals ≤ MAX_BRIDGE_GAP days apart
    MEDIUM exact match but wide gap, OR fuzzy match with high relevance
    LOW    fuzzy match, or low relevance on either signal
    """
    gap_str = f"{days}d gap" if days is not None else "unknown gap"

    if match_type == "exact":
        if days is not None and days <= MAX_BRIDGE_GAP:
            return "HIGH", f"Exact name match, {gap_str}, both signals in portfolio"
        else:
            return "MEDIUM", f"Exact name match but {gap_str} — may be distinct roles"
    elif match_type == "fuzzy_last":
        if rel_a >= 4 and rel_b >= 4:
            return "MEDIUM", f"Last-name match, high relevance on both signals, {gap_str}"
        return "LOW", f"Last-name match, mixed relevance ({rel_a}/{rel_b}), {gap_str}"
    else:  # fuzzy_partial
        return "LOW", f"Partial token match, {gap_str} — review manually"


def detect_bridges(
    records: list[SignalRecord],
    fuzzy_min_score: int = FUZZY_MIN_SCORE,
) -> list[BridgeCandidate]:
    """
    Compare every pair of records from different companies and emit bridge
    candidates where names match.
    """
    # Group by company so we can quickly check cross-company pairs
    by_company: dict[str, list[SignalRecord]] = defaultdict(list)
    for rec in records:
        by_company[rec.company_id].append(rec)

    company_ids = list(by_company.keys())
    candidates: list[BridgeCandidate] = []

    # Seen set prevents emitting the same bridge twice (A→B and B→A)
    seen: set[tuple[int, int]] = set()

    for i, cid_a in enumerate(company_ids):
        for cid_b in company_ids[i + 1:]:
            if cid_a == cid_b:
                continue

            for rec_a in by_company[cid_a]:
                for rec_b in by_company[cid_b]:
                    pair_key = (min(rec_a.signal_id, rec_b.signal_id),
                                max(rec_a.signal_id, rec_b.signal_id))
                    if pair_key in seen:
                        continue

                    result = match_names(rec_a.person_name, rec_b.person_name)
                    if result is None:
                        continue

                    match_type, match_note = result

                    # Apply fuzzy score gate
                    if match_type != "exact":
                        if rec_a.relevance < fuzzy_min_score or rec_b.relevance < fuzzy_min_score:
                            log.debug(
                                "  Fuzzy match dropped (score gate): %s ↔ %s  rel=%d/%d",
                                rec_a.person_name, rec_b.person_name,
                                rec_a.relevance, rec_b.relevance,
                            )
                            seen.add(pair_key)
                            continue

                    days = _days_apart(rec_a.signal_date, rec_b.signal_date)
                    conf, conf_note = _confidence(match_type, days, rec_a.relevance, rec_b.relevance)

                    # Canonical person name: prefer the longer / more complete version
                    canon_name = (
                        rec_a.person_name
                        if len(rec_a.person_name) >= len(rec_b.person_name)
                        else rec_b.person_name
                    )

                    # Orient from→to by signal date (earlier is from)
                    if rec_a.signal_date <= rec_b.signal_date:
                        from_sig, to_sig = rec_a, rec_b
                    else:
                        from_sig, to_sig = rec_b, rec_a

                    candidates.append(BridgeCandidate(
                        person_name     = canon_name,
                        from_signal     = from_sig,
                        to_signal       = to_sig,
                        confidence      = conf,
                        confidence_note = f"{match_note}. {conf_note}",
                        match_type      = match_type,
                    ))
                    seen.add(pair_key)
                    log.debug(
                        "  Bridge [%s] %s → %s via %s  (%s)",
                        conf, from_sig.company_id, to_sig.company_id,
                        canon_name, match_type,
                    )

    # Sort: HIGH first, then MEDIUM, then LOW; within tier by date descending
    tier = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    candidates.sort(key=lambda c: (tier.get(c.confidence, 9), c.move_date), reverse=False)
    candidates.sort(key=lambda c: tier.get(c.confidence, 9))

    return candidates


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _already_stored(conn, person_name: str, from_company: str, to_company: str) -> bool:
    """Idempotency check: skip if this exact bridge is already in the table."""
    row = conn.execute(
        """
        SELECT 1 FROM executive_moves
        WHERE detected_by    = 'cross_ref'
          AND person_name    = ?
          AND from_company   = ?
          AND to_company     = ?
        LIMIT 1
        """,
        (person_name, from_company, to_company),
    ).fetchone()
    return row is not None


def store_bridge(conn, candidate: BridgeCandidate) -> bool:
    """Insert bridge into executive_moves. Returns True if newly inserted."""
    if _already_stored(conn, candidate.person_name, candidate.from_company, candidate.to_company):
        return False

    from_role = candidate.from_signal.person_role or None
    to_role   = candidate.to_signal.person_role   or None

    conn.execute(
        """
        INSERT INTO executive_moves
            (person_name, from_company, to_company, move_date,
             old_role, new_role,
             company_id, news_item_id,
             source_signal_id, confidence_note, detected_by,
             move_type)
        VALUES
            (?, ?, ?, ?,
             ?, ?,
             ?, ?,
             ?, ?, 'cross_ref',
             ?)
        """,
        (
            candidate.person_name,
            candidate.from_company,
            candidate.to_company,
            candidate.move_date,
            from_role,
            to_role,
            candidate.to_signal.company_id,      # company_id = destination company
            candidate.to_signal.signal_id,        # nearest news item (via signal)
            candidate.to_signal.signal_id,        # source_signal_id
            candidate.confidence_note,
            f"ecosystem_bridge_{candidate.confidence.lower()}",  # move_type
        ),
    )
    return True


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_candidate(c: BridgeCandidate) -> None:
    icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "⚪"}.get(c.confidence, "  ")
    print(
        f"\n  {icon} [{c.confidence}] {c.person_name}\n"
        f"     {c.from_signal.company_name} ({c.from_signal.person_role or '?'}) "
        f"→ {c.to_signal.company_name} ({c.to_signal.person_role or '?'})\n"
        f"     Date: {c.move_date[:10]}  |  Match: {c.match_type}\n"
        f"     Note: {c.confidence_note}\n"
        f"     From signal #{c.from_signal.signal_id}: {c.from_signal.summary[:80]}\n"
        f"     To   signal #{c.to_signal.signal_id}: {c.to_signal.summary[:80]}"
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    days: int = DEFAULT_DAYS,
    min_score: int = DEFAULT_MIN_SCORE,
    dry_run: bool = False,
) -> dict[str, Any]:
    conn = init_db()

    since = (
        datetime.now(timezone.utc) - timedelta(days=days)
    ).strftime("%Y-%m-%dT%H:%M:%S")

    log.info(
        "Cross-reference scan: leadership_change signals since %s (min_score=%d)%s",
        since[:10], min_score, " [DRY-RUN]" if dry_run else "",
    )

    records = load_leadership_signals(conn, since, min_score)
    if not records:
        log.info("No qualifying signals found — run extract_signals.py first.")
        conn.close()
        return {"candidates": 0, "stored": 0, "skipped": 0}

    company_ids_seen = {r.company_id for r in records}
    log.info(
        "Scanning %d person-mentions across %d companies for cross-portfolio bridges...",
        len(records), len(company_ids_seen),
    )

    candidates = detect_bridges(records)

    if not candidates:
        log.info("No cross-portfolio bridges detected.")
        conn.close()
        return {"candidates": 0, "stored": 0, "skipped": 0}

    log.info("Found %d bridge candidate(s):", len(candidates))

    stored = skipped = 0
    for c in candidates:
        _print_candidate(c)

        if dry_run:
            print("     [DRY-RUN] — not written to DB")
            continue

        if store_bridge(conn, c):
            stored += 1
            log.debug("  Stored bridge: %s → %s via %s", c.from_company, c.to_company, c.person_name)
        else:
            skipped += 1
            log.debug("  Skipped (already exists): %s", c.person_name)

    if not dry_run:
        conn.commit()

    conn.close()

    summary = {
        "candidates": len(candidates),
        "stored":     stored,
        "skipped":    skipped,
    }
    by_conf = defaultdict(int)
    for c in candidates:
        by_conf[c.confidence] += 1

    print(
        f"\n{'─'*60}\n"
        f"Scan complete — {len(candidates)} candidate(s) found:\n"
        + "\n".join(f"  {k}: {v}" for k, v in sorted(by_conf.items()))
        + f"\n{stored} new bridge(s) stored, {skipped} already existed."
        + (" [DRY-RUN — nothing written]" if dry_run else "")
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect ecosystem bridges: people moving across portfolio companies"
    )
    parser.add_argument(
        "--days", type=int, default=DEFAULT_DAYS,
        help=f"Scan signals from the last N days (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--min-score", type=int, default=DEFAULT_MIN_SCORE,
        dest="min_score",
        help=f"Minimum relevance_score to include (default: {DEFAULT_MIN_SCORE})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Detect and print bridges without writing to the database",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging (shows name comparisons and match details)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    run(days=args.days, min_score=args.min_score, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
