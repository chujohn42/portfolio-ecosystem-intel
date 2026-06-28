"""Ecosystem bridge detection: find people moving between portfolio companies.

This script runs two independent detection passes, both writing to the same
executive_moves table but tagged with different `detected_by` values:

PASS 1 — Cross-portfolio bridges (detected_by='cross_ref')
------------------------------------------------------------
1. Load every extracted_signal where signal_type = 'leadership_change' that
   has a non-empty people_mentioned JSON list.
2. Build a name index: canonical_name → list of (signal, company, date, role).
3. For each name that appears in signals from two or more *distinct* companies,
   emit a bridge candidate.
4. Rank candidates by a confidence heuristic:
     HIGH   — same name, 2+ portfolio companies, signals within 180 days
     MEDIUM — same name, portfolio + non-portfolio mention, or >180-day gap
     LOW    — fuzzy/partial name match across companies
5. Insert qualifying candidates, skipping rows where the same
   (person_name, from_company, to_company) already exists.

PASS 2 — EDGAR-sourced single-company moves (detected_by='edgar')
------------------------------------------------------------------
1. Load every edgar_filings row with exec_signal IN ('appointment','departure','both')
   — these come from SEC 8-K Item 5.02 filings fetched by ingestion/fetch_edgar.py.
2. Run regex-based person-name extraction over the filing snippet (best-effort —
   SEC filing prose is unstructured; extraction is heuristic, not Claude-based).
3. Insert a row per identified name even when no cross-company match exists:
     appointment → to_company   = the filing's company, from_company = NULL (origin unknown / external hire)
     departure   → from_company = the filing's company, to_company   = NULL (destination unknown)
4. Idempotent on (detected_by='edgar', source_filing_id, person_name, move_type).

Name matching (Pass 1 only)
----------------------------
Exact canonical match (lowercase, whitespace-normalised) is HIGH confidence.
Fuzzy match (last-name-only or first+last transposition) is LOW confidence and
requires relevance_score >= 3 on both signals to be included.

Usage
-----
    python processing/cross_reference.py                 # full scan (bridges + EDGAR)
    python processing/cross_reference.py --days 180      # bridge signal window (default: 365)
    python processing/cross_reference.py --edgar-days 90 # EDGAR filing window (default: 90, matches fetch_edgar.py)
    python processing/cross_reference.py --min-score 3   # minimum relevance_score (bridges only)
    python processing/cross_reference.py --no-edgar      # skip the EDGAR detection pass
    python processing/cross_reference.py --dry-run       # print only, no DB writes
    python processing/cross_reference.py --debug         # verbose name-matching detail
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

# Windows consoles default to the legacy cp1252 codepage, which can't encode
# the emoji used in our print() status output (🔴 🟢 etc). Force UTF-8 so this
# script runs cleanly in a plain `python` invocation on Windows, not just
# inside terminals that already default to UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass  # stdout doesn't support reconfigure (e.g. redirected/piped in some environments)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DAYS       = 365   # scan leadership_change signals from the last N days
EDGAR_LOOKBACK_DAYS = 90   # matches ingestion/fetch_edgar.py's own lookback window —
                           # kept as a dedicated window (independent of --days) so the
                           # EDGAR pass always scans the same period fetch_edgar.py just fetched
DEFAULT_MIN_SCORE  = 2     # ignore signals with relevance_score below this
FUZZY_MIN_SCORE    = 3     # floor for including LOW-confidence fuzzy matches
MAX_BRIDGE_GAP     = 180   # days between two signals to qualify as HIGH confidence

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
# EDGAR-sourced appointment / departure detection (Pass 2)
#
# edgar_filings.snippet holds raw, unstructured prose pulled from SEC 8-K
# filings (see ingestion/fetch_edgar.py). There is no structured person field,
# so we extract a candidate name with regex heuristics. This is intentionally
# conservative — filings where no name can be confidently isolated are logged
# and skipped rather than stored with a placeholder name.
# ---------------------------------------------------------------------------

# A "name" is 2-4 capitalized tokens (allows middle initials like "Jane A. Smith")
_NAME_TOKEN = r"[A-Z][a-zA-Z'’\-]+\.?"
_NAME_GROUP = rf"((?:{_NAME_TOKEN}\s+){{1,3}}{_NAME_TOKEN})"

# Words that occasionally get swept into a name match by mistake (titles, org
# boilerplate) — if every token in a candidate is one of these, reject it.
_ROLE_STOPWORDS = {
    "chief", "executive", "officer", "financial", "operating", "technology",
    "revenue", "president", "general", "counsel", "board", "directors",
    "company", "corporation", "item", "form", "vice", "senior", "interim",
}

_APPOINTMENT_PATTERNS = [
    re.compile(rf"(?i:appoint(?:ed|s|ing))\s+{_NAME_GROUP}\s+(?i:as|to)\b"),
    re.compile(rf"{_NAME_GROUP}\s+(?i:has been|was)\s+(?i:appointed|named|elected)\b"),
    re.compile(rf"{_NAME_GROUP}\s+(?i:will\s+(?:serve as|become|join))\b"),
    re.compile(rf"(?i:mr\.|ms\.|mrs\.|dr\.)\s+{_NAME_GROUP}\b"),
]

_DEPARTURE_PATTERNS = [
    re.compile(rf"(?i:resignation|departure)\s+of\s+{_NAME_GROUP}\b"),
    re.compile(rf"{_NAME_GROUP}\s+(?i:resigned|departed|stepped down|is leaving|will depart)\b"),
]


def extract_person_name(snippet: str, patterns: list[re.Pattern]) -> str | None:
    """Try each pattern in order, return the first plausible name match."""
    for pat in patterns:
        m = pat.search(snippet)
        if not m:
            continue
        candidate = m.group(1).strip()
        tokens = [t.strip(".").lower() for t in candidate.split()]
        if len(tokens) < 2:
            continue   # require at least first + last name
        if all(t in _ROLE_STOPWORDS for t in tokens):
            continue   # entire match is title/boilerplate, not a person
        return candidate
    return None


def load_edgar_signals(conn, since: str) -> list[dict]:
    """Return edgar_filings rows that indicate an executive appointment or departure."""
    rows = conn.execute(
        """
        SELECT id, company_id, ticker, accession_no, filing_date, fetched_at, snippet, exec_signal
        FROM edgar_filings
        WHERE exec_signal IN ('appointment', 'departure', 'both')
          AND COALESCE(filing_date, fetched_at) >= ?
        ORDER BY COALESCE(filing_date, fetched_at) DESC
        """,
        (since,),
    ).fetchall()
    log.info("Loaded %d EDGAR filing(s) with appointment/departure signal", len(rows))
    return [dict(r) for r in rows]


def detect_edgar_moves(filings: list[dict]) -> list[dict]:
    """
    For each filing, attempt to extract a person name for its exec_signal direction(s).
    Returns a list of {filing, person_name, direction, note} dicts.
    """
    results: list[dict] = []

    for f in filings:
        snippet = (f.get("snippet") or "").strip()
        if not snippet:
            log.debug("  Filing %s has no snippet — skipping", f["accession_no"])
            continue

        signal = f["exec_signal"]
        appt_name = extract_person_name(snippet, _APPOINTMENT_PATTERNS) if signal in ("appointment", "both") else None
        dep_name  = extract_person_name(snippet, _DEPARTURE_PATTERNS)  if signal in ("departure", "both") else None

        # Same name matched both directions — likely ambiguous transition language;
        # record once as an appointment and flag for manual review.
        if appt_name and dep_name and _canon(appt_name) == _canon(dep_name):
            results.append({
                "filing": f, "person_name": appt_name, "direction": "appointment",
                "note": (
                    f"EDGAR 8-K Item 5.02 ({f['ticker']}, accession {f['accession_no']}) — "
                    f"ambiguous filing language matched both appointment and departure patterns "
                    f"for the same name; recorded as appointment. Recommend manual review."
                ),
            })
            log.debug("  Ambiguous both-direction match for %s in %s", appt_name, f["accession_no"])
            continue

        if appt_name:
            results.append({
                "filing": f, "person_name": appt_name, "direction": "appointment",
                "note": (
                    f"EDGAR 8-K Item 5.02 ({f['ticker']}, accession {f['accession_no']}) — "
                    f"appointment detected via regex extraction from filing snippet."
                ),
            })
            log.debug("  Appointment: %s @ %s", appt_name, f["ticker"])

        if dep_name:
            results.append({
                "filing": f, "person_name": dep_name, "direction": "departure",
                "note": (
                    f"EDGAR 8-K Item 5.02 ({f['ticker']}, accession {f['accession_no']}) — "
                    f"departure detected via regex extraction from filing snippet."
                ),
            })
            log.debug("  Departure: %s @ %s", dep_name, f["ticker"])

        if signal in ("appointment", "both") and not appt_name:
            log.debug("  No appointment name extracted from %s (%s)", f["accession_no"], f["ticker"])
        if signal in ("departure", "both") and not dep_name:
            log.debug("  No departure name extracted from %s (%s)", f["accession_no"], f["ticker"])

    return results


def _print_edgar_move(mv: dict) -> None:
    f = mv["filing"]
    icon = "🟢" if mv["direction"] == "appointment" else "🔴"
    print(
        f"\n  {icon} [EDGAR {mv['direction'].upper()}] {mv['person_name']}\n"
        f"     Company: {f['ticker']}  |  Filing: {f['accession_no']}  |  "
        f"Date: {(f['filing_date'] or f['fetched_at'] or '')[:10]}\n"
        f"     Note: {mv['note']}"
    )


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


def _edgar_already_stored(conn, filing_id: int, person_name: str, move_type: str) -> bool:
    """Idempotency check for EDGAR-sourced moves, keyed by the source filing."""
    row = conn.execute(
        """
        SELECT 1 FROM executive_moves
        WHERE detected_by       = 'edgar'
          AND source_filing_id  = ?
          AND person_name       = ?
          AND move_type         = ?
        LIMIT 1
        """,
        (filing_id, person_name, move_type),
    ).fetchone()
    return row is not None


def store_edgar_move(conn, mv: dict) -> bool:
    """Insert an EDGAR-sourced appointment/departure into executive_moves.

    appointment → to_company   = filing's company, from_company = NULL (origin unknown / external hire)
    departure   → from_company = filing's company, to_company   = NULL (destination unknown)
    """
    f = mv["filing"]
    move_type = f"edgar_{mv['direction']}"

    if _edgar_already_stored(conn, f["id"], mv["person_name"], move_type):
        return False

    if mv["direction"] == "appointment":
        from_company, to_company = None, f["company_id"]
    else:
        from_company, to_company = f["company_id"], None

    move_date = f["filing_date"] or f["fetched_at"]

    conn.execute(
        """
        INSERT INTO executive_moves
            (person_name, from_company, to_company, move_date,
             company_id, source_filing_id, confidence_note, detected_by, move_type)
        VALUES
            (?, ?, ?, ?, ?, ?, ?, 'edgar', ?)
        """,
        (
            mv["person_name"],
            from_company,
            to_company,
            move_date,
            f["company_id"],
            f["id"],
            mv["note"],
            move_type,
        ),
    )
    return True


# ---------------------------------------------------------------------------
# Fallback seed data
#
# Used only when the EDGAR pass finds zero usable exec-move data across the
# entire portfolio — e.g. on a fresh deploy before fetch_edgar.py has run,
# if SEC EDGAR is temporarily unreachable, or while CIKs for recently-IPO'd
# companies (ServiceTitan, Figma) are still unverified in companies.json.
# This guarantees the Ecosystem Map always has at least a few real data
# points to display instead of an empty table.
#
# Records live in data/fallback_exec_moves.json, not in this file — they are
# real, publicly reported executive transitions, included from general
# knowledge rather than fetched live. They are NOT a substitute for the
# EDGAR pipeline and should be treated as illustrative seed data. Verify
# exact dates/details against primary sources (company press releases, SEC
# filings) before relying on them in any external-facing context.
# ---------------------------------------------------------------------------

FALLBACK_EXEC_MOVES_PATH = Path(__file__).parent.parent / "data" / "fallback_exec_moves.json"


def _load_fallback_exec_moves() -> list[dict]:
    """Load fallback exec-move records from data/fallback_exec_moves.json."""
    if not FALLBACK_EXEC_MOVES_PATH.exists():
        log.warning("Fallback exec moves file not found at %s — skipping.", FALLBACK_EXEC_MOVES_PATH)
        return []
    with open(FALLBACK_EXEC_MOVES_PATH, encoding="utf-8") as f:
        return json.load(f)


def _fallback_already_seeded(conn) -> bool:
    """Only seed once — if any 'manual' row exists, assume it was already done."""
    row = conn.execute(
        "SELECT 1 FROM executive_moves WHERE detected_by = 'manual' LIMIT 1"
    ).fetchone()
    return row is not None


def seed_fallback_exec_moves(conn, dry_run: bool = False) -> int:
    """Insert fallback exec moves (from data/fallback_exec_moves.json) so the
    Ecosystem Map is never empty.

    Only runs if the target companies actually exist in this portfolio and
    no 'manual' rows have been seeded before — safe to call on every run.
    """
    if _fallback_already_seeded(conn):
        log.debug("Fallback exec moves already seeded — skipping.")
        return 0

    fallback_moves = _load_fallback_exec_moves()
    if not fallback_moves:
        return 0

    inserted = 0
    for mv in fallback_moves:
        company = conn.execute(
            "SELECT id FROM companies WHERE id = ?", (mv["company_id"],)
        ).fetchone()
        if not company:
            log.debug("  Fallback move for '%s' skipped — company not in this portfolio.", mv["company_id"])
            continue

        icon = "🟢" if mv["direction"] == "appointment" else "🔴"
        print(
            f"\n  {icon} [FALLBACK SEED] {mv['person_name']} — {mv['company_id']} "
            f"({mv['direction']})"
        )

        if dry_run:
            print("     [DRY-RUN] — not written to DB")
            continue

        from_company = mv["company_id"] if mv["direction"] == "departure"   else None
        to_company   = mv["company_id"] if mv["direction"] == "appointment" else None
        move_type    = f"manual_{mv['direction']}"

        conn.execute(
            """
            INSERT INTO executive_moves
                (person_name, from_company, to_company, move_date,
                 old_role, new_role, company_id,
                 confidence_note, detected_by, move_type)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, 'manual', ?)
            """,
            (
                mv["person_name"], from_company, to_company, mv["move_date"],
                mv["old_role"], mv["new_role"], mv["company_id"],
                mv["note"], move_type,
            ),
        )
        inserted += 1

    if not dry_run and inserted:
        conn.commit()

    return inserted


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
    include_edgar: bool = True,
    edgar_days: int = EDGAR_LOOKBACK_DAYS,
) -> dict[str, Any]:
    conn = init_db()

    since = (
        datetime.now(timezone.utc) - timedelta(days=days)
    ).strftime("%Y-%m-%dT%H:%M:%S")

    # EDGAR gets its own (narrower, by default) window so it matches the
    # period ingestion/fetch_edgar.py actually fetched, independent of
    # whatever --days is set to for the leadership-signal bridge scan.
    edgar_since = (
        datetime.now(timezone.utc) - timedelta(days=edgar_days)
    ).strftime("%Y-%m-%dT%H:%M:%S")

    log.info(
        "Cross-reference scan: signals since %s, EDGAR filings since %s (min_score=%d)%s",
        since[:10], edgar_since[:10], min_score, " [DRY-RUN]" if dry_run else "",
    )

    # ── Pass 1: cross-portfolio bridges from extracted_signals ──────────────
    bridge_candidates: list[BridgeCandidate] = []
    bridge_stored = bridge_skipped = 0

    records = load_leadership_signals(conn, since, min_score)
    if not records:
        log.info("No qualifying leadership_change signals found for bridge detection.")
    else:
        company_ids_seen = {r.company_id for r in records}
        log.info(
            "Scanning %d person-mentions across %d companies for cross-portfolio bridges...",
            len(records), len(company_ids_seen),
        )
        bridge_candidates = detect_bridges(records)

        if not bridge_candidates:
            log.info("No cross-portfolio bridges detected.")
        else:
            log.info("Found %d bridge candidate(s):", len(bridge_candidates))
            for c in bridge_candidates:
                _print_candidate(c)
                if dry_run:
                    print("     [DRY-RUN] — not written to DB")
                    continue
                if store_bridge(conn, c):
                    bridge_stored += 1
                    log.debug("  Stored bridge: %s → %s via %s", c.from_company, c.to_company, c.person_name)
                else:
                    bridge_skipped += 1
                    log.debug("  Skipped (already exists): %s", c.person_name)

            if not dry_run:
                conn.commit()

    # ── Pass 2: EDGAR-sourced single-company appointments/departures ────────
    edgar_moves: list[dict] = []
    edgar_stored = edgar_skipped = 0

    if include_edgar:
        edgar_filings = load_edgar_signals(conn, edgar_since)
        if not edgar_filings:
            log.info("No EDGAR filings with appointment/departure signal in this window.")
        else:
            edgar_moves = detect_edgar_moves(edgar_filings)
            if not edgar_moves:
                log.info("No person names could be extracted from EDGAR filings.")
            else:
                log.info("Found %d EDGAR-sourced exec move candidate(s):", len(edgar_moves))
                for mv in edgar_moves:
                    _print_edgar_move(mv)
                    if dry_run:
                        print("     [DRY-RUN] — not written to DB")
                        continue
                    if store_edgar_move(conn, mv):
                        edgar_stored += 1
                        log.debug("  Stored EDGAR move: %s (%s)", mv["person_name"], mv["direction"])
                    else:
                        edgar_skipped += 1
                        log.debug("  Skipped (already exists): %s", mv["person_name"])

                if not dry_run:
                    conn.commit()
    else:
        log.info("EDGAR detection pass skipped (--no-edgar).")

    # ── Fallback: seed hardcoded real exec moves if EDGAR found nothing ─────
    # Triggers when there are zero edgar-sourced rows in the table at all
    # (not just zero new ones this run) — so once real EDGAR data exists,
    # the fallback never re-runs, but a fresh/empty deploy always gets
    # something meaningful to display on the Ecosystem Map.
    fallback_seeded = 0
    edgar_total_in_db = conn.execute(
        "SELECT COUNT(*) FROM executive_moves WHERE detected_by = 'edgar'"
    ).fetchone()[0]

    if edgar_total_in_db == 0:
        log.info(
            "EDGAR has produced zero exec moves across the whole portfolio — "
            "seeding fallback data so the Ecosystem Map isn't empty."
        )
        fallback_seeded = seed_fallback_exec_moves(conn, dry_run=dry_run)
        if fallback_seeded:
            log.info("Seeded %d fallback exec move(s).", fallback_seeded)
        elif not dry_run:
            log.info("No fallback rows seeded (already seeded, or no matching companies in this portfolio).")

    conn.close()

    summary = {
        "bridge_candidates": len(bridge_candidates),
        "bridge_stored":     bridge_stored,
        "bridge_skipped":    bridge_skipped,
        "edgar_candidates":  len(edgar_moves),
        "edgar_stored":      edgar_stored,
        "edgar_skipped":     edgar_skipped,
        "fallback_seeded":   fallback_seeded,
    }

    by_conf = defaultdict(int)
    for c in bridge_candidates:
        by_conf[c.confidence] += 1

    fallback_line = (
        f"\n  Fallback seed data: {fallback_seeded} row(s) inserted "
        f"(EDGAR had zero results across the portfolio)"
        if fallback_seeded else ""
    )

    print(
        f"\n{'─'*60}\n"
        f"Scan complete\n"
        f"  Cross-portfolio bridges: {len(bridge_candidates)} candidate(s)"
        + ("".join(f"\n    {k}: {v}" for k, v in sorted(by_conf.items())) if by_conf else "")
        + f"\n    → {bridge_stored} new, {bridge_skipped} already existed\n"
        f"  EDGAR appointments/departures: {len(edgar_moves)} candidate(s)\n"
        f"    → {edgar_stored} new, {edgar_skipped} already existed"
        + fallback_line
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
        "--no-edgar", action="store_true", dest="no_edgar",
        help="Skip the EDGAR appointment/departure detection pass",
    )
    parser.add_argument(
        "--edgar-days", type=int, default=EDGAR_LOOKBACK_DAYS,
        dest="edgar_days",
        help=f"EDGAR filing lookback window in days, independent of --days (default: {EDGAR_LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Detect and print bridges/moves without writing to the database",
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

    run(
        days=args.days,
        min_score=args.min_score,
        dry_run=args.dry_run,
        include_edgar=not args.no_edgar,
        edgar_days=args.edgar_days,
    )


if __name__ == "__main__":
    main()
