import sqlite3
import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Streamlit Cloud secret injection
# When running on Streamlit Cloud, API keys live in st.secrets rather than
# .env. Inject them into os.environ before anything else reads them so that
# all downstream code (anthropic, newsapi, etc.) works without changes.
# ---------------------------------------------------------------------------
try:
    import streamlit as st  # noqa: F401 — only available when running via Streamlit
    for _key in ("ANTHROPIC_API_KEY", "NEWSAPI_KEY"):
        if _key in st.secrets and _key not in os.environ:
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass  # not running under Streamlit, or secrets not configured — .env handles it

DB_PATH = Path(__file__).parent / "portfolio.db"
COMPANIES_JSON = Path(__file__).parent / "companies.json"


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS companies (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            domain      TEXT,
            sector      TEXT,
            stage       TEXT,
            hq          TEXT,
            description TEXT,
            keywords    TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS news_items (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id   TEXT REFERENCES companies(id),
            title        TEXT NOT NULL,
            source       TEXT,
            url          TEXT UNIQUE,
            published_at TEXT,
            content      TEXT,
            fetched_at   TEXT DEFAULT (datetime('now')),
            processed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS extracted_signals (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            news_item_id     INTEGER REFERENCES news_items(id),
            company_id       TEXT REFERENCES companies(id),
            signal_type      TEXT,
            summary          TEXT,
            people_mentioned TEXT,
            relevance_score  INTEGER,
            extracted_at     TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS executive_moves (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id      TEXT REFERENCES companies(id),
            news_item_id    INTEGER REFERENCES news_items(id),
            person_name     TEXT,
            old_role        TEXT,
            new_role        TEXT,
            move_type       TEXT,
            move_date       TEXT,
            -- cross-reference bridge fields (populated by cross_reference.py)
            from_company    TEXT,
            to_company      TEXT,
            source_signal_id INTEGER REFERENCES extracted_signals(id),
            confidence_note TEXT,
            detected_by     TEXT DEFAULT 'manual',
            recorded_at     TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_exec_person   ON executive_moves(person_name);
        CREATE INDEX IF NOT EXISTS idx_exec_bridge   ON executive_moves(from_company, to_company);
        CREATE INDEX IF NOT EXISTS idx_exec_detected ON executive_moves(detected_by);

        CREATE INDEX IF NOT EXISTS idx_news_company ON news_items(company_id);
        CREATE INDEX IF NOT EXISTS idx_signals_company ON extracted_signals(company_id);
        CREATE INDEX IF NOT EXISTS idx_signals_type    ON extracted_signals(signal_type);
        CREATE INDEX IF NOT EXISTS idx_signals_score   ON extracted_signals(relevance_score);
        CREATE INDEX IF NOT EXISTS idx_news_processed  ON news_items(processed_at);
        CREATE INDEX IF NOT EXISTS idx_exec_company ON executive_moves(company_id);

        CREATE TABLE IF NOT EXISTS edgar_filings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id      TEXT REFERENCES companies(id),
            ticker          TEXT NOT NULL,
            cik             TEXT,
            accession_no    TEXT UNIQUE NOT NULL,
            form_type       TEXT,
            filing_date     TEXT,
            period_of_report TEXT,
            filing_url      TEXT,
            document_url    TEXT,
            item_no         TEXT,
            snippet         TEXT,
            exec_signal     TEXT,
            fetched_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_edgar_company  ON edgar_filings(company_id);
        CREATE INDEX IF NOT EXISTS idx_edgar_ticker   ON edgar_filings(ticker);
        CREATE INDEX IF NOT EXISTS idx_edgar_date     ON edgar_filings(filing_date);
        CREATE INDEX IF NOT EXISTS idx_edgar_signal   ON edgar_filings(exec_signal);

        CREATE TABLE IF NOT EXISTS weekly_briefs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            brief_date      TEXT NOT NULL,
            week_label      TEXT NOT NULL,
            file_path       TEXT,
            body            TEXT NOT NULL,
            signal_count    INTEGER DEFAULT 0,
            move_count      INTEGER DEFAULT 0,
            companies_covered INTEGER DEFAULT 0,
            model_used      TEXT,
            generated_at    TEXT DEFAULT (datetime('now'))
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_briefs_date  ON weekly_briefs(brief_date);
        CREATE INDEX        IF NOT EXISTS idx_briefs_week  ON weekly_briefs(week_label);
    """)

    _migrate(conn)
    _seed_companies(conn)
    conn.commit()
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the initial schema — safe to run repeatedly."""
    existing_news = {r[1] for r in conn.execute("PRAGMA table_info(news_items)").fetchall()}
    existing_sig  = {r[1] for r in conn.execute("PRAGMA table_info(extracted_signals)").fetchall()}

    if "processed_at" not in existing_news:
        conn.execute("ALTER TABLE news_items ADD COLUMN processed_at TEXT")

    if "people_mentioned" not in existing_sig:
        conn.execute("ALTER TABLE extracted_signals ADD COLUMN people_mentioned TEXT")
    if "relevance_score" not in existing_sig:
        conn.execute("ALTER TABLE extracted_signals ADD COLUMN relevance_score INTEGER")

    existing_exec = {r[1] for r in conn.execute("PRAGMA table_info(executive_moves)").fetchall()}
    for col, typedef in [
        ("from_company",     "TEXT"),
        ("to_company",       "TEXT"),
        ("source_signal_id", "INTEGER"),
        ("confidence_note",  "TEXT"),
        ("detected_by",      "TEXT DEFAULT 'manual'"),
    ]:
        if col not in existing_exec:
            conn.execute(f"ALTER TABLE executive_moves ADD COLUMN {col} {typedef}")

    conn.commit()


def _seed_companies(conn: sqlite3.Connection) -> None:
    existing = conn.execute("SELECT id FROM companies").fetchall()
    if existing:
        return
    with open(COMPANIES_JSON) as f:
        companies = json.load(f)
    conn.executemany(
        "INSERT OR IGNORE INTO companies (id, name, domain, sector, stage, hq, description, keywords) VALUES (?,?,?,?,?,?,?,?)",
        [
            (
                c["id"], c["name"], c.get("domain"), c.get("sector"),
                c.get("stage"), c.get("hq"), c.get("description"),
                json.dumps(c.get("keywords", [])),
            )
            for c in companies
        ],
    )


if __name__ == "__main__":
    conn = init_db()
    count = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    print(f"Database initialized at {DB_PATH} with {count} companies.")
    conn.close()
