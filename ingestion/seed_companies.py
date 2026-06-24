"""Load companies.json into the SQLite companies table.

Upserts on company id — safe to re-run. Adds `ticker` and `notes` columns
if they don't exist yet (idempotent migration).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.init_db import init_db, DB_PATH

COMPANIES_JSON = Path(__file__).parent.parent / "data" / "companies.json"


def _ensure_extra_columns(conn) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(companies)").fetchall()}
    for col in ("ticker", "notes", "cik"):
        if col not in existing:
            conn.execute(f"ALTER TABLE companies ADD COLUMN {col} TEXT")
    conn.commit()


def seed(path: Path = COMPANIES_JSON, replace: bool = False) -> int:
    with open(path, encoding="utf-8") as f:
        companies = json.load(f)

    conn = init_db()
    _ensure_extra_columns(conn)

    mode = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
    inserted = updated = 0

    for c in companies:
        before = conn.execute("SELECT id FROM companies WHERE id = ?", (c["id"],)).fetchone()
        conn.execute(
            f"""{mode} INTO companies
               (id, name, domain, sector, stage, hq, description, keywords, ticker, notes, cik)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                c["id"],
                c["name"],
                c.get("domain"),
                c.get("sector"),
                c.get("stage"),
                c.get("hq"),
                c.get("description"),
                json.dumps(c.get("keywords", [])),
                c.get("ticker"),
                c.get("notes"),
                c.get("cik"),
            ),
        )
        if before is None:
            inserted += 1
        elif replace:
            updated += 1

    conn.commit()
    conn.close()
    return inserted, updated


def main() -> None:
    replace = "--replace" in sys.argv
    if replace:
        print("Mode: REPLACE (existing rows will be overwritten)")
    else:
        print("Mode: IGNORE (existing rows kept; use --replace to overwrite)")

    inserted, updated = seed(replace=replace)
    total = inserted + updated
    print(f"Done — {inserted} inserted, {updated} updated ({total} total) into {DB_PATH}")

    if not replace and total == 0:
        print("Tip: all companies already exist. Run with --replace to update them.")


if __name__ == "__main__":
    main()
