"""Portfolio Ecosystem Intelligence — Streamlit Dashboard.

Tabs:
  1. Weekly Brief    — rendered markdown, week selector, download button
  2. Company Detail  — per-company news + signals table
  3. Ecosystem Map   — executive moves table + network graph

Sidebar:
  Run Full Pipeline — fetch_news → extract_signals → cross_reference → generate_brief
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from data.init_db import init_db, DB_PATH  # noqa: E402

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title = "Portfolio Ecosystem Intelligence",
    page_icon  = "🔭",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

BRIEFS_DIR        = ROOT / "briefs"
COMPANIES_JSON    = ROOT / "data" / "companies.json"

# ---------------------------------------------------------------------------
# DB bootstrap — cached so it runs exactly once per Streamlit process.
#
# Streamlit Cloud resets the filesystem on every deploy/restart, so
# portfolio.db is always missing at boot. This function:
#   1. Calls init_db() — creates all tables and indexes (idempotent DDL)
#   2. Checks whether the companies table is empty
#   3. If empty, seeds from data/companies.json on the same connection
#
# Using @st.cache_resource means the returned connection is reused for every
# rerun in the same session, and the setup block only executes once per
# Streamlit worker process — not on every user interaction.
# ---------------------------------------------------------------------------

@st.cache_resource
def get_db():
    """Return a fully-initialised, seeded SQLite connection."""
    conn = init_db()   # creates schema + runs _migrate(); safe to call repeatedly

    # Seed companies if the table is empty (fresh DB on Streamlit Cloud, or first run)
    count = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    if count == 0:
        _seed_companies(conn)

    return conn


def _seed_companies(conn) -> None:
    """Insert companies from data/companies.json into an open connection."""
    import json as _json

    if not COMPANIES_JSON.exists():
        st.warning(f"companies.json not found at {COMPANIES_JSON} — skipping seed.")
        return

    with open(COMPANIES_JSON, encoding="utf-8") as f:
        companies = _json.load(f)

    # Ensure ticker/notes columns exist (added after initial schema)
    existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(companies)").fetchall()}
    for col in ("ticker", "notes"):
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE companies ADD COLUMN {col} TEXT")

    conn.executemany(
        """
        INSERT OR IGNORE INTO companies
            (id, name, domain, sector, stage, hq, description, keywords, ticker, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                c["id"], c["name"],
                c.get("domain"), c.get("sector"), c.get("stage"), c.get("hq"),
                c.get("description"), _json.dumps(c.get("keywords", [])),
                c.get("ticker"), c.get("notes"),
            )
            for c in companies
        ],
    )
    conn.commit()


def db():
    """Return the shared, bootstrapped connection."""
    return get_db()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIGNAL_TYPE_LABELS = {
    "leadership_change":   "👤 Leadership Change",
    "funding_partnership": "💰 Funding / Partnership",
    "notable_hire":        "➕ Notable Hire",
    "product_launch":      "🚀 Product Launch",
    "other":               "📌 Other",
}

SIGNAL_TYPE_COLORS = {
    "leadership_change":   "#d62728",
    "funding_partnership": "#2ca02c",
    "notable_hire":        "#1f77b4",
    "product_launch":      "#ff7f0e",
    "other":               "#7f7f7f",
}


def _score_stars(score: int | None) -> str:
    s = int(score or 0)
    return "★" * s + "☆" * (5 - s)


def _label(signal_type: str) -> str:
    return SIGNAL_TYPE_LABELS.get(signal_type, signal_type.replace("_", " ").title())


def _people_display(raw: str | None) -> str:
    if not raw:
        return ""
    try:
        people = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(people, list):
        return ""
    return ", ".join(
        f"{p.get('name','?')} ({p.get('role','?')})" for p in people[:4]
    )


def _since(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Sidebar — pipeline runner
# ---------------------------------------------------------------------------

def render_sidebar():
    st.sidebar.title("🔭 Portfolio Intel")
    st.sidebar.caption(f"DB: `{DB_PATH.name}`")
    st.sidebar.divider()

    st.sidebar.subheader("Pipeline")
    if st.sidebar.button("▶  Run Full Pipeline Now", use_container_width=True, type="primary"):
        _run_pipeline()

    st.sidebar.divider()
    st.sidebar.subheader("Database")
    conn = db()
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Companies",  conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0])
    c2.metric("News items", conn.execute("SELECT COUNT(*) FROM news_items").fetchone()[0])
    c3, c4 = st.sidebar.columns(2)
    c3.metric("Signals",    conn.execute("SELECT COUNT(*) FROM extracted_signals").fetchone()[0])
    c4.metric("Exec moves", conn.execute("SELECT COUNT(*) FROM executive_moves").fetchone()[0])

    st.sidebar.divider()
    unprocessed = conn.execute(
        "SELECT COUNT(*) FROM news_items WHERE processed_at IS NULL"
    ).fetchone()[0]
    if unprocessed > 0:
        st.sidebar.warning(f"{unprocessed} article(s) not yet processed. Run the pipeline.")
    else:
        st.sidebar.success("All articles processed.")


def _run_pipeline():
    """Run the four pipeline scripts in sequence, streaming output to a Streamlit container."""
    steps = [
        ("📰 Fetching news",           [sys.executable, str(ROOT / "ingestion"  / "fetch_news.py")]),
        ("🔍 Extracting signals",      [sys.executable, str(ROOT / "processing" / "extract_signals.py")]),
        ("🔗 Cross-referencing people",[sys.executable, str(ROOT / "processing" / "cross_reference.py")]),
        ("📝 Generating brief",        [sys.executable, str(ROOT / "processing" / "generate_brief.py")]),
    ]

    placeholder = st.empty()
    log_lines: list[str] = []

    def _update(line: str):
        log_lines.append(line)
        placeholder.code("\n".join(log_lines[-60:]), language="text")  # keep last 60 lines

    for label, cmd in steps:
        _update(f"\n{'─'*50}")
        _update(f"  {label} ...")
        _update(f"{'─'*50}")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(ROOT),
            )
            for line in proc.stdout:
                _update(line.rstrip())
            proc.wait()
            if proc.returncode != 0:
                _update(f"⚠  Step exited with code {proc.returncode}")
        except Exception as exc:
            _update(f"✗  Error running {cmd[1]}: {exc}")

    _update("\n✓  Pipeline complete. Refresh the page to see updated data.")
    st.cache_resource.clear()   # bust the DB connection cache so new data shows immediately


# ---------------------------------------------------------------------------
# Tab 1 — Weekly Brief
# ---------------------------------------------------------------------------

def tab_weekly_brief():
    st.header("Weekly Ecosystem Brief")

    conn = db()
    briefs = conn.execute(
        "SELECT id, brief_date, week_label, file_path, signal_count, move_count, companies_covered, generated_at "
        "FROM weekly_briefs ORDER BY brief_date DESC"
    ).fetchall()

    if not briefs:
        # Fall back to file-system scan for briefs generated before the table existed
        brief_files = sorted(BRIEFS_DIR.glob("brief_*.md"), reverse=True)
        if not brief_files:
            st.info(
                "No briefs generated yet.  \n"
                "Run the pipeline from the sidebar, or:  \n"
                "```\npython processing/generate_brief.py\n```"
            )
            return

        selected_file = st.selectbox(
            "Select brief",
            brief_files,
            format_func=lambda p: p.stem.replace("brief_", ""),
        )
        text = selected_file.read_text(encoding="utf-8")
        _render_brief(text, selected_file.name, {})
        return

    # Normal path — briefs in DB; convert to dict so .get() is safe everywhere
    options = {f"{b['week_label']} ({b['brief_date']})": dict(b) for b in briefs}
    choice  = st.selectbox("Select week", list(options.keys()))
    brief   = options[choice]

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Signals",    brief["signal_count"] or "—")
    col2.metric("Exec moves", brief["move_count"] or "—")
    col3.metric("Companies",  brief["companies_covered"] or "—")
    col4.metric("Generated",  (brief["generated_at"] or "")[:16])

    # Fetch body
    body = brief.get("body") or ""
    if not body and brief["file_path"] and Path(brief["file_path"]).exists():
        body = Path(brief["file_path"]).read_text(encoding="utf-8")

    if not body:
        st.warning("Brief body not found.")
        return

    _render_brief(body, f"brief_{brief['brief_date']}.md", brief)


def _render_brief(body: str, filename: str, meta: dict):
    st.divider()
    st.markdown(body)
    st.divider()
    st.download_button(
        label     = "⬇  Download as Markdown",
        data      = body,
        file_name = filename,
        mime      = "text/markdown",
        use_container_width = False,
    )


# ---------------------------------------------------------------------------
# Tab 2 — Company Detail
# ---------------------------------------------------------------------------

def tab_company_detail():
    st.header("Company Detail")
    conn = db()

    companies = conn.execute("SELECT id, name, sector, stage FROM companies ORDER BY name").fetchall()
    if not companies:
        st.info("No companies in the database. Run `python ingestion/seed_companies.py`.")
        return

    col_sel, col_days = st.columns([3, 1])
    with col_sel:
        company_names = [c["name"] for c in companies]
        selected_name = st.selectbox("Select company", company_names)
    with col_days:
        days = st.number_input("Days back", min_value=7, max_value=365, value=30, step=7)

    company = next(c for c in companies if c["name"] == selected_name)
    since   = _since(days)

    # --- Company header ---
    meta = dict(conn.execute("SELECT * FROM companies WHERE id = ?", (company["id"],)).fetchone())
    ticker_str = f" · {meta['ticker']}" if meta.get("ticker") else ""
    st.subheader(f"{meta['name']}{ticker_str}")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Sector:** {meta['sector'] or '—'}")
    c2.markdown(f"**Stage:** {meta['stage'] or '—'}")
    c3.markdown(f"**HQ:** {meta['hq'] or '—'}")
    if meta.get("notes"):
        st.caption(meta["notes"])
    st.divider()

    # --- Signals ---
    st.subheader("Extracted Signals")
    signals = conn.execute(
        """
        SELECT s.id, s.signal_type, s.summary, s.people_mentioned,
               s.relevance_score, s.extracted_at,
               ni.title AS article_title, ni.source, ni.url, ni.published_at
        FROM extracted_signals s
        JOIN news_items ni ON ni.id = s.news_item_id
        WHERE s.company_id = ? AND s.extracted_at >= ?
        ORDER BY s.relevance_score DESC, s.extracted_at DESC
        """,
        (company["id"], since),
    ).fetchall()

    if not signals:
        st.info(f"No signals extracted for {selected_name} in the last {days} days.")
    else:
        min_score = st.slider("Minimum relevance score", 1, 5, 1, key="company_score_filter")
        filtered  = [s for s in signals if (s["relevance_score"] or 0) >= min_score]
        st.caption(f"{len(filtered)} signal(s) shown")

        rows = []
        for s in filtered:
            rows.append({
                "Score":    _score_stars(s["relevance_score"]),
                "Type":     _label(s["signal_type"] or "other"),
                "Summary":  s["summary"] or "",
                "People":   _people_display(s["people_mentioned"]),
                "Source":   s["source"] or "",
                "Date":     (s["published_at"] or "")[:10],
                "URL":      s["url"] or "",
            })

        df = pd.DataFrame(rows)

        # Render as a styled table with clickable URLs
        for _, row in df.iterrows():
            with st.container():
                head_col, score_col = st.columns([6, 1])
                with head_col:
                    url = row["URL"]
                    title_link = f"[{row['Source']}]({url})" if url else row["Source"]
                    st.markdown(f"**{row['Type']}** · {row['Date']} · {title_link}")
                with score_col:
                    st.markdown(f"`{row['Score']}`")
                st.write(row["Summary"])
                if row["People"]:
                    st.caption(f"👥 {row['People']}")
                st.divider()

    # --- Recent news ---
    with st.expander(f"📰 Raw News Items (last {days} days)", expanded=False):
        news = conn.execute(
            """
            SELECT title, source, url, published_at, processed_at
            FROM news_items
            WHERE company_id = ? AND fetched_at >= ?
            ORDER BY published_at DESC
            LIMIT 50
            """,
            (company["id"], since),
        ).fetchall()

        if not news:
            st.info("No news articles fetched for this period.")
        else:
            news_df = pd.DataFrame([dict(n) for n in news])
            news_df["processed"] = news_df["processed_at"].apply(lambda x: "✓" if x else "⏳")
            news_df = news_df.rename(columns={
                "title": "Title", "source": "Source",
                "published_at": "Published", "processed": "Processed",
            })
            display_cols = ["Title", "Source", "Published", "Processed"]
            st.dataframe(news_df[display_cols], use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3 — Ecosystem Map
# ---------------------------------------------------------------------------

def tab_ecosystem_map():
    st.header("Ecosystem Map")
    conn = db()

    # --- Moves table ---
    st.subheader("Executive Moves")

    col_filt, col_conf = st.columns([2, 2])
    with col_filt:
        show_source = st.multiselect(
            "Source",
            ["cross_ref", "manual", "edgar"],
            default=["cross_ref", "manual", "edgar"],
        )
    with col_conf:
        conf_filter = st.multiselect(
            "Confidence (cross-ref moves)",
            ["HIGH", "MEDIUM", "LOW", "all"],
            default=["all"],
        )

    source_clause = "AND em.detected_by IN ({})".format(
        ",".join("?" * len(show_source))
    ) if show_source else "AND 1=0"

    moves = conn.execute(
        f"""
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
            cf.name AS from_name,
            ct.name AS to_name,
            cd.name AS company_name
        FROM executive_moves em
        LEFT JOIN companies cf ON cf.id = em.from_company
        LEFT JOIN companies ct ON ct.id = em.to_company
        LEFT JOIN companies cd ON cd.id = em.company_id
        WHERE 1=1 {source_clause}
        ORDER BY COALESCE(em.move_date, em.recorded_at) DESC
        """,
        show_source,
    ).fetchall()

    if not moves:
        st.info("No executive moves recorded yet. Run the pipeline to populate.")
    else:
        rows = []
        for m in moves:
            # For cross-ref bridges, from/to are company IDs; resolve display names
            from_display = m["from_name"] or m["from_company"] or m["company_name"] or "—"
            to_display   = m["to_name"]   or m["to_company"]   or m["company_name"] or "—"
            date_display = (m["move_date"] or m["recorded_at"] or "")[:10]
            move_label   = (m["move_type"] or "").replace("_", " ").replace("ecosystem bridge ", "")

            rows.append({
                "Person":      m["person_name"] or "—",
                "From":        from_display,
                "To":          to_display,
                "Old role":    m["old_role"] or "—",
                "New role":    m["new_role"] or "—",
                "Type":        move_label,
                "Date":        date_display,
                "Source":      m["detected_by"] or "manual",
                "Confidence":  _extract_confidence(m["confidence_note"]),
                "_note":       m["confidence_note"] or "",
            })

        df = pd.DataFrame(rows)

        # Apply confidence filter (only meaningful for cross_ref rows)
        if "all" not in conf_filter:
            df = df[
                (df["Source"] != "cross_ref") |
                (df["Confidence"].isin(conf_filter))
            ]

        display_df = df.drop(columns=["_note", "Confidence"] if "all" in conf_filter else ["_note"])
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption(f"{len(display_df)} move(s) shown")

        # Detailed expander for cross-ref notes
        cross_ref_rows = [r for r in rows if r["Source"] == "cross_ref" and r["_note"]]
        if cross_ref_rows:
            with st.expander("🔍 Cross-reference confidence notes"):
                for r in cross_ref_rows:
                    st.markdown(f"**{r['Person']}** ({r['From']} → {r['To']})")
                    st.caption(r["_note"])
                    st.divider()

    st.divider()

    # --- Network graph ---
    st.subheader("Company Connection Network")
    st.caption("Nodes = portfolio companies · Edges = people who moved between them")

    _render_network(conn)


def _extract_confidence(note: str | None) -> str:
    """Pull HIGH/MEDIUM/LOW out of a confidence_note string."""
    if not note:
        return "—"
    for level in ("HIGH", "MEDIUM", "LOW"):
        if level in note:
            return level
    return "—"


def _render_network(conn):
    """Build and display a pyvis network graph of cross-portfolio moves."""
    try:
        import networkx as nx
        from pyvis.network import Network
    except ImportError:
        st.warning(
            "Install `networkx` and `pyvis` to enable the network graph:  \n"
            "`pip install networkx pyvis`"
        )
        return

    # Only bridge moves have meaningful from/to company data
    bridges = conn.execute(
        """
        SELECT em.person_name, em.old_role, em.new_role, em.confidence_note,
               cf.name AS from_name, ct.name AS to_name,
               cf.sector AS from_sector, ct.sector AS to_sector
        FROM executive_moves em
        JOIN companies cf ON cf.id = em.from_company
        JOIN companies ct ON ct.id = em.to_company
        WHERE em.detected_by = 'cross_ref'
        """
    ).fetchall()

    # Also include all portfolio companies as nodes even without edges
    all_companies = conn.execute(
        "SELECT id, name, sector, stage FROM companies"
    ).fetchall()

    if not bridges:
        st.info(
            "No cross-portfolio bridges detected yet.  \n"
            "Run the full pipeline to populate the Ecosystem Movement data."
        )
        # Still show all companies as isolated nodes so the graph isn't empty
        if not all_companies:
            return

    G = nx.DiGraph()

    # Add all portfolio companies as nodes
    sector_colors = {
        "Cloud Data Platform":           "#4e79a7",
        "Design Software":               "#f28e2b",
        "Observability & Monitoring":    "#e15759",
        "DevOps / Software Development": "#76b7b2",
        "Vertical SaaS / Field Service": "#59a14f",
        "Construction Management Software": "#edc948",
        "Productivity / No-Code Platform": "#b07aa1",
        "Collaboration / Visual Workspace": "#ff9da7",
        "Customer Experience Management":  "#9c755f",
        "Payments":                        "#bab0ac",
        "Data Intelligence / Governance":  "#d37295",
    }
    default_color = "#aec7e8"

    for c in all_companies:
        G.add_node(
            c["name"],
            title   = f"{c['name']}\n{c['sector']}\n{c['stage']}",
            color   = sector_colors.get(c["sector"], default_color),
            size    = 25,
            font    = {"size": 14, "color": "#ffffff"},
        )

    # Add edges for bridges
    edge_labels: dict[tuple[str, str], list[str]] = {}
    for b in bridges:
        fn, tn = b["from_name"], b["to_name"]
        if not fn or not tn or fn == tn:
            continue
        G.add_edge(fn, tn, weight=1)
        key = (fn, tn)
        edge_labels.setdefault(key, []).append(b["person_name"])

    # Build pyvis network
    net = Network(
        height    = "520px",
        width     = "100%",
        bgcolor   = "#1e1e2e",
        font_color= "#ffffff",
        directed  = True,
        notebook  = False,
    )
    net.from_nx(G)

    # Annotate edge titles with people names
    for edge in net.edges:
        key = (edge["from"], edge["to"])
        people = edge_labels.get(key, [])
        edge["title"] = "People: " + ", ".join(people) if people else ""
        edge["color"] = "#f0a500"
        edge["width"] = 2 + len(people)

    # Physics options for a readable layout
    net.set_options("""{
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 180
        },
        "minVelocity": 0.75
      },
      "edges": {
        "smooth": { "type": "curvedCW", "roundness": 0.2 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      }
    }""")

    # Write to a temp HTML file and embed via st.components
    import tempfile
    html_path = Path(tempfile.gettempdir()) / "ecosystem_graph.html"
    net.save_graph(str(html_path))
    html_content = html_path.read_text(encoding="utf-8")

    import streamlit.components.v1 as components
    components.html(html_content, height=540, scrolling=False)

    # Legend
    if bridges:
        st.caption(
            f"{G.number_of_nodes()} companies · "
            f"{G.number_of_edges()} connection(s) · "
            f"Arrows show direction of move · Edge width = number of people"
        )
        st.caption("Hover over a node or edge for details. Drag nodes to rearrange.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    render_sidebar()

    tab1, tab2, tab3 = st.tabs([
        "📄 Weekly Brief",
        "🏢 Company Detail",
        "🕸 Ecosystem Map",
    ])

    with tab1:
        tab_weekly_brief()

    with tab2:
        tab_company_detail()

    with tab3:
        tab_ecosystem_map()


if __name__ == "__main__":
    main()
