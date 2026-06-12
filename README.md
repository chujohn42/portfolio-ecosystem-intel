# Portfolio Ecosystem Intelligence Tool

An AI-powered portfolio monitoring system for VC firms. Tracks news, SEC filings, and leadership changes across portfolio companies, extracts structured signals using Claude, detects when people move between companies, and generates a weekly brief in the style of an internal VC memo.

> **Live demo:** [portfolio-ecosystem-intel.streamlit.app](https://portfolio-ecosystem-intel.streamlit.app) *(deploy your own below)*

---

## Screenshot

![Dashboard screenshot](docs/screenshot-placeholder.png)

*Weekly Brief tab showing rendered markdown memo with per-company highlights and ecosystem movement section.*

---

## Architecture

```
portfolio-intel/
│
├── data/
│   ├── companies.json        ← Seed list of portfolio companies
│   ├── init_db.py            ← SQLite schema + migration runner
│   └── portfolio.db          ← Created at runtime (gitignored)
│
├── ingestion/
│   ├── seed_companies.py     ← Load companies.json → SQLite
│   ├── fetch_news.py         ← NewsAPI → news_items table
│   └── fetch_edgar.py        ← SEC EDGAR 8-K filings → edgar_filings table
│
├── processing/
│   ├── extract_signals.py    ← Claude (Sonnet 4.6) → extracted_signals table
│   ├── cross_reference.py    ← Name-match people across companies → executive_moves
│   └── generate_brief.py     ← Claude (Opus 4.8) → weekly_briefs table + .md file
│
├── briefs/                   ← Generated Markdown briefs (committed to git)
│
├── app/
│   └── dashboard.py          ← Streamlit app (3 tabs + pipeline sidebar)
│
├── .streamlit/
│   ├── config.toml           ← Theme + server settings
│   └── secrets.toml.example  ← Template for API keys
│
├── requirements.txt
└── .env.example
```

### Data flow

```
NewsAPI ──────────────────────┐
                              ▼
SEC EDGAR ──────────► news_items / edgar_filings
                              │
                    extract_signals.py
                    (Claude Sonnet 4.6)
                              │
                              ▼
                     extracted_signals
                              │
                    cross_reference.py
                    (name matching)
                              │
                              ▼
                     executive_moves
                              │
                    generate_brief.py
                    (Claude Opus 4.8)
                              │
                        ┌─────┴──────┐
                        ▼            ▼
                  weekly_briefs   briefs/*.md
                        │
                   dashboard.py
                   (Streamlit)
```

### Database schema

| Table | Purpose |
|---|---|
| `companies` | Portfolio company registry (name, sector, stage, ticker, notes) |
| `news_items` | Raw articles from NewsAPI; `processed_at` tracks extraction status |
| `extracted_signals` | Structured signals from Claude: type, summary, people, relevance 1-5 |
| `executive_moves` | Leadership changes from EDGAR + cross-portfolio bridges |
| `edgar_filings` | SEC 8-K Item 5.02 filings with snippet and exec_signal classification |
| `weekly_briefs` | Generated briefs stored in DB alongside the .md file |

---

## Quickstart (local)

### 1. Clone and install

```bash
git clone https://github.com/chujohn42/portfolio-ecosystem-intel.git
cd portfolio-ecosystem-intel
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
NEWSAPI_KEY=...
```

Get your keys:
- **Anthropic:** https://console.anthropic.com
- **NewsAPI:** https://newsapi.org (free tier: 100 req/day)
- **SEC EDGAR:** No key required — just respect the rate limit

### 3. Initialize the database

```bash
python data/init_db.py
python ingestion/seed_companies.py
```

### 4. Run the pipeline

```bash
# Fetch news for all portfolio companies (last 7 days)
python ingestion/fetch_news.py

# Fetch SEC EDGAR 8-K filings for public companies (last 30 days)
python ingestion/fetch_edgar.py

# Extract signals with Claude
python processing/extract_signals.py

# Detect cross-portfolio people movement
python processing/cross_reference.py

# Generate the weekly brief
python processing/generate_brief.py
```

### 5. Launch the dashboard

```bash
streamlit run app/dashboard.py
```

Open http://localhost:8501

---

## Deploying to Streamlit Cloud

### Prerequisites

- This repo is pushed to GitHub as `chujohn42/portfolio-ecosystem-intel`
- You have a [Streamlit Cloud](https://streamlit.io/cloud) account

### Steps

1. In Streamlit Cloud, click **New app**
2. Select repository: `chujohn42/portfolio-ecosystem-intel`
3. Set main file path: `app/dashboard.py`
4. Under **Advanced settings → Secrets**, add:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
NEWSAPI_KEY = "your_newsapi_key"
```

5. Click **Deploy**

> **Note on the database:** Streamlit Cloud has an ephemeral filesystem — `portfolio.db` is recreated on each restart. The app handles this gracefully (schema is created fresh, companies are seeded from `companies.json`). Use the **Run Full Pipeline Now** sidebar button to populate data after a restart.
>
> For a persistent database across restarts, consider adding [Supabase](https://supabase.com) or [PlanetScale](https://planetscale.com) as the backend — open an issue if you'd like that integration.

---

## Pipeline reference

### `ingestion/fetch_news.py`

Fetches the last 7 days of news for each portfolio company via NewsAPI.

```
--days N          Lookback window (default: 7)
--company ID      Single company by id (e.g. snowflake)
--dry-run         Print articles without writing to DB
--debug           Verbose logging (shows queries, timing)
```

Rate limits: ≥1.2s between calls, retries on 429 with Retry-After.

### `ingestion/fetch_edgar.py`

Searches SEC EDGAR for 8-K Item 5.02 filings (executive changes) for all public portfolio companies.

```
--days N          Lookback window (default: 30)
--ticker TICKER   Single ticker (e.g. SNOW)
--dry-run
--debug
```

No API key required. Rate limit: ≤10 req/s (enforced at 0.15s intervals).

### `processing/extract_signals.py`

Sends each unprocessed article to **Claude Sonnet 4.6** to extract:
- `signal_type`: leadership_change | funding_partnership | notable_hire | product_launch | other
- `summary`: 1-2 sentences
- `people_mentioned`: list of `{name, role, company}`
- `relevance_score`: 1-5

```
--limit N         Articles per run (default: 50)
--company ID      Single company
--dry-run
--debug
```

Retry policy: rate limits → wait and retry (3×), timeouts → retry (3×), 4xx → fail fast.

### `processing/cross_reference.py`

Scans `extracted_signals` for `leadership_change` entries and finds people who appear in signals for two or more portfolio companies. Confidence levels:

| Level | Condition |
|---|---|
| HIGH | Exact name match, signals ≤180 days apart |
| MEDIUM | Exact match with wide gap, or fuzzy match with relevance ≥4 |
| LOW | Partial token match, review manually |

```
--days N          Signal window (default: 365)
--min-score N     Minimum relevance_score (default: 2)
--dry-run
--debug           Shows every name comparison
```

### `processing/generate_brief.py`

Generates a weekly memo using **Claude Opus 4.8** (adaptive thinking, streaming). Sections: Executive Summary · Per-Company Highlights · Ecosystem Movement · People to Watch.

```
--days N          Lookback window (default: 7)
--dry-run         Stream to terminal only, no saves
--no-stream       Silent generation (good for cron)
--rerun           Overwrite today's existing brief
```

---

## Customizing the portfolio

Edit `data/companies.json` with your actual portfolio companies, then re-seed:

```bash
python ingestion/seed_companies.py --replace
```

Each entry supports:

```json
{
  "id":          "unique-slug",
  "name":        "Company Name",
  "domain":      "company.com",
  "sector":      "Cloud Data Platform",
  "stage":       "Series B",
  "hq":          "San Francisco, CA",
  "ticker":      "TICK",
  "description": "One sentence description.",
  "keywords":    ["company name", "ticker", "product name"],
  "notes":       "Investment context / why we're tracking this."
}
```

Set `ticker` to `null` for private companies — they'll be excluded from EDGAR searches.

---

## Automating weekly runs

**cron (Linux/macOS):**

```cron
0 7 * * MON  cd /path/to/portfolio-intel && python ingestion/fetch_news.py && python ingestion/fetch_edgar.py && python processing/extract_signals.py && python processing/cross_reference.py && python processing/generate_brief.py --no-stream
```

**Task Scheduler (Windows):** Create a `.bat` file running the same sequence and schedule it for Monday 7 AM.

---

## Contributing

PRs welcome. Main areas for improvement:

- **Persistent DB** — swap SQLite for Postgres/Supabase for Streamlit Cloud persistence
- **More sources** — LinkedIn RSS, Crunchbase API, PitchBook webhooks
- **Slack integration** — post the weekly brief to a `#portfolio-intel` channel
- **Alert thresholds** — push notification when a relevance ≥4 signal is detected

---

## License

MIT
