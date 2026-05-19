# LANG Quiz Dashboard

Streamlit dashboard for Canvas LANG 1406 quiz CSV exports. Upload a file in the app; data is scrubbed for PII in memory only (nothing is written to disk).

## Features

- Overview: score distribution, section comparison, submission timeline
- Question analysis: answer distributions, correctness, hover tooltips
- Open-ended responses: word frequency, topic themes, browse by student
- Student detail: anonymous IDs, filterable views
- Section filters with select all / unselect all

## Setup

```bash
cd streamlit_app
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open http://localhost:8501 and upload a Canvas quiz CSV (latin1 encoding).

## Privacy

- Identifiers are removed on upload; rows are shuffled and labelled `Student 1`, `Student 2`, …
- Do not commit real quiz CSV files to this repository

## Deploy (Railway / similar)

The repo includes:

- `streamlit_app/.streamlit/config.toml` — headless server defaults for PaaS
- `streamlit_app/railpack.json` — Railpack `deploy.startCommand` (uses `$PORT`)
- `railway.json` — Railway builder (`RAILPACK`) and health check settings

**Important:** In Railway, set the service **Root Directory** to `streamlit_app` so Railpack finds `railpack.json`, `requirements.txt`, and `app.py` in the same folder.

The start command in `railpack.json` is:

```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false
```

## Repository

https://github.com/fantasy-library/LANG_Quiz
