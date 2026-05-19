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

Open http://localhost:8501 and upload a Canvas quiz CSV (Windows/cp1252 or UTF-8).

## Privacy

- Identifiers are removed on upload; rows are shuffled and labelled `Student 1`, `Student 2`, …
- Do not commit real quiz CSV files to this repository

## Deploy (Railway / similar)

The repo includes:

- `requirements.txt` + `railpack.json` at the **repo root** (default Railway setup)
- `streamlit_app/.streamlit/config.toml` — headless server defaults for PaaS
- `streamlit_app/railpack.json` — alternate config if Root Directory is `streamlit_app`
- `railway.json` — Railway builder (`RAILPACK`) and health check only

### Railway settings

1. **Builder:** Railpack (from `railway.json`)
2. **Root Directory:** leave empty (repo root) — recommended
3. Do **not** set a custom build command like `pip install ...` in the Railway UI; Railpack installs Python and dependencies automatically.

If build fails with `pip: not found`, clear any manual **Build Command** in Railway service settings and redeploy.

### Alternative: Root Directory = `streamlit_app`

Use `streamlit_app/railpack.json` and this start command:

```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false
```

## Repository

https://github.com/fantasy-library/LANG_Quiz
