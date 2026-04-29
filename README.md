# Acron

A small Flask app for indexing GitHub repos and asking questions about them. It walks a repo with [Kit](https://pypi.org/project/cased-kit/), extracts symbols (functions, classes, methods, types), embeds them with `jinaai/jina-embeddings-v2-base-code`, stores everything in SQLite with [`sqlite-vec`](https://github.com/asg017/sqlite-vec), and answers questions over the indexed code via OpenRouter.

## Pipeline

```
GitHub URL  →  Kit  →  SQLite (symbols)  →  sqlite-vec (embeddings)  →  similarity search  →  LLM answer
```

## Requirements

- Python 3.11+
- An [OpenRouter](https://openrouter.ai) API key (for chat)
- Optional: a GitHub token for higher rate limits when indexing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in keys, or set them via the Settings page
```

`requirements.txt` covers the Flask app; the pipeline also uses `sqlite-vec` and `sentence-transformers`, which come in via `cased-kit`'s deps.

## Run

```bash
python app.py
# open http://localhost:5001
```

From the homepage:
1. Paste a `https://github.com/...` URL to index a repo.
2. Click into the repo to browse files/symbols and run vector search.
3. Use the chat box to ask questions — answers cite `file:line` and are saved as reports.

## Files

- `app.py` — Flask routes (home, repo detail, chat, reports, settings).
- `pipeline.py` — indexing + embedding + search; also runnable standalone as a demo.
- `schema.sql` — base relational schema (vector table is created at runtime).
- `templates/` — Jinja templates.
- `prototype.py`, `prototype_vec.py`, `prototype/` — earlier scratch versions kept for reference.

## Notes

- `pipeline.db` and `acron.db` are SQLite files written by the app. They're gitignored.
- API keys can live in `.env` or be set via the in-app Settings page (stored in the DB).
