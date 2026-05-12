-- Acron — full relational schema.
--
-- Apply with:   sqlite3 data/pipeline.db < sql/create_schema.sql
--
-- This file is the canonical schema for grading / cold-start setup.
-- The Python pipeline (src/pipeline.py) embeds the same DDL so the app can
-- bootstrap a fresh DB on first run; keep the two in sync if you change one.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- tags: small lookup table of note categories with display colors.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tags (
    tag_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_name  TEXT NOT NULL UNIQUE,
    color_hex TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- repositories: one row per indexed GitHub repo.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS repositories (
    repo_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    github_url TEXT NOT NULL UNIQUE,
    repo_name  TEXT NOT NULL,
    owner      TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    file_count INTEGER NOT NULL
);

-- ---------------------------------------------------------------------------
-- indexing_jobs: tracks the lifecycle of a single indexing attempt
-- (queued -> extracting -> embedding -> done/failed).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS indexing_jobs (
    job_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    github_url   TEXT NOT NULL,
    status       TEXT NOT NULL,
    repo_id      INTEGER NOT NULL REFERENCES repositories(repo_id) ON DELETE CASCADE,
    file_count   INTEGER NOT NULL,
    symbol_count INTEGER NOT NULL,
    error        TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_indexing_jobs_status
ON indexing_jobs(status);

CREATE INDEX IF NOT EXISTS idx_indexing_jobs_github_url_status
ON indexing_jobs(github_url, status);

-- ---------------------------------------------------------------------------
-- symbols: every function/class/method extracted from an indexed repo.
-- code_snippet is the raw source text used for display and embedding.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS symbols (
    symbol_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id      INTEGER NOT NULL REFERENCES repositories(repo_id) ON DELETE CASCADE,
    file_path    TEXT NOT NULL,
    symbol_name  TEXT NOT NULL,
    symbol_type  TEXT NOT NULL,
    start_line   INTEGER NOT NULL,
    end_line     INTEGER NOT NULL,
    code_snippet TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- reports: saved Q&A from the in-app chat. sources is JSON-encoded list of
-- file:line citations the model used.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reports (
    report_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id    INTEGER NOT NULL REFERENCES repositories(repo_id) ON DELETE CASCADE,
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL,
    sources    TEXT NOT NULL,
    model      TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- settings: key/value table used by the Settings page for API keys and
-- runtime config. Overrides .env when present.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- notes: user-authored annotations attached to a specific symbol, with a tag.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS notes (
    note_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_id  INTEGER NOT NULL REFERENCES symbols(symbol_id) ON DELETE CASCADE,
    note_text  TEXT NOT NULL,
    tag_id     INTEGER NOT NULL REFERENCES tags(tag_id),
    created_at TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- symbol_vectors: sqlite-vec virtual table holding 768-dim embeddings keyed
-- by symbol_id. Dimension matches the embedding model in src/pipeline.py
-- (jinaai/jina-embeddings-v2-base-code).
--
-- Requires the sqlite-vec extension to be loaded on the connection. The
-- Python app handles this via sqlite_vec.load(conn); when running this
-- file directly with sqlite3, load the extension first:
--     .load /path/to/vec0
-- ---------------------------------------------------------------------------
CREATE VIRTUAL TABLE IF NOT EXISTS symbol_vectors USING vec0(
    symbol_id INTEGER PRIMARY KEY,
    embedding FLOAT[768]
);
