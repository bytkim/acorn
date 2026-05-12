PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS tags (
    tag_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_name   TEXT NOT NULL UNIQUE,
    color_hex  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS repositories (
    repo_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    github_url  TEXT NOT NULL UNIQUE,
    repo_name   TEXT NOT NULL,
    owner       TEXT NOT NULL,
    indexed_at  TEXT NOT NULL,
    file_count  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS indexing_jobs (
    job_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    github_url   TEXT NOT NULL,
    status       TEXT NOT NULL,
    repo_id      INTEGER REFERENCES repositories(repo_id) ON DELETE SET NULL,
    file_count   INTEGER,
    symbol_count INTEGER,
    error        TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_indexing_jobs_status
ON indexing_jobs(status);

CREATE INDEX IF NOT EXISTS idx_indexing_jobs_github_url_status
ON indexing_jobs(github_url, status);

CREATE TABLE IF NOT EXISTS symbols (
    symbol_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id       INTEGER NOT NULL REFERENCES repositories(repo_id) ON DELETE CASCADE,
    file_path     TEXT NOT NULL,
    symbol_name   TEXT NOT NULL,
    symbol_type   TEXT NOT NULL,
    start_line    INTEGER NOT NULL,
    end_line      INTEGER NOT NULL,
    code_snippet  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (
    note_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_id   INTEGER NOT NULL REFERENCES symbols(symbol_id) ON DELETE CASCADE,
    note_text   TEXT NOT NULL,
    tag_id      INTEGER NOT NULL REFERENCES tags(tag_id),
    created_at  TEXT NOT NULL
);

INSERT OR IGNORE INTO tags (tag_name, color_hex) VALUES
    ('todo',     '#f59e0b'),
    ('bug',      '#ef4444'),
    ('insight',  '#10b981'),
    ('question', '#3b82f6');
