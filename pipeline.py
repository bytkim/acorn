"""
Acron — bare-bones end-to-end pipeline.

Walks through every stage the real app will perform, in order, with
explanations. Run this file top-to-bottom and you have:

    GitHub URL  ->  Kit  ->  SQLite (relational)  ->  sqlite-vec (vectors)
                                                  ->  similarity search

This file is for understanding the pipeline, NOT the real app. The real app
splits these stages across schema.sql / db.py / app.py.

Run:  .venv/bin/python pipeline.py
"""

import sqlite3
import struct
import datetime
from pathlib import Path

import sqlite_vec
from kit import Repository
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Stage 0 — config
# ---------------------------------------------------------------------------
# DB lives in a single file alongside this script. The grader sees one .db.
DB_PATH = Path(__file__).parent / "pipeline.db"

# 384-dim embedding model. Small (~80 MB), fast on CPU, decent quality for
# code search. Vector dimension MUST match what we declare in the vec0 table.
EMBED_MODEL = "jinaai/jina-embeddings-v2-base-code"
EMBED_DIM = 768

# A small public repo so the pipeline finishes in seconds.
DEMO_REPO = "https://github.com/sindresorhus/slugify"


# ---------------------------------------------------------------------------
# Stage 1 — open a SQLite connection with sqlite-vec loaded
# ---------------------------------------------------------------------------
# sqlite-vec is a C extension. It has to be loaded into each connection
# BEFORE you can reference vec0 virtual tables or call vec_*() functions.
# Loading happens in three steps: enable -> load -> disable. Disabling after
# loading is a defensive habit (prevents arbitrary later extension loads).
def open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # Foreign keys are OFF by default in SQLite (legacy compat). Turn them on
    # per-connection so ON DELETE CASCADE actually fires.
    conn.execute("PRAGMA foreign_keys = ON")
    # Row factory so templates can use r['col'] instead of positional indexing.
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Stage 2 — schema
# ---------------------------------------------------------------------------
# Four relational tables (the rubric ones) plus one virtual table for
# vectors. The vector table is keyed on symbol_id so vectors join cleanly
# back to the relational symbol rows — that JOIN is the whole point.
#
# Note: vec0 virtual tables do NOT enforce foreign keys to regular tables.
# If you DELETE a symbol, its vector row is NOT auto-removed. You have to
# delete from symbol_vectors explicitly (or do it inside the same function
# that handles symbol deletion).
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS tags (
    tag_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_name  TEXT NOT NULL UNIQUE,
    color_hex TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS repositories (
    repo_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    github_url TEXT NOT NULL UNIQUE,
    repo_name  TEXT NOT NULL,
    owner      TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    file_count INTEGER NOT NULL
);

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

CREATE TABLE IF NOT EXISTS reports (
    report_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id     INTEGER NOT NULL REFERENCES repositories(repo_id) ON DELETE CASCADE,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    sources     TEXT NOT NULL,
    model       TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (
    note_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_id  INTEGER NOT NULL REFERENCES symbols(symbol_id) ON DELETE CASCADE,
    note_text  TEXT NOT NULL,
    tag_id     INTEGER NOT NULL REFERENCES tags(tag_id),
    created_at TEXT NOT NULL
);

-- Virtual table provided by sqlite-vec. Stores fixed-width float vectors
-- with a fast nearest-neighbor index. Queried with `MATCH ? AND k = N`.
CREATE VIRTUAL TABLE IF NOT EXISTS symbol_vectors USING vec0(
    symbol_id INTEGER PRIMARY KEY,
    embedding FLOAT[{EMBED_DIM}]
);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    # Seed tags so notes always have something to reference. INSERT OR IGNORE
    # makes this idempotent — running pipeline.py twice won't duplicate them.
    conn.executemany(
        "INSERT OR IGNORE INTO tags (tag_name, color_hex) VALUES (?, ?)",
        [("todo", "#f59e0b"), ("bug", "#ef4444"),
         ("insight", "#10b981"), ("question", "#3b82f6")],
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Stage 3 — index a repo with cased-kit
# ---------------------------------------------------------------------------
# Kit's job: clone the repo (cached in /tmp), walk its file tree, and run
# tree-sitter to extract symbol metadata (name, type, line range, code).
#
# Two methods we use:
#   repo.get_file_tree()        -> list of {path, is_dir, ...}
#   repo.extract_symbols(path)  -> list of {name, type, start_line,
#                                            end_line, code, file}
#
# Kit's line numbers are 0-indexed. We pass them through as-is here; the
# UI layer can +1 when displaying.
def get_setting(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_setting(conn, key: str, value: str) -> None:
    if value:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
    else:
        conn.execute("DELETE FROM settings WHERE key=?", (key,))
    conn.commit()


def index_repo(github_url: str, github_token: str | None = None) -> tuple[list, list]:
    repo = Repository(github_url, github_token=github_token) if github_token else Repository(github_url)
    file_tree = repo.get_file_tree()
    files = [f for f in file_tree if not f.get("is_dir")]

    symbols = []
    for f in files:
        try:
            for sym in repo.extract_symbols(f["path"]):
                symbols.append((f["path"], sym))
        except Exception:
            # Non-source files (markdown, json, lockfiles) raise or return
            # nothing. Skip silently — they're not symbols.
            pass
    return files, symbols


# ---------------------------------------------------------------------------
# Stage 4 — bulk-insert symbols into the relational table
# ---------------------------------------------------------------------------
# One INSERT per repo, one bulk executemany() per repo for symbols.
# Returns the new repo_id and the list of (symbol_id, code_snippet) pairs
# we'll need for embedding in stage 5.
def insert_repo(conn, github_url, files, symbols) -> list[tuple[int, str]]:
    owner, repo_name = github_url.rstrip("/").split("/")[-2:]
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO repositories (github_url, repo_name, owner, indexed_at, file_count) "
        "VALUES (?, ?, ?, ?, ?)",
        (github_url, repo_name, owner,
         datetime.datetime.now(datetime.UTC).isoformat(), len(files)),
    )
    repo_id = cur.lastrowid

    # Build rows for executemany. Order of columns must match the INSERT.
    rows = [
        (repo_id, path,
         sym.get("name", "?"), sym.get("type", "?"),
         sym.get("start_line", 0) or 0, sym.get("end_line", 0) or 0,
         sym.get("code", "") or "")
        for path, sym in symbols
    ]
    cur.executemany(
        "INSERT INTO symbols "
        "(repo_id, file_path, symbol_name, symbol_type, start_line, end_line, code_snippet) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()

    # Pull back the auto-generated symbol_ids paired with their snippets so
    # we can embed in stage 5. lastrowid only gives us the LAST inserted id,
    # so we re-query.
    pairs = cur.execute(
        "SELECT symbol_id, code_snippet FROM symbols WHERE repo_id = ?",
        (repo_id,),
    ).fetchall()
    return pairs


# ---------------------------------------------------------------------------
# Stage 5 — embed snippets and store vectors
# ---------------------------------------------------------------------------
# sqlite-vec stores vectors as raw little-endian float32 blobs. We pack the
# Python list with struct.pack(). The vec0 table validates length matches
# the FLOAT[384] declaration and rejects mismatches.
#
# normalize_embeddings=True puts vectors on the unit sphere. With
# normalized vectors, L2 distance and cosine distance rank identically,
# so the default L2 metric in vec0 gives you cosine-equivalent ranking.
def to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def embed_and_store(conn, model, pairs: list[tuple[int, str]]) -> None:
    if not pairs:
        return
    snippets = [code for _, code in pairs]
    embeddings = model.encode(
        snippets, normalize_embeddings=True, show_progress_bar=False
    )

    conn.executemany(
        "INSERT INTO symbol_vectors (symbol_id, embedding) VALUES (?, ?)",
        [(pairs[i][0], to_blob(embeddings[i].tolist()))
         for i in range(len(pairs))],
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Stage 6 — similarity search
# ---------------------------------------------------------------------------
# The query: embed the user's question with the SAME model used at index
# time, then MATCH against the vec0 index.
#
# Two sqlite-vec gotchas baked into this query:
#   1. Use `AND k = N` to limit results, NOT `LIMIT N`. LIMIT alone returns
#      nothing because vec0 needs k to know how many neighbors to compute.
#   2. The JOIN to `symbols` is what makes this rubric-friendly: vector
#      search returns symbol_ids and distances, the JOIN turns them into
#      human-readable rows. This is real relational + vector hybrid SQL.
def search(conn, model, question: str, k: int = 5, repo_id: int | None = None):
    qvec = model.encode([question], normalize_embeddings=True)[0].tolist()
    # Pull more from vec0 than k when scoping to a repo, then filter in SQL.
    # vec0 itself doesn't filter by joined columns, so we over-fetch and
    # let the JOIN + WHERE drop non-matching rows. ~5x is a safe over-fetch
    # for repos that share the DB.
    fetch_k = k * 5 if repo_id else k
    if repo_id is None:
        return conn.execute(
            """
            SELECT s.symbol_id, s.symbol_name, s.symbol_type,
                   s.file_path, s.start_line, s.end_line,
                   s.code_snippet, v.distance
            FROM symbol_vectors v
            JOIN symbols s ON s.symbol_id = v.symbol_id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (to_blob(qvec), fetch_k),
        ).fetchall()
    return conn.execute(
        """
        SELECT s.symbol_id, s.symbol_name, s.symbol_type,
               s.file_path, s.start_line, s.end_line,
               s.code_snippet, v.distance
        FROM symbol_vectors v
        JOIN symbols s ON s.symbol_id = v.symbol_id
        WHERE v.embedding MATCH ? AND k = ? AND s.repo_id = ?
        ORDER BY v.distance
        LIMIT ?
        """,
        (to_blob(qvec), fetch_k, repo_id, k),
    ).fetchall()


# ---------------------------------------------------------------------------
# Drive the whole pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Fresh DB each run so the demo is reproducible.
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = open_db()
    init_schema(conn)
    print(f"[1] schema initialized at {DB_PATH}")

    files, symbols = index_repo(DEMO_REPO)
    print(f"[2] kit: {len(files)} files, {len(symbols)} symbols")

    pairs = insert_repo(conn, DEMO_REPO, files, symbols)
    print(f"[3] inserted repo + {len(pairs)} symbols into SQLite")

    # Loading the model is the slow step (~3s first run, cached after).
    # Done once and reused for both indexing and querying.
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    model.max_seq_length = 1024
    embed_and_store(conn, model, pairs)
    print(f"[4] embedded + stored {len(pairs)} vectors in symbol_vectors")

    print("\n[5] similarity search")
    for q in ["convert a string into a URL-friendly slug",
              "build a regex character class"]:
        print(f"\n  Q: {q}")
        for name, path, line, dist in search(conn, model, q, k=3):
            print(f"     {dist:.3f}  {path}:{line}  {name}")

    conn.close()
