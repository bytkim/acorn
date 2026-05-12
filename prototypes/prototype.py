"""Feasibility spike: Kit -> SQLite."""
import os
import sqlite3
import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(__file__).parent / "acron.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"
TEST_REPOS = [
    "https://github.com/sindresorhus/slugify",
    "https://github.com/sindresorhus/is-odd",
    "https://github.com/pallets/itsdangerous",
]


def init_db():
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()
    conn.close()
    print(f"[db] initialized {DB_PATH}")


def test_kit(github_url: str):
    print(f"\n[kit] indexing {github_url}")
    from kit import Repository

    token = os.getenv("KIT_GITHUB_TOKEN")
    repo = Repository(github_url, github_token=token)

    file_tree = repo.get_file_tree()
    files = [f for f in file_tree if not f.get("is_dir")]
    print(f"[kit] file_tree returned {len(file_tree)} entries ({len(files)} files)")

    symbols = []
    for f in files[:25]:  # cap so spike is fast
        path = f["path"]
        try:
            for sym in repo.extract_symbols(path):
                symbols.append((path, sym))
        except Exception as e:
            print(f"[kit]   skip {path}: {e}")
    print(f"[kit] extracted {len(symbols)} symbols")
    for i, (path, sym) in enumerate(symbols):
        print(f"  [{i}] {path}:{sym.get('start_line')}-{sym.get('end_line')}  {sym.get('type')}  {sym.get('name')}")
        snippet = (sym.get("code") or "").splitlines()
        for line in snippet[:3]:
            print(f"        | {line}")
        if len(snippet) > 3:
            print(f"        | ... ({len(snippet)-3} more lines)")
    return files, symbols


def insert_repo(github_url, files, symbols):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    owner, repo_name = github_url.rstrip("/").split("/")[-2:]
    cur.execute(
        "INSERT INTO repositories (github_url, repo_name, owner, indexed_at, file_count) VALUES (?,?,?,?,?)",
        (github_url, repo_name, owner, datetime.datetime.now(datetime.UTC).isoformat(), len(files)),
    )
    repo_id = cur.lastrowid

    rows = []
    for path, sym in symbols:
        rows.append((
            repo_id,
            path,
            sym.get("name", "?"),
            sym.get("type", "?"),
            sym.get("start_line", 0) or 0,
            sym.get("end_line", 0) or 0,
            sym.get("code", "") or "",
        ))
    cur.executemany(
        "INSERT INTO symbols (repo_id, file_path, symbol_name, symbol_type, start_line, end_line, code_snippet) VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM symbols WHERE repo_id=?", (repo_id,))
    print(f"[db] inserted repo_id={repo_id} with {cur.fetchone()[0]} symbols")
    conn.close()
    return repo_id


def seed_notes(repo_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()
    cur.execute("SELECT symbol_id, symbol_name FROM symbols WHERE repo_id=? LIMIT 2", (repo_id,))
    targets = cur.fetchall()
    now = datetime.datetime.now(datetime.UTC).isoformat()
    for sym_id, name in targets:
        cur.execute(
            "INSERT INTO notes (symbol_id, note_text, tag_id, created_at) VALUES (?,?,?,?)",
            (sym_id, f"auto-seeded note on {name}", 3, now),
        )
    conn.commit()
    conn.close()


def dump_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    print("\n=== repositories ===")
    for row in cur.execute("SELECT repo_id, owner, repo_name, file_count, indexed_at FROM repositories"):
        print(" ", row)
    print("\n=== symbols (first 5 per repo) ===")
    repo_ids = [r[0] for r in cur.execute("SELECT repo_id FROM repositories").fetchall()]
    for repo_id in repo_ids:
        rows = cur.execute(
            "SELECT symbol_id, file_path, symbol_type, symbol_name, start_line, end_line FROM symbols WHERE repo_id=? LIMIT 5",
            (repo_id,),
        ).fetchall()
        for row in rows:
            print(f"  repo={repo_id}", row)
        total = cur.execute("SELECT COUNT(*) FROM symbols WHERE repo_id=?", (repo_id,)).fetchone()[0]
        print(f"  repo={repo_id} total={total}")
    print("\n=== tags ===")
    for row in cur.execute("SELECT * FROM tags"):
        print(" ", row)
    print("\n=== notes (joined) ===")
    for row in cur.execute(
        "SELECT n.note_id, n.note_text, t.tag_name, s.symbol_name, s.file_path "
        "FROM notes n JOIN tags t ON n.tag_id=t.tag_id JOIN symbols s ON n.symbol_id=s.symbol_id"
    ):
        print(" ", row)
    conn.close()


if __name__ == "__main__":
    init_db()
    first_repo_id = None
    for url in TEST_REPOS:
        try:
            files, symbols = test_kit(url)
            rid = insert_repo(url, files, symbols)
            if first_repo_id is None:
                first_repo_id = rid
        except Exception as e:
            print(f"[err] {url}: {e}")
    if first_repo_id is not None:
        seed_notes(first_repo_id)
    dump_db()
    print(f"\n[ok] persisted DB at {DB_PATH}")
