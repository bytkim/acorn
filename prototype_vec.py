"""Spike: sqlite-vec on top of the existing acron.db.

Reads the symbols already in the DB (populated by prototype.py), embeds each
code_snippet with sentence-transformers, stores vectors in a vec0 virtual
table keyed by symbol_id, and runs a sample similarity query.
"""
import sqlite3
import struct
from pathlib import Path

import sqlite_vec
from sentence_transformers import SentenceTransformer

DB_PATH = Path(__file__).parent / "acron.db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-d
DIM = 384


def open_db():
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def to_blob(vec):
    return struct.pack(f"{len(vec)}f", *vec)


def main():
    conn = open_db()
    cur = conn.cursor()

    vec_version = cur.execute("SELECT vec_version()").fetchone()[0]
    print(f"[vec] sqlite-vec loaded, version={vec_version}")

    cur.execute("DROP TABLE IF EXISTS symbol_vectors")
    cur.execute(f"""
        CREATE VIRTUAL TABLE symbol_vectors USING vec0(
            symbol_id INTEGER PRIMARY KEY,
            embedding FLOAT[{DIM}]
        )
    """)
    print("[vec] created virtual table symbol_vectors")

    rows = cur.execute(
        "SELECT symbol_id, symbol_name, code_snippet FROM symbols"
    ).fetchall()
    print(f"[vec] embedding {len(rows)} symbols with {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)
    snippets = [r[2] for r in rows]
    embeddings = model.encode(snippets, normalize_embeddings=True, show_progress_bar=False)

    cur.executemany(
        "INSERT INTO symbol_vectors(symbol_id, embedding) VALUES (?, ?)",
        [(rows[i][0], to_blob(embeddings[i].tolist())) for i in range(len(rows))],
    )
    conn.commit()

    n = cur.execute("SELECT COUNT(*) FROM symbol_vectors").fetchone()[0]
    print(f"[vec] inserted {n} vectors")

    queries = [
        "function that converts a string to a URL slug",
        "verify a cryptographic signature timestamp",
        "raise BadSignature when payload is tampered with",
    ]
    for q in queries:
        qvec = model.encode([q], normalize_embeddings=True)[0].tolist()
        results = cur.execute(
            """
            SELECT s.symbol_name, s.file_path, s.start_line, v.distance
            FROM symbol_vectors v
            JOIN symbols s ON s.symbol_id = v.symbol_id
            WHERE v.embedding MATCH ? AND k = 5
            ORDER BY v.distance
            """,
            (to_blob(qvec),),
        ).fetchall()
        print(f"\n[query] {q!r}")
        for name, path, line, dist in results:
            print(f"  {dist:.4f}  {path}:{line}  {name}")

    conn.close()
    print("\n[ok] sqlite-vec spike complete")


if __name__ == "__main__":
    main()
