"""Acron — Flask app. Homepage + modal for indexing a GitHub repo."""
import datetime
import json
import os

from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from pipeline import (
    DB_PATH, EMBED_MODEL, open_db, init_schema,
    index_repo, insert_repo, embed_and_store, search,
    get_setting, set_setting,
)

load_dotenv()

CHAT_MODEL = "stepfun/step-3.5-flash"


def _resolve_key(conn, key: str) -> str:
    """Settings table wins; fall back to env so existing .env users keep working."""
    return (get_setting(conn, key) or os.environ.get(key) or "").strip()


def _openrouter_client(api_key: str) -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def _rewrite_query(api_key: str, question: str, file_paths: list[str], repo) -> str:
    """One-shot query rewrite. Given the file inventory, ask the model to
    produce a denser, code-flavored search string for vector retrieval.
    Returns the original question on any failure."""
    if not file_paths:
        return question
    # Cap file list to keep the prompt small. ~400 paths is plenty signal.
    paths = "\n".join(file_paths[:400])
    if len(file_paths) > 400:
        paths += f"\n... ({len(file_paths) - 400} more)"
    system = (
        "You rewrite a user's question into a dense search string for vector "
        "search over a codebase's symbols (functions, classes, methods, types). "
        "Use terms likely to appear as identifiers or in code comments — file "
        "names, framework primitives, common verbs (handler, register, fetch, "
        "auth, route, middleware). Output the rewritten query ONLY, no quotes, "
        "no preamble."
    )
    user = (
        f"Repo: {repo['owner']}/{repo['repo_name']}\n"
        f"File inventory:\n{paths}\n\n"
        f"User question: {question}\n\n"
        "Rewritten query:"
    )
    try:
        resp = _openrouter_client(api_key).chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=120,
        )
        out = (resp.choices[0].message.content or "").strip().strip('"').strip("'")
        return out or question
    except Exception:
        return question


def _mask(secret: str) -> str:
    if not secret:
        return ""
    return secret[:4] + "…" + secret[-4:] if len(secret) > 10 else "•" * len(secret)


# Language pill: (label, css class) keyed by suffix. .d.ts checked before .ts.
LANG_BY_EXT = [
    (".d.ts", ("TS", "ts")),
    (".tsx",  ("TSX", "react")),
    (".jsx",  ("JSX", "react")),
    (".ts",   ("TS", "ts")),
    (".js",   ("JS", "js")),
    (".py",   ("PY", "python")),
    (".rb",   ("RB", "ruby")),
    (".go",   ("GO", "go")),
    (".rs",   ("RS", "rust")),
    (".java", ("JAVA", "java")),
    (".c",    ("C", "c")),
    (".cpp",  ("C++", "c")),
    (".h",    ("H", "c")),
]


def split_path(p: str):
    if "/" in p:
        d, base = p.rsplit("/", 1)
        return d + "/", base
    return "", p


def lang_for(filename: str):
    low = filename.lower()
    for suf, label in LANG_BY_EXT:
        if low.endswith(suf):
            return label
    return ("FILE", "other")

app = Flask(__name__)
app.secret_key = "dev"  # only for flash messages; replace for prod

# Load the embedding model once at startup. ~3s cold, then cached in RAM.
_model = SentenceTransformer(EMBED_MODEL)

# Make sure the schema exists.
with open_db() as _conn:
    init_schema(_conn)


@app.route("/repo/<int:repo_id>")
def repo_detail(repo_id):
    conn = open_db()
    repo = conn.execute(
        "SELECT * FROM repositories WHERE repo_id = ?", (repo_id,)
    ).fetchone()
    if not repo:
        conn.close()
        return redirect(url_for("home"))

    files = conn.execute(
        """
        SELECT file_path, COUNT(*) AS sym_count,
               MIN(start_line) AS min_line, MAX(end_line) AS max_line
        FROM symbols WHERE repo_id = ?
        GROUP BY file_path ORDER BY file_path
        """,
        (repo_id,),
    ).fetchall()

    type_counts = conn.execute(
        "SELECT symbol_type, COUNT(*) AS n FROM symbols WHERE repo_id=? "
        "GROUP BY symbol_type ORDER BY n DESC",
        (repo_id,),
    ).fetchall()

    symbols = conn.execute(
        """
        SELECT symbol_id, file_path, symbol_type, symbol_name,
               start_line, end_line, code_snippet
        FROM symbols WHERE repo_id = ?
        ORDER BY file_path, start_line
        """,
        (repo_id,),
    ).fetchall()

    has_vectors = conn.execute(
        "SELECT COUNT(*) AS n FROM symbol_vectors v "
        "JOIN symbols s ON s.symbol_id=v.symbol_id WHERE s.repo_id=?",
        (repo_id,),
    ).fetchone()["n"]

    query = (request.args.get("q") or "").strip()
    results = []
    if query and has_vectors:
        results = search(conn, _model, query, k=10, repo_id=repo_id)

    reports = conn.execute(
        "SELECT report_id, question, created_at FROM reports "
        "WHERE repo_id=? ORDER BY created_at DESC LIMIT 20",
        (repo_id,),
    ).fetchall()

    conn.close()

    # groups[file_path][symbol_type] = [symbols sorted by start_line]
    type_order = ["class", "function", "method", "type"]
    groups = {}
    for s in symbols:
        by_type = groups.setdefault(s["file_path"], {})
        by_type.setdefault(s["symbol_type"], []).append(s)

    # Group files under their parent directory; enrich with basename + lang.
    dirs = {}
    for f in files:
        directory, basename = split_path(f["file_path"])
        label, css = lang_for(basename)
        entry = dict(f)  # sqlite3.Row -> plain dict so we can add keys
        entry["basename"] = basename
        entry["lang_label"] = label
        entry["lang_css"] = css
        dirs.setdefault(directory, []).append(entry)
    dir_list = sorted(dirs.items())  # [(dir, [files]), ...] alphabetical

    return render_template(
        "repo.html",
        repo=repo, files=files, type_counts=type_counts,
        symbols=symbols, groups=groups, type_order=type_order,
        dir_list=dir_list, vector_count=has_vectors,
        query=query, results=results, reports=reports,
    )


@app.route("/")
def home():
    conn = open_db()
    repos = conn.execute(
        """
        SELECT r.repo_id, r.owner, r.repo_name, r.github_url,
               r.indexed_at, r.file_count,
               (SELECT COUNT(*) FROM symbols WHERE repo_id = r.repo_id) AS sym_count
        FROM repositories r
        ORDER BY r.indexed_at DESC
        """
    ).fetchall()
    conn.close()
    return render_template("index.html", repos=repos)


@app.route("/index-repo", methods=["POST"])
def index_repo_route():
    url = (request.form.get("github_url") or "").strip()
    if not url.startswith("https://github.com/"):
        flash("Please enter a valid https://github.com/ URL.", "error")
        return redirect(url_for("home"))

    conn = open_db()
    try:
        # Skip if already indexed.
        existing = conn.execute(
            "SELECT repo_id FROM repositories WHERE github_url = ?", (url,)
        ).fetchone()
        if existing:
            flash(f"Repo already indexed.", "warn")
            return redirect(url_for("home"))

        gh_token = _resolve_key(conn, "KIT_GITHUB_TOKEN") or None
        files, symbols = index_repo(url, github_token=gh_token)
        pairs = insert_repo(conn, url, files, symbols)
        embed_and_store(conn, _model, pairs)
        flash(f"Indexed {len(files)} files, {len(pairs)} symbols.", "ok")
    except Exception as e:
        flash(f"Indexing failed: {e}", "error")
    finally:
        conn.close()
    return redirect(url_for("home"))


@app.route("/chat/<int:repo_id>", methods=["POST"])
def chat(repo_id):
    question = (request.json or {}).get("question", "").strip()
    if not question:
        return jsonify(error="empty question"), 400

    conn = open_db()
    try:
        repo = conn.execute(
            "SELECT owner, repo_name FROM repositories WHERE repo_id=?", (repo_id,)
        ).fetchone()
        if not repo:
            return jsonify(error="repo not found"), 404
        api_key = _resolve_key(conn, "OPENROUTER_API_KEY")
        file_paths = [r["file_path"] for r in conn.execute(
            "SELECT DISTINCT file_path FROM symbols WHERE repo_id=? ORDER BY file_path",
            (repo_id,),
        ).fetchall()]
    finally:
        conn.close()

    if not api_key:
        return jsonify(error="No OpenRouter API key configured. Add one in Settings."), 400

    # Step 0 — rewrite the question into a code-search query, given the file
    # inventory. Falls back to the raw question if the rewrite call fails.
    rewritten = _rewrite_query(api_key, question, file_paths, repo)

    conn = open_db()
    try:
        hits = search(conn, _model, rewritten, k=20, repo_id=repo_id)
    finally:
        conn.close()

    context = "\n\n".join(
        f"### {h['file_path']}:{h['start_line']}–{h['end_line']}  "
        f"({h['symbol_type']} {h['symbol_name']})\n{h['code_snippet']}"
        for h in hits
    ) or "(no symbols indexed)"

    system = (
        f"You answer questions about the GitHub repo {repo['owner']}/{repo['repo_name']}. "
        "Use only the code snippets provided as context. Cite file:line for any claim. "
        "If the context doesn't contain the answer, say so."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"

    try:
        resp = _openrouter_client(api_key).chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        return jsonify(error=f"model call failed: {e}"), 502

    sources = [
        {"file_path": h["file_path"], "start_line": h["start_line"],
         "end_line": h["end_line"], "symbol_name": h["symbol_name"],
         "symbol_type": h["symbol_type"], "distance": h["distance"]}
        for h in hits
    ]

    conn = open_db()
    try:
        cur = conn.execute(
            "INSERT INTO reports (repo_id, question, answer, sources, model, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (repo_id, question, answer, json.dumps(sources), CHAT_MODEL,
             datetime.datetime.now(datetime.UTC).isoformat()),
        )
        report_id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()

    return jsonify(
        answer=answer,
        sources=sources,
        rewritten=rewritten if rewritten != question else None,
        report_id=report_id,
        report_url=url_for("report_detail", report_id=report_id),
    )


@app.route("/report/<int:report_id>")
def report_detail(report_id):
    conn = open_db()
    try:
        row = conn.execute(
            "SELECT r.*, p.owner, p.repo_name "
            "FROM reports r JOIN repositories p ON p.repo_id = r.repo_id "
            "WHERE r.report_id = ?",
            (report_id,),
        ).fetchone()
        if not row:
            return redirect(url_for("home"))
        sources = json.loads(row["sources"])
    finally:
        conn.close()
    return render_template("report.html", r=row, sources=sources)


@app.route("/report/<int:report_id>/delete", methods=["POST"])
def report_delete(report_id):
    conn = open_db()
    try:
        repo_id = conn.execute(
            "SELECT repo_id FROM reports WHERE report_id=?", (report_id,)
        ).fetchone()
        conn.execute("DELETE FROM reports WHERE report_id=?", (report_id,))
        conn.commit()
        flash("Report deleted.", "ok")
    finally:
        conn.close()
    return redirect(url_for("repo_detail", repo_id=repo_id["repo_id"]) if repo_id else url_for("home"))


@app.route("/settings", methods=["GET", "POST"])
def settings():
    conn = open_db()
    try:
        if request.method == "POST":
            for key in ("OPENROUTER_API_KEY", "KIT_GITHUB_TOKEN"):
                # Empty string clears; "********" placeholder leaves value alone.
                value = (request.form.get(key) or "").strip()
                if value and not set(value) <= {"•", "*"}:
                    set_setting(conn, key, value)
                elif request.form.get(f"clear_{key}"):
                    set_setting(conn, key, "")
            flash("Settings saved.", "ok")
            return redirect(url_for("settings"))

        keys = {}
        for key in ("OPENROUTER_API_KEY", "KIT_GITHUB_TOKEN"):
            stored = get_setting(conn, key)
            env = os.environ.get(key, "")
            keys[key] = {
                "stored": bool(stored),
                "env": bool(env) and not stored,
                "masked": _mask(stored or env),
            }
        return render_template("settings.html", keys=keys)
    finally:
        conn.close()


@app.route("/delete-repo/<int:repo_id>", methods=["POST"])
def delete_repo(repo_id):
    conn = open_db()
    try:
        # Vec0 doesn't honor FK cascades — delete vectors explicitly first.
        conn.execute(
            "DELETE FROM symbol_vectors WHERE symbol_id IN "
            "(SELECT symbol_id FROM symbols WHERE repo_id = ?)",
            (repo_id,),
        )
        conn.execute("DELETE FROM repositories WHERE repo_id = ?", (repo_id,))
        conn.commit()
        flash("Repo deleted.", "ok")
    finally:
        conn.close()
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
