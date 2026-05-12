-- Acron — sample data for grading.
--
-- Apply on a freshly-created schema:
--     sqlite3 data/pipeline.db < sql/create_schema.sql
--     sqlite3 data/pipeline.db < sql/initialize_data.sql
--
-- Populates every non-vector table with 15+ rows. All columns are filled
-- in (no NULLs) so foreign-key and NOT NULL constraints are satisfied even
-- where the application would normally leave a field blank.

PRAGMA foreign_keys = ON;

BEGIN TRANSACTION;

-- ---------------------------------------------------------------------------
-- tags (15 rows)
-- ---------------------------------------------------------------------------
INSERT INTO tags (tag_id, tag_name, color_hex) VALUES
    (1,  'todo',       '#f59e0b'),
    (2,  'bug',        '#ef4444'),
    (3,  'insight',    '#10b981'),
    (4,  'question',   '#3b82f6'),
    (5,  'refactor',   '#8b5cf6'),
    (6,  'security',   '#dc2626'),
    (7,  'perf',       '#0ea5e9'),
    (8,  'docs',       '#6b7280'),
    (9,  'test',       '#22c55e'),
    (10, 'deprecated', '#a16207'),
    (11, 'api',        '#0891b2'),
    (12, 'ui',         '#ec4899'),
    (13, 'db',         '#7c3aed'),
    (14, 'infra',      '#475569'),
    (15, 'review',     '#f97316');

-- ---------------------------------------------------------------------------
-- repositories (15 rows)
-- ---------------------------------------------------------------------------
INSERT INTO repositories (repo_id, github_url, repo_name, owner, indexed_at, file_count) VALUES
    (1,  'https://github.com/pallets/flask',          'flask',          'pallets',  '2026-05-01T10:00:00', 142),
    (2,  'https://github.com/psf/requests',           'requests',       'psf',      '2026-05-01T10:05:00',  87),
    (3,  'https://github.com/django/django',          'django',         'django',   '2026-05-01T10:15:00', 2614),
    (4,  'https://github.com/fastapi/fastapi',        'fastapi',        'fastapi',  '2026-05-01T10:30:00', 312),
    (5,  'https://github.com/tiangolo/typer',         'typer',          'tiangolo', '2026-05-01T10:45:00',  64),
    (6,  'https://github.com/pydantic/pydantic',      'pydantic',       'pydantic', '2026-05-02T09:00:00', 198),
    (7,  'https://github.com/sqlalchemy/sqlalchemy',  'sqlalchemy',     'sqlalchemy','2026-05-02T09:30:00', 728),
    (8,  'https://github.com/encode/httpx',           'httpx',          'encode',   '2026-05-02T10:00:00', 102),
    (9,  'https://github.com/encode/starlette',       'starlette',      'encode',   '2026-05-02T10:20:00',  76),
    (10, 'https://github.com/numpy/numpy',            'numpy',          'numpy',    '2026-05-03T08:00:00', 1450),
    (11, 'https://github.com/pandas-dev/pandas',      'pandas',         'pandas-dev','2026-05-03T08:30:00',1132),
    (12, 'https://github.com/scikit-learn/scikit-learn','scikit-learn', 'scikit-learn','2026-05-03T09:00:00',  892),
    (13, 'https://github.com/huggingface/transformers','transformers',  'huggingface','2026-05-03T10:00:00',2103),
    (14, 'https://github.com/openai/openai-python',   'openai-python',  'openai',   '2026-05-04T07:30:00', 154),
    (15, 'https://github.com/anthropics/anthropic-sdk-python','anthropic-sdk-python','anthropics','2026-05-04T08:00:00', 121);

-- ---------------------------------------------------------------------------
-- indexing_jobs (15 rows)
-- ---------------------------------------------------------------------------
INSERT INTO indexing_jobs (job_id, github_url, status, repo_id, file_count, symbol_count, error, created_at, updated_at) VALUES
    (1,  'https://github.com/pallets/flask',          'done',       1,   142,  1820, '', '2026-05-01T09:58:00','2026-05-01T10:01:30'),
    (2,  'https://github.com/psf/requests',           'done',       2,    87,   910, '', '2026-05-01T10:03:00','2026-05-01T10:05:45'),
    (3,  'https://github.com/django/django',          'done',       3,  2614, 31044, '', '2026-05-01T10:10:00','2026-05-01T10:16:12'),
    (4,  'https://github.com/fastapi/fastapi',        'done',       4,   312,  3850, '', '2026-05-01T10:27:00','2026-05-01T10:31:05'),
    (5,  'https://github.com/tiangolo/typer',         'done',       5,    64,   712, '', '2026-05-01T10:42:00','2026-05-01T10:45:30'),
    (6,  'https://github.com/pydantic/pydantic',      'done',       6,   198,  2410, '', '2026-05-02T08:55:00','2026-05-02T09:00:50'),
    (7,  'https://github.com/sqlalchemy/sqlalchemy',  'done',       7,   728,  9180, '', '2026-05-02T09:20:00','2026-05-02T09:31:00'),
    (8,  'https://github.com/encode/httpx',           'done',       8,   102,  1240, '', '2026-05-02T09:55:00','2026-05-02T10:00:40'),
    (9,  'https://github.com/encode/starlette',       'done',       9,    76,   880, '', '2026-05-02T10:15:00','2026-05-02T10:20:25'),
    (10, 'https://github.com/numpy/numpy',            'done',       10, 1450, 18402, '', '2026-05-03T07:45:00','2026-05-03T08:02:00'),
    (11, 'https://github.com/pandas-dev/pandas',      'done',       11, 1132, 14210, '', '2026-05-03T08:20:00','2026-05-03T08:32:10'),
    (12, 'https://github.com/scikit-learn/scikit-learn','done',     12,  892, 10980, '', '2026-05-03T08:50:00','2026-05-03T09:01:30'),
    (13, 'https://github.com/huggingface/transformers','done',      13, 2103, 27110, '', '2026-05-03T09:40:00','2026-05-03T10:02:00'),
    (14, 'https://github.com/openai/openai-python',   'done',       14,  154,  1865, '', '2026-05-04T07:25:00','2026-05-04T07:31:00'),
    (15, 'https://github.com/anthropics/anthropic-sdk-python','done',15, 121,  1402, '', '2026-05-04T07:55:00','2026-05-04T08:00:40');

-- ---------------------------------------------------------------------------
-- symbols (15 rows) — one per repo for FK variety
-- ---------------------------------------------------------------------------
INSERT INTO symbols (symbol_id, repo_id, file_path, symbol_name, symbol_type, start_line, end_line, code_snippet) VALUES
    (1,  1,  'src/flask/app.py',                  'Flask',           'class',    50,  410, 'class Flask(_PackageBoundObject): ...'),
    (2,  2,  'src/requests/api.py',               'get',             'function', 62,   78, 'def get(url, params=None, **kwargs): ...'),
    (3,  3,  'django/db/models/base.py',          'Model',           'class',   320, 1980, 'class Model(metaclass=ModelBase): ...'),
    (4,  4,  'fastapi/applications.py',           'FastAPI',         'class',    40,  610, 'class FastAPI(Starlette): ...'),
    (5,  5,  'typer/main.py',                     'Typer',           'class',    78,  340, 'class Typer: ...'),
    (6,  6,  'pydantic/main.py',                  'BaseModel',       'class',   110,  720, 'class BaseModel(metaclass=ModelMetaclass): ...'),
    (7,  7,  'lib/sqlalchemy/orm/session.py',     'Session',         'class',   142, 1280, 'class Session(_SessionClassMethods): ...'),
    (8,  8,  'httpx/_client.py',                  'AsyncClient',     'class',   810, 1520, 'class AsyncClient(BaseClient): ...'),
    (9,  9,  'starlette/applications.py',         'Starlette',       'class',    18,  220, 'class Starlette: ...'),
    (10, 10, 'numpy/core/numeric.py',             'array',           'function', 200,  280, 'def array(object, dtype=None, ...): ...'),
    (11, 11, 'pandas/core/frame.py',              'DataFrame',       'class',   312, 3210, 'class DataFrame(NDFrame, OpsMixin): ...'),
    (12, 12, 'sklearn/linear_model/_base.py',     'LinearRegression','class',   400,  640, 'class LinearRegression(MultiOutputMixin, ...): ...'),
    (13, 13, 'src/transformers/pipelines/__init__.py','pipeline',    'function', 420,  720, 'def pipeline(task, model=None, ...): ...'),
    (14, 14, 'src/openai/_client.py',             'OpenAI',          'class',   110,  430, 'class OpenAI(SyncAPIClient): ...'),
    (15, 15, 'src/anthropic/_client.py',          'Anthropic',       'class',   115,  450, 'class Anthropic(SyncAPIClient): ...');

-- ---------------------------------------------------------------------------
-- reports (15 rows)
-- ---------------------------------------------------------------------------
INSERT INTO reports (report_id, repo_id, question, answer, sources, model, created_at) VALUES
    (1,  1,  'How does Flask handle routing?',                 'Flask matches URL rules via Werkzeug Map/Rule and dispatches to view functions.', '["src/flask/app.py:200"]', 'moonshotai/kimi-k2.6', '2026-05-01T11:00:00'),
    (2,  2,  'Where is the default timeout set in requests?',  'No timeout is set by default; requests will block until the server responds.',     '["src/requests/api.py:62"]','moonshotai/kimi-k2.6', '2026-05-01T11:05:00'),
    (3,  3,  'How are Django models registered?',              'Models are registered via ModelBase metaclass when the class is defined.',         '["django/db/models/base.py:320"]','moonshotai/kimi-k2.6','2026-05-01T11:15:00'),
    (4,  4,  'How does FastAPI generate OpenAPI docs?',        'FastAPI walks registered routes and builds a schema from Pydantic models.',        '["fastapi/applications.py:80"]','moonshotai/kimi-k2.6','2026-05-01T11:30:00'),
    (5,  5,  'How does Typer parse CLI arguments?',            'Typer wraps Click and uses type hints to build options/arguments.',                '["typer/main.py:78"]', 'moonshotai/kimi-k2.6','2026-05-01T11:45:00'),
    (6,  6,  'What is BaseModel for?',                         'BaseModel provides validation, parsing, and serialization for typed fields.',     '["pydantic/main.py:110"]','moonshotai/kimi-k2.6','2026-05-02T09:10:00'),
    (7,  7,  'How does SQLAlchemy Session track changes?',     'Session keeps a unit-of-work identity map; dirty objects flush on commit.',        '["lib/sqlalchemy/orm/session.py:142"]','moonshotai/kimi-k2.6','2026-05-02T09:35:00'),
    (8,  8,  'Does httpx support HTTP/2?',                     'Yes, via the h2 package; pass http2=True when constructing AsyncClient.',         '["httpx/_client.py:810"]','moonshotai/kimi-k2.6','2026-05-02T10:05:00'),
    (9,  9,  'How does Starlette handle middleware?',          'Middleware wraps the app callable in a stack built at startup.',                  '["starlette/applications.py:18"]','moonshotai/kimi-k2.6','2026-05-02T10:25:00'),
    (10, 10, 'Is numpy.array a copy or a view?',               'np.array copies by default; pass copy=False to attempt a view when possible.',     '["numpy/core/numeric.py:200"]','moonshotai/kimi-k2.6','2026-05-03T08:10:00'),
    (11, 11, 'How does DataFrame store columns?',              'Internally as a BlockManager of typed ndarray blocks, not per-column.',             '["pandas/core/frame.py:312"]','moonshotai/kimi-k2.6','2026-05-03T08:40:00'),
    (12, 12, 'What solver does LinearRegression use?',         'It uses scipy.linalg.lstsq (LAPACK gelsd) by default.',                            '["sklearn/linear_model/_base.py:400"]','moonshotai/kimi-k2.6','2026-05-03T09:05:00'),
    (13, 13, 'How does pipeline() pick a default model?',      'It consults a per-task default in TASK_MAPPING when no model is given.',          '["src/transformers/pipelines/__init__.py:420"]','moonshotai/kimi-k2.6','2026-05-03T10:10:00'),
    (14, 14, 'How does the OpenAI client handle retries?',     'Idempotent HTTP errors are retried with exponential backoff (max_retries=2).',     '["src/openai/_client.py:110"]','moonshotai/kimi-k2.6','2026-05-04T07:35:00'),
    (15, 15, 'How does the Anthropic client stream tokens?',   'It uses Server-Sent Events; iterate the stream() context manager.',               '["src/anthropic/_client.py:115"]','moonshotai/kimi-k2.6','2026-05-04T08:05:00');

-- ---------------------------------------------------------------------------
-- settings (15 rows) — example config keys; values are placeholders, never real secrets.
-- ---------------------------------------------------------------------------
INSERT INTO settings (key, value) VALUES
    ('OPENROUTER_API_KEY',  'sk-or-v1-placeholder-replace-me'),
    ('GITHUB_TOKEN',        'ghp_placeholder_replace_me'),
    ('CHAT_MODEL',          'moonshotai/kimi-k2.6'),
    ('EMBED_MODEL',         'jinaai/jina-embeddings-v2-base-code'),
    ('EMBED_DIM',           '768'),
    ('TOP_K',               '8'),
    ('MAX_TOKENS',          '2048'),
    ('TEMPERATURE',         '0.2'),
    ('REWRITE_QUERIES',     'true'),
    ('CACHE_TTL_SECONDS',   '3600'),
    ('LOG_LEVEL',           'info'),
    ('UI_THEME',            'dark'),
    ('AUTO_INDEX_ON_PUSH',  'false'),
    ('MAX_FILE_SIZE_KB',    '512'),
    ('IGNORE_PATTERNS',     'node_modules,dist,build,.venv');

-- ---------------------------------------------------------------------------
-- notes (15 rows)
-- ---------------------------------------------------------------------------
INSERT INTO notes (note_id, symbol_id, note_text, tag_id, created_at) VALUES
    (1,  1,  'Routing dispatch could be extracted into its own module.',          5,  '2026-05-01T12:00:00'),
    (2,  2,  'Document the absence of a default timeout more prominently.',       8,  '2026-05-01T12:05:00'),
    (3,  3,  'ModelBase metaclass is the right place to plug in tenant scoping.', 3,  '2026-05-01T12:10:00'),
    (4,  4,  'OpenAPI generation is hot path; consider memoizing per-app.',       7,  '2026-05-01T12:20:00'),
    (5,  5,  'Add an example for Annotated[..., typer.Option(...)] in docs.',     8,  '2026-05-01T12:25:00'),
    (6,  6,  'Validate that arbitrary_types_allowed is documented for v2.',       4,  '2026-05-02T09:40:00'),
    (7,  7,  'Identity map size is unbounded — flag for long-running sessions.',  6,  '2026-05-02T09:50:00'),
    (8,  8,  'AsyncClient retry semantics differ from sync; needs a test.',       9,  '2026-05-02T10:30:00'),
    (9,  9,  'Middleware ordering bug surfaced when adding CORS after Auth.',     2,  '2026-05-02T10:40:00'),
    (10, 10, 'Reference: np.asarray vs np.array memory behavior.',                3,  '2026-05-03T08:20:00'),
    (11, 11, 'BlockManager refactor on the roadmap — track upstream PR.',         5,  '2026-05-03T08:50:00'),
    (12, 12, 'Switch default solver investigation: lsqr vs gelsd.',               7,  '2026-05-03T09:15:00'),
    (13, 13, 'TASK_MAPPING is a god-dict; consider per-task registries.',         5,  '2026-05-03T10:20:00'),
    (14, 14, 'Backoff jitter would reduce thundering-herd on rate limit.',        7,  '2026-05-04T07:45:00'),
    (15, 15, 'Stream() error path swallows the underlying SSE message.',          2,  '2026-05-04T08:15:00');

-- Keep AUTOINCREMENT counters in sync with the explicit IDs above so that
-- application-generated inserts after this seed won't collide.
INSERT OR REPLACE INTO sqlite_sequence (name, seq) VALUES
    ('tags',           15),
    ('repositories',   15),
    ('indexing_jobs',  15),
    ('symbols',        15),
    ('reports',        15),
    ('notes',          15);

COMMIT;
