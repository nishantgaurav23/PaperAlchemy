# Spec S9b.1 -- Alembic Migration Setup

## Overview
Initialize Alembic for async PostgreSQL migrations. Auto-generate the initial migration from the existing `Paper` ORM model, configure the async driver (`asyncpg`), and add Makefile commands (`make db-migrate`, `make db-upgrade`, `make db-downgrade`). This is a prerequisite for all future models (User, Comment, Vote, Note, Avatar, etc.) that will be added in P14+.

## Dependencies
- **S2.2** (Database layer) — `done` — provides `src/db/base.py` (DeclarativeBase), `src/db/database.py` (async engine), `src/config.py` (PostgresSettings with `url` and `sync_url`)

## Target Location
- `alembic/` — Alembic directory (env.py, versions/)
- `alembic.ini` — Alembic config (root)
- `Makefile` — new db-migrate, db-upgrade, db-downgrade targets
- `src/db/` — minor updates if needed

## Functional Requirements

### FR-1: Alembic Initialization
- **What**: Initialize Alembic with an async-compatible `env.py` that uses `asyncpg` via SQLAlchemy's async engine
- **Inputs**: `alembic.ini` config pointing to `alembic/` directory
- **Outputs**: `alembic.ini`, `alembic/env.py`, `alembic/versions/` directory, `alembic/script.py.mako` template
- **Edge cases**: Must work with both Docker Compose PostgreSQL and local dev PostgreSQL

### FR-2: Async env.py Configuration
- **What**: Custom `env.py` that imports `Base.metadata` from `src.db.base` and runs migrations using async engine from `src.config.get_settings().postgres.url`
- **Inputs**: Database URL from config (env vars or `.env`)
- **Outputs**: Migrations run against the configured PostgreSQL instance
- **Edge cases**: Must handle missing database gracefully, must support both online and offline migration modes

### FR-3: Initial Migration (Paper Model)
- **What**: Auto-generate the first migration from the existing `Paper` model in `src/models/paper.py`
- **Inputs**: `Paper` model with all columns (id, arxiv_id, title, authors, abstract, categories, published_date, updated_date, pdf_url, pdf_content, sections, parsing_status, parsing_error, created_at, updated_at) and indexes
- **Outputs**: Migration file in `alembic/versions/` that creates the `papers` table
- **Edge cases**: Migration must be idempotent-safe (won't fail if table already exists from `create_tables()`)

### FR-4: Makefile Commands
- **What**: Add three Makefile targets for database migrations
- **Inputs**: Developer runs `make db-migrate msg="description"`, `make db-upgrade`, or `make db-downgrade`
- **Outputs**:
  - `db-migrate`: Runs `alembic revision --autogenerate -m "$(msg)"`
  - `db-upgrade`: Runs `alembic upgrade head`
  - `db-downgrade`: Runs `alembic downgrade -1`
- **Edge cases**: `db-migrate` should require a message parameter

### FR-5: Model Import Registration
- **What**: Ensure all ORM models are imported in `env.py` so autogenerate detects them
- **Inputs**: `src/models/paper.py` (and future models)
- **Outputs**: `env.py` imports `src.models` package which re-exports all models
- **Edge cases**: New models added later must only need to be imported in `src/models/__init__.py`

## Tangible Outcomes
- [ ] `alembic.ini` exists at project root with correct config
- [ ] `alembic/env.py` uses async engine and imports `Base.metadata`
- [ ] `alembic/versions/` contains initial migration for `papers` table
- [ ] `make db-migrate msg="test"` generates a new migration file
- [ ] `make db-upgrade` applies pending migrations
- [ ] `make db-downgrade` reverts the last migration
- [ ] All ORM models auto-discovered via `src/models/__init__.py` import
- [ ] Tests validate Alembic config, migration file existence, and env.py correctness

## Test-Driven Requirements

### Tests to Write First
1. `test_alembic_ini_exists`: Verify `alembic.ini` exists and has correct `script_location`
2. `test_alembic_env_imports_base_metadata`: Verify `env.py` imports `Base.metadata` target
3. `test_initial_migration_exists`: Verify at least one migration file exists in `alembic/versions/`
4. `test_initial_migration_creates_papers_table`: Verify the initial migration's `upgrade()` creates the `papers` table
5. `test_alembic_config_loads`: Verify `alembic.config.Config("alembic.ini")` loads without error
6. `test_models_init_imports_paper`: Verify `from src.models import Paper` works (model registration)
7. `test_makefile_has_db_targets`: Verify Makefile contains `db-migrate`, `db-upgrade`, `db-downgrade` targets

### Mocking Strategy
- No external service mocking needed — these are structural/config tests
- Tests validate file existence, imports, and config correctness
- No database connection required for unit tests

### Coverage
- All Alembic config files tested for correctness
- Migration file structure validated
- Model registration verified
- Makefile targets verified
