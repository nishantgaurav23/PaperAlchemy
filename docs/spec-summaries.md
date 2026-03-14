# PaperAlchemy — Spec Implementation Summaries

> Developer-facing log of completed specs. Each entry explains **what** was built, **how** it was implemented, **why** it matters, and its **core features**. Updated automatically after each spec is verified.

---

## How to Read This Document

Each spec summary follows this structure:

| Section | Purpose |
|---------|---------|
| **What Was Done** | High-level description of the deliverable |
| **How It Was Done** | Implementation approach, key patterns, libraries used |
| **Why It Matters** | How this spec enables downstream work and improves the system |
| **Core Features** | Bulleted list of concrete capabilities added |
| **Key Files** | Files created or modified |
| **Dependencies Unlocked** | Which specs can now proceed |

---

## S1.1 — Dependency Declaration

**Phase:** P1 (Project Foundation) | **Status:** done

### What Was Done
Clean dependency declaration for PaperAlchemy using UV package manager. The `pyproject.toml` was restructured with all production and development dependencies properly versioned and organized by purpose (web framework, database, search, LLM, embeddings, caching, evaluation, etc.).

### How It Was Done
- Rewrote `pyproject.toml` with grouped dependencies: web (FastAPI, Uvicorn), database (SQLAlchemy async, asyncpg), search (opensearch-py), LLM (LangChain, LangGraph, Gemini), embeddings (sentence-transformers), caching (Redis), evaluation (RAGAS), PDF parsing (Docling)
- Replaced sync drivers with async alternatives: `psycopg2-binary` → `asyncpg`, `requests` → `httpx`
- Added `langchain-google-genai` for Gemini cloud LLM
- Added `ragas` for RAG evaluation framework and `sentence-transformers` for cross-encoder re-ranking
- Added `aiofiles` for async file I/O (PDF downloads, uploads, temp files)
- Added `sse-starlette` for Server-Sent Events streaming (token-by-token responses)
- Removed legacy/unused deps: `gradio`, `types-sqlalchemy`
- `python-telegram-bot` deferred to P12 (Telegram Bot phase) — will be re-added in S12.1
- Configured tooling: Ruff (lint + format, line-length=130), pytest (asyncio_mode=auto), mypy (pydantic plugin)
- Ran `uv sync` to generate lockfile and verify resolution

### Why It Matters
This is the **root spec** — every other spec depends on packages declared here. Clean dependency management ensures:
- No version conflicts as specs are implemented
- Async-first foundation (asyncpg, httpx) for the entire backend
- All advanced RAG components (re-ranking, HyDE, multi-query) have their deps ready
- Tooling (ruff, pytest, mypy) is configured consistently from day one

### Core Features
- All 20+ production dependencies declared with version bounds
- 15+ dev dependencies for testing, linting, type checking, notebooks
- Async PostgreSQL driver (asyncpg) replacing sync psycopg2
- Async HTTP client (httpx) replacing sync requests
- Ruff configured: E, F, I, N, W, UP, B, SIM, ASYNC rules
- pytest configured: auto asyncio mode, .env.test loading
- mypy configured: Pydantic plugin, explicit package bases
- UV lockfile generated and verified

### Key Files
- `pyproject.toml` — dependency declaration + tool config
- `uv.lock` — resolved lockfile
- `notebooks/specs/S1.1_dependency.ipynb` — verification notebook

### Dependencies Unlocked
- **S1.2** (Environment Config) — can now use pydantic-settings
- **S1.3** (Docker Infrastructure) — deps ready for Dockerfile
- **S1.4** (CI/CD Setup) — ruff, pytest, mypy configured

---

## S1.2 — Environment Configuration

**Phase:** P1 (Project Foundation) | **Status:** done

### What Was Done
Centralized application configuration using Pydantic Settings. 11 nested sub-settings classes with `env_prefix` convention (double underscore), composed into a single root `Settings` object. Supports `.env` files, environment variable overrides, and type-safe validation.

### How It Was Done
- Created `src/config.py` with 11 `BaseSettings` subclasses: PostgresSettings, OpenSearchSettings, OllamaSettings, GeminiSettings (NEW), RedisSettings, JinaSettings (NEW), RerankerSettings (NEW), LangfuseSettings, ArxivSettings, ChunkingSettings, AppSettings
- Each class uses `env_prefix` (e.g., `POSTGRES__`, `GEMINI__`) for env var mapping
- URL properties on PostgresSettings (async `postgresql+asyncpg://` + sync `postgresql://`), OllamaSettings (`http://`), RedisSettings (`redis://` with optional password)
- Root `Settings` class composes all 11 sub-settings with `.env` file reading and `extra="ignore"`
- `get_settings()` factory with `@lru_cache` for singleton behavior
- Updated `.env.example` with all 50+ documented environment variables
- Updated `.env.test` with test-safe defaults (no real credentials)
- TDD: 25 tests written first, all passing

### Why It Matters
Every service in PaperAlchemy needs connection details. This spec provides a **single source of truth** for all configuration:
- Local dev: defaults work out of the box (localhost, default ports)
- Docker: compose.yml sets `POSTGRES__HOST=postgres`, etc.
- Production: env vars or `.env` file configure everything
- Type safety: Pydantic validates types (port must be int, url must be str)
- New vs reference: Added GeminiSettings (cloud LLM), JinaSettings (embeddings), RerankerSettings (cross-encoder) — not in old code

### Core Features
- 11 nested sub-settings classes with env_prefix convention
- Async PostgreSQL URL (`postgresql+asyncpg://`) for SQLAlchemy async engine
- Sync PostgreSQL URL for Alembic migrations
- Redis URL with optional password handling
- GeminiSettings for Google Gemini (cloud LLM)
- JinaSettings for Jina AI embeddings (1024-dim, batch 100)
- RerankerSettings for cross-encoder re-ranking (ms-marco-MiniLM-L-12-v2)
- Cached singleton via `get_settings()` with `@lru_cache`
- Comprehensive `.env.example` with all variables documented
- Test-safe `.env.test` for pytest

### Key Files
- `src/config.py` — all settings classes + get_settings()
- `.env.example` — documented example configuration
- `.env.test` — test-safe defaults
- `tests/unit/test_config.py` — 25 tests
- `notebooks/specs/S1.2_config.ipynb` — interactive verification

### Dependencies Unlocked
- **S2.1** (FastAPI App Factory) — needs AppSettings for app config
- **S2.2** (Database Layer) — needs PostgresSettings for connection
- **S3.2** (ArXiv Client) — needs ArxivSettings
- **S4.3** (Embedding Service) — needs JinaSettings
- **S5.1** (LLM Client) — needs OllamaSettings + GeminiSettings
- **S4b.1** (Reranker) — needs RerankerSettings

---

## S1.3 — Docker Infrastructure

**Phase:** P1 (Project Foundation) | **Status:** done

### What Was Done
Multi-service Docker Compose setup with Docker profiles for selective service startup, health checks on all 14 services, and an updated Makefile with profile-aware commands. The existing compose.yml, Dockerfile, and Makefile were enhanced — not rewritten — to support core/full/langfuse/dev-tools profiles.

### How It Was Done
- Added Docker profiles to compose.yml: core services (postgres, redis, opensearch) have no profile and always start; `full` profile for api, ollama, airflow; `langfuse` profile for the 6-service Langfuse v3 stack (ClickHouse, Langfuse Postgres/Redis/MinIO/Web/Worker); `dev-tools` profile for opensearch-dashboards and pgAdmin
- Added missing healthchecks to pgAdmin (wget spider) and langfuse-worker (wget health endpoint)
- Updated Makefile with profile-aware targets: `up` (core only), `up-all` (all profiles), `up-langfuse` (core + langfuse), `down` / `down-clean` (stop all profiles), `build` (API image)
- Makefile supports per-service logs with `make logs s=<service>`
- Verified existing Dockerfile multi-stage UV build pattern (already correct from prior work)
- TDD: 22 structural tests validating YAML structure, profiles, healthchecks, volumes, network, Dockerfile stages, and Makefile targets

### Why It Matters
Docker profiles solve the **resource problem** — developers no longer need to run all 14 services. `docker compose up` starts only the 3 core services (postgres, redis, opensearch), using ~1GB RAM instead of ~8GB. The Langfuse observability stack and dev tools are opt-in. This makes development accessible on machines with limited resources while keeping the full stack available when needed.

### Core Features
- 14 services with health checks, restart policies, named volumes, and shared network
- 4 Docker profiles: (default), full, langfuse, dev-tools
- `make up` starts core (3 services) — fast dev startup
- `make up-all` starts everything (14 services)
- `make up-langfuse` starts core + Langfuse (9 services)
- `make down` / `make down-clean` stop all profiles
- `make build` builds API image
- `make logs s=api` for per-service log tailing
- Multi-stage Dockerfile: UV base → Python slim final (with docling system libs)
- 8 named volumes for data persistence

### Key Files
- `compose.yml` — 14 services with profiles and health checks
- `Dockerfile` — multi-stage UV build (unchanged, verified)
- `Makefile` — profile-aware Docker commands
- `tests/unit/test_docker_infrastructure.py` — 22 structural tests
- `notebooks/specs/S1.3_docker.ipynb` — interactive verification

### Dependencies Unlocked
- **S2.2** (Database Layer) — depends on S1.3 for PostgreSQL container
- **S4.1** (OpenSearch Client) — depends on S1.3 for OpenSearch container
- **S11.3** (Ops Documentation) — depends on S1.3 for Docker commands reference

---

## S1.4 — CI/CD Setup

**Phase:** P1 (Project Foundation) | **Status:** done

### What Was Done
GitHub Actions CI workflow for PaperAlchemy. Four jobs run on every push to `main` and on pull requests: linting (Ruff check + format), type checking (mypy), testing (pytest with coverage), and Docker image build verification. Lint, type-check, and test run in parallel; docker-build runs after test passes.

### How It Was Done
- Created `.github/workflows/ci.yml` with 4 jobs: `lint`, `type-check`, `test`, `docker-build`
- All Python jobs use `astral-sh/setup-uv@v5` with `enable-cache: true` for fast UV dependency installation, keyed on `uv.lock` hash
- All Python jobs use `actions/setup-python@v5` with Python 3.12
- Lint job: `ruff check src/ tests/` + `ruff format --check src/ tests/`
- Type-check job: `mypy src/`
- Test job: `pytest tests/ --cov=src --cov-report=xml --cov-report=term-missing` + upload coverage artifact
- Docker-build job: `docker/setup-buildx-action@v3` + `docker/build-push-action@v6` with `push: false`, `load: true`, GHA cache
- Docs-only changes (*.md, docs/, notebooks/) skip CI via `paths-ignore`
- TDD: 15 structural tests validating YAML structure, triggers, jobs, Python version, caching, and parallelism

### Why It Matters
Automated CI catches lint errors, type issues, test failures, and Docker build breakage **before** code reaches main. This is the foundation CI — production CI/CD with deployment (staging/prod environments) is covered in S11.5. UV caching ensures fast CI runs (~30s for dependency install vs ~2min without cache).

### Core Features
- 4 CI jobs: lint, type-check, test, docker-build
- Parallel execution: lint, type-check, test run simultaneously
- Docker build gated on test success
- UV dependency caching via `astral-sh/setup-uv@v5`
- Docker BuildKit GHA cache for fast image builds
- Coverage report uploaded as artifact (XML format)
- Docs-only changes skip CI (paths-ignore)
- Read-only permissions (security best practice)

### Key Files
- `.github/workflows/ci.yml` — CI workflow (4 jobs)
- `tests/unit/test_ci_cd_setup.py` — 15 structural tests
- `notebooks/specs/S1.4_cicd.ipynb` — interactive verification

### Dependencies Unlocked
- **S11.5** (Production CI/CD) — extends this foundation CI with deployment stages

---

## S2.1 — FastAPI App Factory

**Phase:** P2 (Backend Core) | **Status:** done

### What Was Done
App factory pattern for the FastAPI application with async lifespan management, CORS middleware, router registration under `/api/v1` prefix, and a health check endpoint. This is the entry point for the entire PaperAlchemy backend.

### How It Was Done
- Created `src/main.py` with `create_app(settings_override=None)` factory function
- Async lifespan via `@asynccontextmanager` — logs startup/shutdown messages, stubs for future DB/Redis/OpenSearch init
- `CORSMiddleware` added with permissive dev defaults (`allow_origins=["*"]`)
- Router registration: `app.include_router(ping_router, prefix="/api/v1")`
- Module-level `app = create_app()` for `uvicorn src.main:app` entry point
- Created `src/routers/ping.py` with `GET /ping` endpoint returning `PingResponse`
- Created `src/schemas/api/health.py` with `PingResponse` Pydantic model (`status`, `version`)
- `settings_override` parameter on `create_app()` enables test isolation
- TDD: 8 tests written first, all passing

### Why It Matters
This is the **backbone** of the backend — every router, middleware, and service wires through the app factory. The lifespan pattern provides clean startup/shutdown hooks for database pools, search clients, and cache connections. The `settings_override` parameter makes testing isolated and fast.

### Core Features
- `create_app()` factory with optional `Settings` override for testing
- Async lifespan with startup/shutdown logging (extensible for future services)
- CORS middleware (permissive dev defaults)
- `GET /api/v1/ping` health check returning `{"status": "ok", "version": "0.1.0"}`
- `PingResponse` Pydantic schema
- Router registration with `/api/v1` prefix convention
- 404 for unknown routes

### Key Files
- `src/main.py` — app factory + lifespan
- `src/routers/__init__.py` — routers package
- `src/routers/ping.py` — health check endpoint
- `src/schemas/__init__.py` — schemas package
- `src/schemas/api/__init__.py` — API schemas package
- `src/schemas/api/health.py` — PingResponse model
- `tests/unit/test_main.py` — 8 tests
- `notebooks/specs/S2.1_app_factory.ipynb` — interactive verification

### Dependencies Unlocked
- **S2.3** (Error Handling) — needs app factory to add middleware and exception handlers
- **S2.4** (Dependency Injection) — needs app factory to register DI providers
- **S11.2** (Monitoring) — needs app factory for /metrics endpoint

---

## S2.2 — Database Layer

**Phase:** P2 (Backend Core) | **Status:** done

### What Was Done
Async SQLAlchemy 2.0 database layer with connection pooling, async session factory with auto-commit/rollback/close lifecycle, health check, table management, and FastAPI dependency injection support. Rebuilt from scratch using async-first design (`asyncpg` + `AsyncSession`), replacing the sync reference implementation.

### How It Was Done
- Created `src/db/base.py` with `DeclarativeBase` subclass for ORM model inheritance
- Created `src/db/database.py` with `Database` class wrapping `create_async_engine` and `async_sessionmaker`
- Engine configured with production-ready pool settings: `pool_size=5`, `max_overflow=10`, `pool_pre_ping=True`, `pool_recycle=3600`
- `get_session()` async context manager: yields `AsyncSession`, auto-commits on success, auto-rollbacks on `SQLAlchemyError`, always closes
- `health_check()` executes `SELECT 1` to verify connectivity
- `create_tables()` / `drop_tables()` use `run_sync` with `Base.metadata`
- `close()` disposes the engine and releases all connections
- Created `src/db/__init__.py` with `init_database()` singleton factory and `get_db_session()` FastAPI dependency
- Wired into `src/main.py` lifespan: `init_database()` on startup, `db.close()` on shutdown
- TDD: 21 unit tests with mocked engines/sessions (no real DB needed)

### Why It Matters
This is the **data foundation** — every model, repository, and service that touches PostgreSQL depends on this layer. The async-first design ensures non-blocking I/O throughout the stack. The session lifecycle (auto-commit/rollback/close) prevents connection leaks and data corruption. The singleton pattern via `init_database()` ensures one engine per process with proper connection pooling.

### Core Features
- `Base` declarative base for all ORM models (importable without side effects)
- `Database` class with async engine, session factory, health check, table management
- Connection pooling: pool_size=5, max_overflow=10, pool_pre_ping, 1h recycle
- Session lifecycle: auto-commit on success, auto-rollback on error, always close
- `expire_on_commit=False` for post-commit attribute access
- `health_check()` returns bool (True/False) — no exceptions leak
- `init_database()` singleton factory for app-wide use
- `get_db_session()` FastAPI `Depends()` async generator
- Lifespan integration: engine created on startup, disposed on shutdown

### Key Files
- `src/db/base.py` — DeclarativeBase for ORM models
- `src/db/database.py` — async Database class (engine, sessions, health check)
- `src/db/__init__.py` — public API (init_database, get_db_session, Base, Database)
- `src/main.py` — lifespan updated with DB init/close
- `tests/unit/test_database.py` — 21 unit tests
- `notebooks/specs/S2.2_database.ipynb` — interactive verification

### Dependencies Unlocked
- **S2.4** (Dependency Injection) — needs Database + get_db_session for DI container
- **S3.1** (Paper Model) — needs Base for ORM model + sessions for repository

---

## S2.3 — Error Handling & Middleware

**Phase:** P2 (Backend Core) | **Status:** done

### What Was Done
Global error handling and request logging middleware for the FastAPI application. Custom exception hierarchy (19 classes) organized by subsystem, structured JSON error responses via `ErrorResponse` schema, global exception handlers that map exceptions to HTTP status codes, and a `RequestLoggingMiddleware` with timing, correlation IDs (`X-Request-ID`), and path exclusions.

### How It Was Done
- Created `src/exceptions.py` with `PaperAlchemyError` base class carrying `detail`, `status_code`, and `context` attributes. 19 exception classes organized by subsystem: Repository (404/500), Parsing (422/500), ExternalService (503), ArXiv (429/503), LLM, Embedding, Search, Cache, Pipeline, Configuration
- Created `src/schemas/api/error.py` with `ErrorDetail` + `ErrorResponse` Pydantic models for structured JSON error bodies: `{"error": {"type", "message", "request_id", "detail"}}`
- Created `src/middlewares.py` with `RequestLoggingMiddleware` (Starlette `BaseHTTPMiddleware`): generates/forwards `X-Request-ID`, logs `{method} {path} -> {status} ({duration}ms)`, excludes noisy paths (`/api/v1/ping`, `/docs`)
- Middleware also catches unhandled exceptions that bypass Starlette's exception handler layer (known `BaseHTTPMiddleware` limitation), returning safe 500 JSON without stack trace leaks in production, with traceback in debug mode
- `register_error_handlers(app)` registers `PaperAlchemyError` and generic `Exception` handlers on the FastAPI app
- Wired into `create_app()` in `src/main.py`: error handlers registered first, then middleware added
- TDD: 34 tests (22 exception hierarchy + schema, 12 handler + middleware integration)

### Why It Matters
Every router and service needs consistent error handling. Without this, exceptions would return raw HTML 500 errors or leak stack traces. The structured `ErrorResponse` schema gives frontend consumers a predictable contract. The `X-Request-ID` correlation enables request tracing across logs. The subsystem-specific exception hierarchy enables precise `try/except` in downstream specs (e.g., `except PaperNotFoundError: return 404` in repositories, `except ExternalServiceError: return 503` in service clients).

### Core Features
- 19 exception classes organized by subsystem with default status codes
- `PaperAlchemyError` base: `detail`, `status_code`, `context` attributes
- Structured `ErrorResponse` JSON: type, message, request_id, optional detail
- Exception → HTTP mapping: 404 (NotFound), 422 (Validation), 429 (RateLimit), 500 (Internal), 503 (ServiceUnavailable)
- `RequestLoggingMiddleware`: auto-generated UUID4 `X-Request-ID` or forwarded from incoming header
- Request timing in milliseconds logged at INFO level
- Path exclusions: `/api/v1/ping`, `/docs`, `/openapi.json`, `/redoc`
- Unhandled exceptions: safe 500 in production, traceback in debug mode
- Error responses include `request_id` for correlation

### Key Files
- `src/exceptions.py` — 19 exception classes in subsystem hierarchy
- `src/schemas/api/error.py` — ErrorDetail + ErrorResponse Pydantic models
- `src/middlewares.py` — RequestLoggingMiddleware + exception handlers + register_error_handlers()
- `src/main.py` — updated create_app() with error handlers and middleware
- `tests/unit/test_exceptions.py` — 22 tests (hierarchy, schema)
- `tests/unit/test_middlewares.py` — 12 tests (handlers, middleware)
- `notebooks/specs/S2.3_error_handling.ipynb` — interactive verification

### Dependencies Unlocked
- **S2.4** (Dependency Injection) — can now use exception classes in DI error handling

---

## S2.4 — Dependency Injection

**Phase:** P2 (Backend Core) | **Status:** done

### What Was Done
FastAPI `Depends()` pattern with typed `Annotated[]` aliases for clean, testable service injection. Routers declare what they need via type hints (`SettingsDep`, `DatabaseDep`, `SessionDep`) — they never import factories or create clients directly.

### How It Was Done
- Created `src/dependency.py` with 3 getter functions and 3 `Annotated[]` type aliases
- `get_settings()` re-exports `src.config.get_settings` (lru_cached singleton)
- `get_database()` wraps `src.db._get_database()` — raises `RuntimeError` if not initialized
- `get_db_session()` re-exports `src.db.get_db_session` (async generator yielding `AsyncSession`)
- `SettingsDep`, `DatabaseDep`, `SessionDep` are `Annotated[Type, Depends(getter)]` aliases
- All dependencies are overridable via `app.dependency_overrides` for testing
- Module exports all functions and aliases via `__all__`
- TDD: 15 tests covering unit, integration, and dependency override patterns

### Why It Matters
This is the **DI foundation** — every future router uses these aliases to receive services without importing factories. Testing becomes trivial: swap `app.dependency_overrides[get_settings] = lambda: mock` and the entire route handler uses mocks. The module is designed to grow — each future spec (OpenSearch, Jina, LLM, Cache) adds its own getter + alias here.

### Core Features
- `get_settings()` → `Settings` singleton
- `get_database()` → `Database` instance (or `RuntimeError`)
- `get_db_session()` → async generator yielding `AsyncSession` with auto-commit/rollback/close
- `SettingsDep`, `DatabaseDep`, `SessionDep` type aliases for concise router signatures
- Full `app.dependency_overrides` support for testing
- Extensible pattern for future service dependencies

### Key Files
- `src/dependency.py` — dependency functions + type aliases
- `tests/unit/test_dependency.py` — 15 tests
- `notebooks/specs/S2.4_di.ipynb` — interactive verification

### Dependencies Unlocked
- **S3.1** (Paper Model) — can use `SessionDep` in repository endpoints
- **S8.1** (PDF Upload) — can use `SessionDep` + future service deps in upload router
- All future routers — can use typed dependency injection

---

## S3.1 — Paper ORM Model & Repository

**Phase:** P3 (Data Layer) | **Status:** done

### What Was Done
Core Paper SQLAlchemy ORM model with UUID primary key and an async repository layer providing full CRUD, upsert, bulk upsert, and filtered query operations. Includes Pydantic schemas for API input/output validation and FastAPI dependency injection wiring.

### How It Was Done
- Created `src/models/paper.py` with `Paper` class inheriting from `Base` — 15 columns including UUID PK, arXiv metadata (arxiv_id, title, authors, abstract, categories, dates), PDF content (pdf_content, sections), parsing status, and timestamps (created_at, updated_at with server defaults)
- Used `JSON` column type (portable across SQLite/PostgreSQL) for authors, categories, and sections — enables in-memory SQLite testing while working with PostgreSQL in production
- Created `src/repositories/paper.py` with async `PaperRepository` — 14 async methods: create, get_by_id, get_by_arxiv_id, exists, update, update_parsing_status, delete, upsert, bulk_upsert, get_by_date_range, get_by_category, get_pending_parsing, search, count
- Upsert uses get-then-update pattern for SQLite compatibility (PostgreSQL ON CONFLICT can be swapped in for production perf)
- Category filtering uses `cast(categories, String).like(...)` for cross-database compatibility
- Created `src/schemas/paper.py` with PaperCreate, PaperUpdate (all fields optional), PaperResponse (from_attributes=True)
- Wired `PaperRepoDep` into `src/dependency.py` for router injection
- Imported Paper model in `src/db/__init__.py` so `Base.metadata.create_all` discovers it
- TDD: 36 tests using in-memory SQLite async engine — all CRUD, queries, edge cases, and schema validation

### Why It Matters
This is the **data backbone** — every downstream spec that stores, retrieves, or queries papers depends on this model and repository. The ingestion pipeline (S3.4) writes papers here, the search system (S4.x) reads from here, and the RAG pipeline (S5.x) retrieves paper content. The async repository pattern ensures non-blocking database access throughout the stack.

### Core Features
- `Paper` ORM model with UUID PK, 15 columns, 3 indexes (arxiv_id unique, published_date, parsing_status)
- Async `PaperRepository` with 14 methods — all `async def` using `AsyncSession`
- Create, read (by ID, by arXiv ID), update (partial), delete
- Upsert (single) and bulk upsert (batch)
- Query by date range, category, parsing status, text search (title/abstract ILIKE)
- Multi-filter search with combined conditions
- Count with optional status filter
- `PaperCreate`, `PaperUpdate`, `PaperResponse` Pydantic schemas
- `PaperRepoDep` DI alias for FastAPI router injection
- Cross-database compatible (SQLite for tests, PostgreSQL for production)

### Key Files
- `src/models/paper.py` — Paper ORM class (UUID PK, 15 columns, indexes)
- `src/models/__init__.py` — re-exports Paper
- `src/repositories/paper.py` — async PaperRepository (14 methods)
- `src/repositories/__init__.py` — re-exports PaperRepository
- `src/schemas/paper.py` — PaperCreate, PaperUpdate, PaperResponse
- `src/dependency.py` — updated with PaperRepoDep + get_paper_repository
- `src/db/__init__.py` — updated to import Paper for metadata discovery
- `tests/unit/test_paper_model.py` — 36 tests
- `notebooks/specs/S3.1_paper_model.ipynb` — interactive verification

### Dependencies Unlocked
- **S3.4** (Ingestion Pipeline) — can now store fetched arXiv papers via PaperRepository
- **S8.1** (PDF Upload) — can now create Paper records for uploaded PDFs

---

## S3.2 — ArXiv API Client

**Phase:** P3 (Data Layer) | **Status:** done

### What Was Done
Async arXiv API client with rate limiting, exponential backoff retry, query building, Atom XML parsing, and PDF download with local caching. Uses `httpx` for HTTP and `feedparser` for Atom feed parsing. Includes `ArxivPaper` Pydantic schema and factory function with singleton caching.

### How It Was Done
- Created `src/schemas/arxiv.py` with `ArxivPaper` model (arxiv_id, title, authors, abstract, categories, published_date, updated_date, pdf_url) + `arxiv_url` computed property
- Created `src/services/arxiv/client.py` with `ArxivClient` class — 7 methods covering the full arXiv API interaction lifecycle
- Rate limiting: tracks `_last_request_time`, enforces >= 3.0s delay via `asyncio.sleep`, clamps any configured delay < 3.0 to 3.0
- Retry logic: exponential backoff on 503 (`delay * 2^attempt`), extended wait on 429 (`delay * 2^attempt * 10`), retry on timeout/connection errors, immediate raise on other HTTP errors
- Query building: supports category filter (`cat:cs.AI`), date range (`submittedDate:[YYYYMMDD TO YYYYMMDD]`), search terms (`all:query`), combined with AND
- Entry parsing: extracts arXiv ID from URL, strips version suffix (`v\d+$`), parses authors/categories/dates from feedparser entry, builds PDF URL
- PDF download: caches to local directory, validates Content-Type (must contain "pdf"), enforces 50MB size limit, verifies `%PDF-` magic bytes, atomic write (temp file + rename), returns None on any failure
- Created `src/services/arxiv/factory.py` with `make_arxiv_client()` using `@lru_cache` for singleton
- Wired `ArxivClientDep` into `src/dependency.py` for router injection
- TDD: 33 tests covering schema, rate limiting, retry (429/503/timeout/fatal/exhausted), query building (7 variants), fetch papers (success/empty/malformed/params), PDF download (7 scenarios), and factory

### Why It Matters
This is the **data ingestion entry point** — every paper in PaperAlchemy originates from arXiv. The rate limiting and retry logic respect arXiv API guidelines (mandatory 3s delay, backoff on errors), ensuring reliable fetching without getting blocked. The PDF download with caching prevents redundant downloads during re-ingestion. Downstream specs (S3.4 ingestion pipeline, S3.3 PDF parser) depend on this client to provide paper metadata and PDF files.

### Core Features
- `ArxivClient` with rate limiting (>= 3s between requests)
- Exponential backoff retry on 429, 503, timeout, connection errors (up to `max_retries` attempts)
- Immediate error on non-retryable HTTP status codes (400, 404, etc.)
- Query builder supporting category, date range, search terms, and combined queries
- `fetch_papers()` — query arXiv API, parse Atom XML via feedparser, return `list[ArxivPaper]`
- `download_pdf()` — download with Content-Type validation, 50MB limit, magic byte verification, atomic write, local caching
- `ArxivPaper` Pydantic schema with `arxiv_url` property
- `make_arxiv_client()` factory with singleton caching from `ArxivSettings`
- `ArxivClientDep` for FastAPI dependency injection
- All external HTTP mocked in tests — 33 tests, 0 real network calls

### Key Files
- `src/schemas/arxiv.py` — ArxivPaper Pydantic model
- `src/services/arxiv/client.py` — ArxivClient (rate limit, retry, fetch, download)
- `src/services/arxiv/factory.py` — make_arxiv_client singleton factory
- `src/services/arxiv/__init__.py` — public exports
- `src/dependency.py` — updated with ArxivClientDep
- `tests/unit/test_arxiv_client.py` — 33 tests
- `notebooks/specs/S3.2_arxiv_client.ipynb` — interactive verification

### Dependencies Unlocked
- **S3.4** (Ingestion Pipeline) — can now fetch papers from arXiv and download PDFs
- **S3.3** (PDF Parser) — receives downloaded PDFs from this client

---

## S3.3 — PDF Parser (Docling)

**Phase:** P3 (Data Layer) | **Status:** done

### What Was Done
Section-aware PDF parsing service using Docling for structured content extraction from academic papers. Extracts sections (with heading hierarchy), tables (text representation), figure captions, and raw text. Enforces file validation (50MB max, PDF magic bytes, extension check). Runs Docling synchronously in a thread pool with async wrapper and timeout protection. Lazy-initializes the Docling converter for fast startup.

### How It Was Done
- Created `src/schemas/pdf.py` with `Section` (title, content, level 1-6) and `PDFContent` (raw_text, sections, tables, figures, page_count, parser_used, parser_time_seconds) Pydantic models
- Created `src/services/pdf_parser/service.py` with `PDFParserService` class — 6 methods covering validation, sync parsing, async parsing, batch parsing, and cleanup
- File validation (`_validate_file`): checks existence, `.pdf` extension, file size <= max_file_size_mb, and `%PDF-` magic bytes — raises `PDFValidationError` (422) on failure
- Sync parsing (`_parse_sync`): lazy-initializes Docling `DocumentConverter`, extracts raw text via `export_to_text()`, iterates `doc.texts` to build sections (heading detection via label containing "heading"/"header"), extracts tables via `export_to_text()`, extracts figure captions from `doc.pictures`
- Content with no headings gets grouped into a single "Introduction" section
- Async parsing (`parse_pdf`): validates file, runs `_parse_sync` in `ThreadPoolExecutor` via `loop.run_in_executor`, wraps with `asyncio.wait_for` for timeout protection — raises `PDFParsingError` on timeout
- Batch parsing (`parse_multiple`): sequential iteration with `continue_on_error` flag — returns `dict[str, PDFContent | None]`
- Created `src/services/pdf_parser/factory.py` with `make_pdf_parser_service()` using `@lru_cache` singleton from `PDFParserSettings`
- Added `PDFParserSettings` to `src/config.py` with `env_prefix="PDF_PARSER__"` (max_pages=30, max_file_size_mb=50, timeout=120)
- Uses existing exception hierarchy: `PDFValidationError` (422) and `PDFParsingError` (500) from `src/exceptions.py`
- TDD: 31 tests — all Docling interactions mocked via direct `svc._converter` injection, no real PDF parsing in unit tests

### Why It Matters
This is the **content extraction layer** — every paper that enters PaperAlchemy needs its PDF content extracted into structured sections for downstream chunking (S4.2), indexing (S4.1), and RAG retrieval (S5.x). The section-aware parsing preserves document structure, enabling section-based chunking that respects paper boundaries (Abstract, Methods, Results, etc.). The async wrapper with timeout protection prevents slow PDFs from blocking the event loop.

### Core Features
- `PDFParserService` with lazy Docling converter initialization
- File validation: existence, extension, size (50MB), magic bytes (`%PDF-`)
- Section extraction from Docling document: heading detection, content grouping
- Table extraction as text representation
- Figure caption extraction
- Default "Introduction" section for documents without headings
- Async parsing in thread pool with configurable timeout (default 120s)
- Batch parsing with `continue_on_error` flag
- `PDFParserSettings` config with env var support (`PDF_PARSER__MAX_PAGES`, etc.)
- Singleton factory via `make_pdf_parser_service()`
- Resource cleanup (`close()`) — shuts executor, clears converter

### Key Files
- `src/schemas/pdf.py` — Section + PDFContent Pydantic models
- `src/services/pdf_parser/service.py` — PDFParserService (validate, parse, batch, cleanup)
- `src/services/pdf_parser/factory.py` — make_pdf_parser_service singleton factory
- `src/services/pdf_parser/__init__.py` — public exports
- `src/config.py` — updated with PDFParserSettings
- `tests/unit/test_pdf_parser.py` — 31 tests
- `notebooks/specs/S3.3_pdf_parser.ipynb` — interactive verification

### Dependencies Unlocked
- **S3.4** (Ingestion Pipeline) — can now parse downloaded PDFs into structured content
- **S4.2** (Text Chunker) — depends on S3.3 for section-aware content to chunk
- **S8.1** (PDF Upload) — depends on S3.3 for parsing uploaded PDFs

---

## S3.4 — Ingestion Pipeline (Airflow DAG)

**Phase:** P3 (Data Layer) | **Status:** done

### What Was Done
Airflow DAG orchestrating the daily paper ingestion pipeline (fetch arXiv papers, download PDFs, parse with Docling, store to PostgreSQL) plus a FastAPI ingestion endpoint that performs the actual fetch-parse-store workflow. The DAG tasks communicate via XCom and delegate all write operations to the REST API via HTTP calls to avoid SQLAlchemy version conflicts between Airflow and the application.

### How It Was Done
- Created `airflow/dags/arxiv_paper_ingestion.py` with DAG definition: 4 tasks in linear chain (setup → fetch → report → cleanup), Mon-Fri 6am UTC schedule, 2 retries with 30-min delay, `catchup=False`, `max_active_runs=1`
- Created `airflow/dags/arxiv_ingestion/` package with task modules: `common.py` (API URLs, timeouts), `setup.py` (health check via `GET /api/v1/ping`), `fetching.py` (calls `POST /api/v1/ingest/fetch` with `target_date = execution_date - 1 day`), `reporting.py` (aggregates XCom results into structured JSON report)
- Task functions use `httpx` for HTTP calls with proper error handling: `HTTPStatusError` → `RuntimeError` (triggers Airflow retry), `ConnectError` → `RuntimeError`
- Created `src/routers/ingest.py` with `POST /api/v1/ingest/fetch` endpoint that orchestrates: fetch papers from arXiv → upsert metadata to DB → download PDFs → parse with Docling → update parsing status — all idempotent via upserts
- Created `src/schemas/api/ingest.py` with `IngestRequest` (target_date with regex validation) and `IngestResponse` (papers_fetched, pdfs_downloaded, pdfs_parsed, papers_stored, arxiv_ids, errors, processing_time)
- DAG file uses `sys.path.insert` to add dags directory, avoiding namespace conflict with apache-airflow package
- Cleanup task uses `BashOperator` with `find -mtime +30 -delete || true` for safe old PDF removal
- TDD: 34 tests total — 24 DAG task tests (9 DAG config tests skipped locally as apache-airflow not installed, validated in Airflow container), 10 router endpoint tests

### Why It Matters
This completes Phase 3 (Data Layer) — PaperAlchemy now has a **fully automated ingestion pipeline**. Papers flow from arXiv API → PostgreSQL → parsed PDF content, ready for downstream chunking (S4.2), indexing (S4.1), and search (S4.4). The separation of Airflow orchestration from application logic (via REST API) ensures clean architecture and avoids dependency conflicts. The idempotent design allows safe re-runs without data duplication.

### Core Features
- Airflow DAG with 4-task linear pipeline: setup → fetch → report → cleanup
- Mon-Fri 6am UTC schedule with 2 retries, 30-min retry delay
- Health check gate (fail fast if API/DB is down)
- XCom-based cross-task data flow (fetch results → report)
- `POST /api/v1/ingest/fetch` endpoint: arXiv fetch → PDF download → Docling parse → DB upsert
- Idempotent ingestion via upserts — safe to re-run
- Structured daily report with paper counts, error tracking, processing time
- Automatic cleanup of PDFs older than 30 days
- Graceful error handling: PDF download/parse failures don't block other papers
- `IngestRequest` with regex-validated target_date, defaults to yesterday
- `IngestResponse` with comprehensive metrics

### Key Files
- `airflow/dags/arxiv_paper_ingestion.py` — DAG definition (4 tasks)
- `airflow/dags/arxiv_ingestion/common.py` — API URLs, timeouts
- `airflow/dags/arxiv_ingestion/setup.py` — health check task
- `airflow/dags/arxiv_ingestion/fetching.py` — fetch task (XCom push)
- `airflow/dags/arxiv_ingestion/reporting.py` — report task (XCom pull)
- `src/routers/ingest.py` — POST /api/v1/ingest/fetch endpoint
- `src/schemas/api/ingest.py` — IngestRequest, IngestResponse schemas
- `src/main.py` — updated with ingest router registration
- `tests/unit/test_ingestion_dag.py` — 24 DAG/task tests
- `tests/unit/test_ingest_router.py` — 10 endpoint tests
- `notebooks/specs/S3.4_ingestion.ipynb` — interactive verification

### Dependencies Unlocked
- **S12.4** (Telegram Notifications) — can now trigger paper alerts based on ingestion results

---

## S9.1 — Next.js Project Setup

**Phase:** P9 (Frontend) | **Status:** done

### What Was Done
Next.js 16 frontend application scaffolded with TypeScript strict mode, Tailwind CSS v4, shadcn/ui component library, dark mode via next-themes, Vitest testing with React Testing Library, and a typed API client for communicating with the FastAPI backend.

### How It Was Done
- Scaffolded Next.js 16 with `create-next-app` using App Router, TypeScript, Tailwind CSS v4, `src/` directory, pnpm package manager
- Initialized shadcn/ui with Button component as proof of integration
- Added `next-themes` ThemeProvider wrapping the app in `layout.tsx` with `suppressHydrationWarning` for SSR hydration safety
- Created `ThemeToggle` component using lucide-react icons (Sun/Moon) and `useTheme` hook
- Configured Vitest with `@vitejs/plugin-react`, jsdom environment, `@/` path aliases, and `@testing-library/jest-dom` setup
- Created typed `apiClient` utility (`src/lib/api-client.ts`) with GET/POST/PUT/DELETE methods, timeout support via `AbortController`, custom `ApiError` class, and base URL from `NEXT_PUBLIC_API_URL` env var
- Updated home page with PaperAlchemy branding and theme toggle
- TDD: 16 tests across 3 test files — all passing

### Why It Matters
This is the **frontend foundation** — every Phase 9 spec (search, chat, upload, dashboard, export) builds on this project setup. The Vitest + React Testing Library configuration enables TDD for all frontend specs. The API client provides typed communication with the FastAPI backend. Dark mode support is baked in from day one.

### Core Features
- Next.js 16 with App Router, TypeScript strict mode, `src/` directory
- Tailwind CSS v4 with CSS variables for light/dark themes
- shadcn/ui component library with Button component
- Dark mode toggle (light/dark/system) via next-themes
- Vitest + React Testing Library + jsdom testing setup
- Typed API client with error handling, timeouts, and env-based base URL
- ESLint with Next.js recommended rules
- pnpm package manager

### Key Files
- `frontend/src/app/layout.tsx` — root layout with ThemeProvider
- `frontend/src/app/page.tsx` — home page with PaperAlchemy branding
- `frontend/src/components/theme-provider.tsx` — next-themes wrapper
- `frontend/src/components/theme-toggle.tsx` — dark mode toggle button
- `frontend/src/components/ui/button.tsx` — shadcn/ui Button
- `frontend/src/lib/api-client.ts` — typed API client (GET/POST/PUT/DELETE)
- `frontend/src/lib/utils.ts` — cn() utility for Tailwind class merging
- `frontend/vitest.config.ts` — Vitest configuration
- `frontend/src/test/setup.ts` — test setup (jest-dom matchers)
- `frontend/src/app/page.test.tsx` — 3 home page tests
- `frontend/src/components/ui/button.test.tsx` — 5 Button component tests
- `frontend/src/lib/api-client.test.ts` — 8 API client tests

### Dependencies Unlocked
- **S9.2** (Layout & Navigation) — can now build app shell on this foundation
- **S10.3** (Human Evaluation UI) — depends on S9.1 for frontend project
- **S10.5** (Benchmark Dashboard) — depends on S9.1 for frontend project

---

## S4.1 — OpenSearch Client + Index Configuration

**Phase:** P4 (Search & Retrieval) | **Status:** done

### What Was Done
OpenSearch client wrapping opensearch-py with PaperAlchemy-specific logic: hybrid index mappings (BM25 + KNN 1024-dim HNSW), RRF search pipeline, query builder, BM25/vector/hybrid search, bulk chunk indexing, chunk lifecycle management, and factory functions with singleton caching.

### How It Was Done
- Created `src/services/opensearch/index_config.py` with `ARXIV_PAPERS_CHUNKS_MAPPING` (strict dynamic mapping, 19 fields including knn_vector with HNSW/nmslib, text_analyzer with snowball stemming, standard_analyzer) and `HYBRID_RRF_PIPELINE` (score-ranker-processor with RRF technique, k=60)
- Created `src/services/opensearch/query_builder.py` with `QueryBuilder` class — builds BM25 queries with multi_match (fuzziness=AUTO), category filters, highlighting (`<mark>` tags), pagination, and sort (relevance vs date). Supports chunk mode (chunk_text^3, title^2, abstract^1) and paper mode (title^3, abstract^2, authors^1)
- Created `src/services/opensearch/client.py` with `OpenSearchClient` — 12 public methods: health_check, setup_indices, search_papers, search_chunks_vectors, search_unified, search_chunks_hybrid, bulk_index_chunks, delete_paper_chunks, get_chunks_by_paper, get_index_stats, plus private helpers for index/pipeline creation and result formatting
- Unified search entry point (`search_unified`) routes to BM25-only or hybrid (BM25+KNN+RRF) based on whether embedding is provided
- Hybrid search uses OpenSearch native `hybrid` query with `search_pipeline` parameter pointing to RRF pipeline
- Created `src/services/opensearch/factory.py` with `make_opensearch_client()` (lru_cache singleton) and `make_opensearch_client_fresh()` (new instance for notebooks/tests)
- Wired `OpenSearchDep` into `src/dependency.py` for FastAPI router injection
- TDD: 40 tests written first, all OpenSearch API calls mocked — no real cluster needed

### Why It Matters
This is the **search engine foundation** — every retrieval operation in PaperAlchemy flows through this client. The hybrid search capability (BM25 + KNN + RRF) enables the advanced RAG pipeline: BM25 finds keyword matches while KNN finds semantically similar chunks, and RRF fuses their rankings without fragile score normalization. The bulk indexing supports the ingestion pipeline, and the chunk lifecycle methods enable re-indexing. Downstream specs (S4.2 text chunker, S4.3 embeddings, S4.4 hybrid search endpoint) all depend on this client.

### Core Features
- Hybrid chunk index mapping: 19 fields, strict dynamic, knn_vector (1024-dim, HNSW, nmslib, ef_construction=512, m=16)
- Custom analyzers: text_analyzer (snowball stemming), standard_analyzer (stop words)
- RRF search pipeline: score-ranker-processor with rank_constant=60
- `QueryBuilder`: multi_match with fuzziness, category filter, highlighting, pagination, relevance/date sort
- `health_check()`: cluster status green/yellow → True
- `setup_indices(force)`: creates hybrid index + RRF pipeline
- `search_papers()`: BM25 keyword search
- `search_chunks_vectors()`: pure KNN vector search with optional category filter
- `search_unified()`: routes to BM25 or hybrid based on embedding presence
- `search_chunks_hybrid()`: native OpenSearch hybrid query with RRF pipeline
- `bulk_index_chunks()`: opensearchpy.helpers.bulk with refresh
- `delete_paper_chunks()` / `get_chunks_by_paper()`: chunk lifecycle by arxiv_id
- `get_index_stats()`: document count, size, existence status
- Factory: singleton (lru_cache) + fresh instance patterns
- `OpenSearchDep` for FastAPI dependency injection

### Key Files
- `src/services/opensearch/__init__.py` — public exports
- `src/services/opensearch/index_config.py` — ARXIV_PAPERS_CHUNKS_MAPPING, HYBRID_RRF_PIPELINE
- `src/services/opensearch/query_builder.py` — QueryBuilder class
- `src/services/opensearch/client.py` — OpenSearchClient (12 public methods)
- `src/services/opensearch/factory.py` — make_opensearch_client, make_opensearch_client_fresh
- `src/dependency.py` — updated with OpenSearchDep
- `tests/unit/test_opensearch_client.py` — 40 tests
- `notebooks/specs/S4.1_opensearch.ipynb` — interactive verification

### Dependencies Unlocked
- **S4.4** (Hybrid Search) — can now build search endpoint using OpenSearchClient
- **S4b.1** (Re-ranking) — re-ranks results from OpenSearch search
- **S4b.5** (Retrieval Pipeline) — orchestrates OpenSearch as retrieval backend

---

## S4.2 — Text Chunker (Section-Aware)

**Phase:** P4 (Search & Retrieval) | **Status:** done

### What Was Done
Section-aware text chunking service that splits parsed PDF content into overlapping chunks for embedding and indexing. Uses a hybrid strategy: section-based chunking when sections are available (from Docling PDF parser), falling back to word-based sliding window. Produces `List[TextChunk]` with full positional metadata for the downstream embedding + indexing pipeline.

### How It Was Done
- Created `src/schemas/indexing.py` with `ChunkMetadata` (position, overlaps, section_title) and `TextChunk` (text + metadata + paper IDs) Pydantic models
- Created `src/services/indexing/text_chunker.py` with `TextChunker` class implementing:
  - **Word-based chunking** (`chunk_text`): sliding window with configurable chunk_size (600), overlap (100), min_chunk_size (100); tracks character offsets for source highlighting
  - **Section parsing** (`_parse_sections`): handles `dict`, `list[dict]`, `list[Section]`, JSON strings — normalizes to `dict[str, str]`
  - **Section filtering** (`_filter_sections`): removes metadata sections (authors, affiliations), abstract duplicates (substring + 80% word overlap detection), short metadata-only sections
  - **Section-based chunking** (`_chunk_by_sections`): small sections (<100 words) combined, medium (100-800) become single chunks with header, large (>800) split with word-based chunking
  - **Header prepending**: every section chunk includes `{title}\n\nAbstract: {abstract}\n\n` for standalone context
  - **Main entry point** (`chunk_paper`): tries section-based first, falls back to word-based on failure
- Configuration validation: raises `ValueError` if overlap >= chunk_size
- TDD: 41 unit tests covering all FRs, edge cases, and metadata accuracy
- No external dependencies to mock — pure computation

### Why It Matters
This is the **bridge between PDF parsing and search indexing**. Without proper chunking, retrieval quality degrades significantly. Section-aware chunking preserves document structure (Introduction, Methods, Results), producing chunks that are more coherent and yield better search results. Header prepending ensures each chunk has enough context to be understood in isolation during retrieval.

### Core Features
- Word-based sliding window chunking (600 words, 100 overlap, min 100)
- Section-based hybrid chunking (small/medium/large handling)
- Header prepending (title + abstract on every section chunk)
- Section parsing from multiple input formats (dict, list, Section objects, JSON)
- Metadata section filtering (authors, affiliations, abstract duplicates)
- Full chunk metadata: chunk_index, start_char, end_char, word_count, overlaps, section_title
- Graceful fallback from section-based to word-based
- Configuration validation (overlap < chunk_size)

### Key Files
- `src/schemas/indexing.py` — `ChunkMetadata` and `TextChunk` models
- `src/services/indexing/text_chunker.py` — `TextChunker` class
- `src/services/indexing/__init__.py` — public API exports
- `tests/unit/test_text_chunker.py` — 41 unit tests
- `notebooks/specs/S4.2_chunker.ipynb` — interactive verification

### Dependencies Unlocked
- **S4.4** (Hybrid Search) — can now chunk papers before embedding and indexing
- **S4b.4** (Parent-Child Chunks) — extends chunking with parent-child relationships

---

## S9.2 — Layout & Navigation

**Phase:** P9 (Frontend — Next.js) | **Status:** done

### What Was Done
Built the application shell for PaperAlchemy's Next.js frontend: a responsive layout with a collapsible sidebar, top header with theme toggle and breadcrumbs, and a mobile navigation drawer. This establishes the consistent UI frame that all page-level specs (S9.3-S9.9) will render inside.

### How It Was Done
- Created 7 layout components using React 19 + TypeScript strict mode + Tailwind CSS v4
- `SidebarNavItem`: Link component with active route detection via `usePathname()`, supports collapsed (icon-only) mode
- `Sidebar`: Desktop sidebar with 6 nav items (Search, Chat, Upload, Papers, Collections, Dashboard), collapse toggle with localStorage persistence, branding header
- `Breadcrumbs`: Route-aware breadcrumb trail generated from pathname segments, capitalized labels, Home link
- `Header`: Top bar integrating breadcrumbs + existing ThemeToggle + optional mobile hamburger menu button
- `MobileNav`: Full-screen overlay drawer with backdrop click-to-close and close button
- `AppShell`: Root wrapper composing Sidebar + Header + MobileNav + main content area
- Updated `layout.tsx` to wrap all pages in AppShell within ThemeProvider
- TDD approach: wrote 38 failing tests first across 6 test files, then implemented to green
- Fixed lint issue by using `useState` initializer function instead of `useEffect` for reading localStorage

### Why It Matters
The app shell is the foundational UI structure that every subsequent frontend spec builds on. S9.3-S9.8 all depend on S9.2 for layout, navigation, and responsive behavior. Without this, no page-level UI work can proceed.

### Core Features
- Responsive app shell with sidebar (desktop) + drawer (mobile) navigation
- 6 navigation items with icons: Search, Chat, Upload, Papers, Collections, Dashboard
- Collapsible sidebar with localStorage state persistence
- Route-aware breadcrumbs with clickable intermediate segments
- Theme toggle (dark/light) integrated into header
- Mobile hamburger menu with overlay drawer
- Accessible: aria-labels on all interactive elements, semantic HTML (nav, header, main, dialog)
- Hidden sidebar on mobile (`md:` breakpoint), always visible on desktop

### Key Files
- `frontend/src/components/layout/app-shell.tsx` — root layout wrapper
- `frontend/src/components/layout/sidebar.tsx` — collapsible sidebar with nav items
- `frontend/src/components/layout/header.tsx` — header bar with breadcrumbs + theme toggle
- `frontend/src/components/layout/breadcrumbs.tsx` — route-aware breadcrumb trail
- `frontend/src/components/layout/mobile-nav.tsx` — mobile drawer navigation
- `frontend/src/components/layout/sidebar-nav-item.tsx` — individual nav link with active state
- `frontend/src/components/layout/nav-items.ts` — shared nav item definitions
- `frontend/src/components/layout/index.ts` — barrel export
- `frontend/src/app/layout.tsx` — updated to use AppShell
- 6 test files with 38 layout-specific tests (53 total across project)

### Dependencies Unlocked
- **S9.3** (Search Interface) — renders inside AppShell at `/search`
- **S9.4** (Chat Interface) — renders inside AppShell at `/chat`
- **S9.5** (Paper Upload UI) — renders inside AppShell at `/upload`
- **S9.6** (Paper Detail View) — renders inside AppShell at `/papers/[id]`
- **S9.7** (Reading Lists) — renders inside AppShell at `/collections`
- **S9.8** (Trends Dashboard) — renders inside AppShell at `/dashboard`

---

## S4.3 — Embedding Service (Jina AI)

**Completed**: 2026-03-11 | **Phase**: P4 (Search & Retrieval) | **Depends on**: S1.2

### What Was Done
Async client for the Jina AI Embeddings v3 API that converts text into 1024-dimensional vectors. Supports asymmetric encoding (`retrieval.passage` for documents, `retrieval.query` for queries) and batch processing for efficient indexing.

### How It Was Done
- `JinaEmbeddingsClient` wraps `httpx.AsyncClient` with Bearer auth and configurable timeout
- `embed_passages()` processes texts in configurable batches (default 100) — 250 texts with batch_size=100 makes exactly 3 API calls
- `embed_query()` embeds a single query with `retrieval.query` task for asymmetric search
- All HTTP errors (401, 429, 500, timeout, connection) are caught and wrapped in `EmbeddingServiceError` with descriptive messages
- Pydantic schemas (`JinaEmbeddingRequest`, `JinaEmbeddingResponse`) validate API payloads
- Factory validates API key presence and raises `ConfigurationError` if missing
- Async context manager with idempotent `close()` for safe lifecycle management

### Why It Matters
Embeddings are the foundation of semantic (KNN) search. Without them, PaperAlchemy can only do keyword-based BM25 search. The Jina client enables the vector component of hybrid search (S4.4), which combines BM25 + KNN + RRF fusion for significantly better retrieval quality. The asymmetric passage/query encoding improves search accuracy by optimizing embeddings for their respective roles.

### Core Features
- `embed_passages(texts, batch_size)` — batch embedding for indexing (1024-dim vectors)
- `embed_query(query)` — single query embedding for search (1024-dim vector)
- Automatic batching with configurable batch size
- HTTP error wrapping: 401 → auth message, 429 → rate limit message, timeout → timeout message
- Empty input handling (returns `[]` without API call)
- Input validation (empty/whitespace query raises `ValueError`)
- Async context manager (`async with JinaEmbeddingsClient(...) as client:`)
- Factory with settings validation (`make_embeddings_client()`)
- DI wiring via `EmbeddingsDep` in `src/dependency.py`

### Key Files
- `src/services/embeddings/client.py` — `JinaEmbeddingsClient` class
- `src/services/embeddings/factory.py` — `make_embeddings_client()` factory
- `src/services/embeddings/__init__.py` — module exports
- `src/schemas/embeddings.py` — `JinaEmbeddingRequest`, `JinaEmbeddingResponse`, `JinaEmbeddingData`, `JinaUsage`
- `src/dependency.py` — added `EmbeddingsDep` type alias
- `tests/unit/test_embedding_client.py` — 14 client tests
- `tests/unit/test_embedding_schemas.py` — 8 schema tests
- `tests/unit/test_embedding_factory.py` — 3 factory tests
- `notebooks/specs/S4.3_embeddings.ipynb` — interactive verification notebook

### Dependencies Unlocked
- **S4.4** (Hybrid Search) — can now embed queries for KNN search alongside BM25

---

## S9.3 — Search Interface

### What Was Done
Built the full search page for the Next.js frontend with search bar, arXiv category filters, sort options, paper result cards with arXiv links, pagination, and loading/empty/error states. All search state is URL-driven for shareable, bookmark-friendly URLs.

### How It Was Done
- **URL-driven state**: All search params (q, category, sort, page) persisted in URL via `useSearchParams` + `useRouter` — enables back/forward and shareable links
- **Component architecture**: 6 focused components — `SearchBar`, `CategoryFilter`, `SortSelect`, `PaperCard`, `Pagination`, `SearchResults` — each independently tested
- **API layer**: `searchPapers()` function in `lib/api/search.ts` wraps the API client, ready to connect to the backend `GET /api/v1/search` endpoint
- **Type system**: `Paper`, `SearchResponse`, `SearchParams` types + `ARXIV_CATEGORIES` and `SORT_OPTIONS` constants in `types/paper.ts`
- **Native selects**: Used native `<select>` elements for category/sort filters for maximum test reliability with jsdom (vs complex portal-based shadcn Select)
- **TDD**: 49 new tests across 7 test files, all passing (106 total across project)

### Why It Matters
- Primary discovery interface for the knowledge base — users search and browse papers here
- URL-driven architecture means search results are shareable and SEO-friendly
- Paper type definitions (`types/paper.ts`) and API client (`lib/api/search.ts`) reusable by other frontend specs
- Ready to wire to backend S4.4 hybrid search endpoint with zero frontend changes

### Core Features
- Full-width search bar with search icon, clear button, Enter-to-submit
- Category filter dropdown with 10 common arXiv CS/ML categories
- Sort by relevance, newest first, oldest first
- Paper cards: title (links to detail page), authors (truncated to 3 + "et al."), abstract (200-char preview), date (formatted), category badges, arXiv external link
- Pagination with page numbers, prev/next, ellipsis for large sets, result count display
- Loading skeletons (4 animated placeholder cards)
- Empty state ("No papers found" with suggestions)
- Error state with retry button
- Initial state ("Search for papers" prompt)
- Responsive layout (stacks on mobile)

### Key Files
- `frontend/src/app/search/page.tsx` — Search page with URL state management
- `frontend/src/components/search/search-bar.tsx` — Search input component
- `frontend/src/components/search/category-filter.tsx` — Category dropdown
- `frontend/src/components/search/sort-select.tsx` — Sort dropdown
- `frontend/src/components/search/paper-card.tsx` — Paper result card
- `frontend/src/components/search/pagination.tsx` — Pagination controls
- `frontend/src/components/search/search-results.tsx` — Results orchestrator (loading/empty/error/results)
- `frontend/src/components/search/index.ts` — Barrel exports
- `frontend/src/lib/api/search.ts` — Search API client function
- `frontend/src/types/paper.ts` — Paper types, categories, sort options
- `frontend/src/components/ui/input.tsx` — shadcn/ui Input (installed)
- `frontend/src/components/ui/badge.tsx` — shadcn/ui Badge (installed)
- `frontend/src/components/ui/skeleton.tsx` — shadcn/ui Skeleton (installed)
- `frontend/src/components/ui/select.tsx` — shadcn/ui Select (installed)

### Dependencies Unlocked
- **S9.4** (Chat Interface) — can reuse Paper types and API patterns
- **S9.6** (Paper Detail View) — paper cards link to `/papers/[id]` detail page

---

## S4.4 — Hybrid Search (BM25 + KNN + RRF)

**Phase:** P4 (Search & Retrieval) | **Status:** done

### What Was Done
Unified search endpoint combining BM25 keyword search with KNN vector search via Reciprocal Rank Fusion (RRF). Gracefully degrades to BM25-only when embedding generation fails. Includes search health check endpoint.

### How It Was Done
- Created `HybridSearchRequest` Pydantic schema with query validation (1-500 chars), pagination (size/from_), category filters, hybrid toggle, and min_score threshold
- Created `SearchHit` and `SearchResponse` schemas with chunk-level data (chunk_text, chunk_id, section_title) and `search_mode` indicator ("hybrid" | "bm25")
- Implemented `POST /api/v1/search` endpoint orchestrating OpenSearch client (S4.1) and Jina embeddings (S4.3) via FastAPI DI
- Implemented graceful fallback: if `embed_query()` raises any exception, search proceeds with BM25-only (no 5xx returned)
- Added health check guard: returns 503 if OpenSearch is unreachable before attempting search
- Implemented `GET /api/v1/search/health` for subsystem connectivity monitoring
- Fixed DI pattern: created no-arg wrapper functions (`get_opensearch_client`, `get_embeddings_client`) in `dependency.py` to prevent FastAPI from misinterpreting factory parameters as sub-dependencies

### Why It Matters
This is the foundation for all retrieval in PaperAlchemy. Every downstream feature — RAG pipeline (P5), agent system (P6), chatbot (P7), and advanced retrieval (P4b) — depends on this search endpoint. The graceful fallback ensures search remains available even when the embedding service is down.

### Core Features
- `POST /api/v1/search` — hybrid BM25 + KNN + RRF search with graceful BM25 fallback
- `GET /api/v1/search/health` — OpenSearch connectivity check
- Query validation: 1-500 chars, size 1-100, from_ >= 0, min_score 0.0-1.0
- Chunk-level results: chunk_text, chunk_id, section_title, highlights
- `search_mode` in response tells clients whether hybrid or BM25 was actually used
- Category filtering and latest-first sorting
- 25 tests covering schemas, endpoint, fallback, error handling, pagination

### Key Files
- `src/routers/search.py` — Search router (hybrid_search + search_health endpoints)
- `src/schemas/api/search.py` — HybridSearchRequest, SearchHit, SearchResponse
- `src/dependency.py` — Added get_opensearch_client(), get_embeddings_client() wrappers
- `src/main.py` — Registered search_router
- `tests/unit/test_search_router.py` — 25 tests
- `notebooks/specs/S4.4_hybrid_search.ipynb` — Interactive verification

### Dependencies Unlocked
- **S4b.1** (Re-ranking) — can re-rank search results
- **S4b.2** (HyDE) — builds on hybrid search with hypothetical document embeddings
- **S4b.3** (Multi-query) — generates multiple queries and fuses results via hybrid search
- **S12.3** (Telegram search) — Telegram bot can use search endpoint

---

## S9.4 — Chat Interface (Streaming)

**Status**: done | **Phase**: P9 (Frontend) | **Depends on**: S9.2

### What Was Done
Built a full RAG chatbot interface for the Next.js frontend with SSE streaming simulation, citation rendering with clickable arXiv links, session management, and a mock API layer ready to be swapped for the real `/api/v1/chat` backend endpoint.

### How It Was Done
- **Component architecture**: 7 focused components — MessageInput, MessageBubble, CitationBadge, SourceCard, WelcomeState, TypingIndicator, ScrollToBottom
- **Streaming simulation**: Mock SSE layer using `setInterval` that emits tokens progressively, with AbortController support for cancellation
- **Citation system**: Regex-based inline `[1]`, `[2]` parsing → renders as CitationBadge buttons; SourceCard renders paper metadata with arXiv links
- **State management**: React `useState`/`useCallback`/`useRef` hooks in the page component; no external state library needed
- **Session management**: UUID-based session IDs via `crypto.randomUUID()`, "New Chat" resets state
- **Error handling**: Error messages rendered as distinct "error" role messages with inline retry buttons
- **Testing**: Vitest + React Testing Library + @testing-library/user-event; streamChat mocked for deterministic page tests
- **ESLint compliance**: Fixed impure `Date.now()` in render by pre-computing timestamp in state

### Why It Matters
This is the **primary research interaction interface** — users will ask questions and receive citation-backed answers here. The mock SSE layer enables full UI development and testing without the backend chat API (S7.3). Once the backend is ready, switching is a single environment variable change (`NEXT_PUBLIC_API_URL`).

### Core Features
- Auto-growing textarea with Enter to send, Shift+Enter for newline, 2000-char limit
- User messages right-aligned (primary bg), assistant messages left-aligned (muted bg)
- Token-by-token streaming with animated typing indicator
- "Stop generating" button during streams (saves partial response)
- Inline citation badges `[1]`, `[2]` rendered as clickable superscript buttons
- Source cards with paper title, authors (truncated to 3), year, and arXiv link (`target="_blank"`)
- Welcome state with 4 suggested research questions (clickable → auto-submit)
- "New Chat" button clears history, generates new session UUID
- Error messages with distinct styling and retry button
- Auto-scroll to bottom on new messages, scroll-to-bottom FAB when scrolled up
- Relative timestamps ("just now", "2m ago")
- Simple markdown rendering (paragraphs, lists, code blocks)
- Responsive layout via Tailwind utilities

### Key Files
- `frontend/src/app/chat/page.tsx` — Chat page (full orchestration)
- `frontend/src/components/chat/` — 7 components + barrel export
- `frontend/src/lib/api/chat.ts` — Mock SSE streaming + real API support
- `frontend/src/types/chat.ts` — ChatMessage, ChatSource, ChatStreamEvent types
- `frontend/src/app/chat/page.test.tsx` — 8 integration tests
- `frontend/src/components/chat/*.test.tsx` — 7 component test files (40 tests total)
- `frontend/src/lib/api/chat.test.ts` — 4 API layer tests

### Dependencies Unlocked
- **S9.5** (Paper upload UI) — can proceed (depends on S9.2, not S9.4)
- No specs directly depend on S9.4, but it is the primary consumer of S7.3 (Chat API) when implemented

---

## S9.5 — Paper Upload UI

**Phase**: P9 (Frontend) · **Status**: done · **Date**: 2026-03-11

### What
Drag-and-drop PDF upload page for the Next.js frontend. Users upload academic PDFs and receive AI-generated analysis: structured summary, key highlights, and methodology breakdown. Includes progress indicator and error handling with retry.

### How
- **DropZone** component: drag-and-drop + file picker, validates PDF-only and 50MB limit, visual drag-over feedback
- **UploadProgress** component: animated progress bar with status states (uploading → processing → complete)
- **AnalysisResults** component: tabbed display (Summary / Highlights / Methodology) with paper metadata header, category badges, arXiv links
- **Upload API client**: mock mode for development (simulated 2s delay + sample Transformer paper analysis), real FormData upload ready for backend S8.x integration
- **Upload page**: state machine (idle → uploading → processing → complete | error), error state with retry + try-different-file options

### Why
- Enables paper upload workflow before backend analysis endpoints (S8.1-S8.4) are built
- Mock data provides full development + demo capability
- Consistent with existing patterns (search page, chat page) for maintainability

### Core Features
- Drag-and-drop zone with click-to-browse fallback
- PDF-only validation (reject non-PDF, max 50MB)
- Animated upload progress with status text
- Tabbed analysis results: Summary (5 sections), Highlights (3 categories), Methodology (approach, datasets, baselines, results)
- Paper metadata display with arXiv link and category badges
- Error handling with retry button
- "Upload Another" reset flow
- Responsive design, dark mode support

### Key Files
- `frontend/src/app/upload/page.tsx` — Upload page (state machine: idle → uploading → processing → complete | error)
- `frontend/src/components/upload/drop-zone.tsx` — Drag-and-drop file upload zone
- `frontend/src/components/upload/upload-progress.tsx` — Progress bar with status states
- `frontend/src/components/upload/analysis-results.tsx` — Tabbed analysis display (Summary, Highlights, Methodology)
- `frontend/src/lib/api/upload.ts` — Upload API client with mock fallback
- `frontend/src/types/upload.ts` — UploadResponse, PaperSummary, PaperHighlights, MethodologyAnalysis types
- `frontend/src/components/upload/drop-zone.test.tsx` — 11 tests
- `frontend/src/components/upload/upload-progress.test.tsx` — 5 tests
- `frontend/src/components/upload/analysis-results.test.tsx` — 8 tests
- `frontend/src/lib/api/upload.test.ts` — 2 tests

### Dependencies Unlocked
- No specs directly depend on S9.5
- Will integrate with S8.1 (PDF upload endpoint) when backend is implemented

---

## S4b.1 — Cross-Encoder Re-ranking

**Completed**: 2026-03-11 | **Phase**: P4b (Advanced RAG Retrieval) | **Tests**: 18 passed

### What
Second-stage re-ranking service that re-scores top-N hybrid search results using a cross-encoder model for higher-accuracy relevance ranking. Supports local cross-encoder (ms-marco-MiniLM-L-12-v2 via sentence-transformers) and cloud Cohere Rerank API.

### How
- `RerankerService` with `rerank()` (raw documents) and `rerank_search_hits()` (SearchHit integration)
- Local provider: `CrossEncoder.predict()` in thread pool executor (non-blocking async), sigmoid normalization to 0.0-1.0
- Cohere provider: httpx async POST to Cohere v2 rerank API
- Factory function `create_reranker_service()` with provider-based instantiation
- `RerankerDep` type alias in `src/dependency.py` for FastAPI DI
- `RerankResult` dataclass with index, relevance_score, document fields

### Why
Stage 3 of the 4-stage advanced RAG pipeline. Hybrid search (BM25+KNN) retrieves top-20 candidates; the cross-encoder re-ranks them to top-5 with much higher precision, since cross-encoders jointly encode the query-document pair rather than relying on independent embeddings.

### Key Files
- `src/services/reranking/service.py` — RerankerService (local + Cohere providers)
- `src/services/reranking/factory.py` — Factory function
- `src/config.py` — RerankerSettings (added `provider`, `cohere_api_key`)
- `src/exceptions.py` — RerankerError
- `src/dependency.py` — RerankerDep
- `tests/unit/test_reranker.py` — 18 tests
- `notebooks/specs/S4b.1_reranker.ipynb` — Interactive verification

### Dependencies Unlocked
- **S4b.5** (Unified retrieval pipeline) — can now use re-ranking stage
- **S6.3** (Retrieval node) — agent retrieval with re-ranking

---

## S9.6 — Paper Detail View

**Phase**: P9 (Frontend) · **Status**: done · **Date**: 2026-03-11

### What
Full paper detail page at `/papers/[id]` displaying complete metadata, paper sections, AI analysis, and related papers. The canonical view for any paper — search results, chat citations, and upload results all link here.

### How
- **Dynamic route** (`app/papers/[id]/page.tsx`) with `useReducer` for state management (loading/success/error/not-found)
- **PaperHeader** — title, full author list, date, category badges, arXiv/PDF external links, copy citation button
- **PaperSections** — collapsible sections (first 2 expanded by default), fallback for unparsed papers
- **PaperAnalysis** — tabbed display (Summary/Highlights/Methodology), reuses patterns from upload analysis, CTA when analysis unavailable
- **RelatedPapers** — horizontal scrollable cards linking to other paper detail pages
- **PaperDetailSkeleton** — skeleton loading state matching page layout
- **API client** (`lib/api/papers.ts`) — `getPaper()` and `getRelatedPapers()` with mock mode for dev
- Extended `types/paper.ts` with `PaperDetail`, `PaperSection`, `RelatedPapersResponse`

### Why
Central paper view page — every paper in the system needs a canonical detail page. Enables deep paper exploration beyond search results and chat citations.

### Key Design Decisions
- Used `useReducer` instead of multiple `useState` to satisfy ESLint `react-hooks/set-state-in-effect` rule
- `retryCount` state variable triggers re-fetch via effect dependency (clean retry pattern)
- Sections collapsible with `data-state` attribute for test assertions
- Analysis reuses the same Summary/Highlights/Methodology pattern from upload UI
- Related papers hidden gracefully on error (no error state shown)
- Mock mode built into API client for frontend development without backend

### Files
- `frontend/src/app/papers/[id]/page.tsx` — dynamic route page
- `frontend/src/components/paper/paper-header.tsx` — metadata display
- `frontend/src/components/paper/paper-sections.tsx` — collapsible sections
- `frontend/src/components/paper/paper-analysis.tsx` — tabbed analysis
- `frontend/src/components/paper/related-papers.tsx` — related papers list
- `frontend/src/components/paper/paper-detail-skeleton.tsx` — loading skeleton
- `frontend/src/lib/api/papers.ts` — paper API client
- `frontend/src/types/paper.ts` — extended with PaperDetail types
- Tests: 47 new tests across 6 test files

### Dependencies Unlocked
- **S9.9** (Export & citations) — can export citations from paper detail view

---

## S5.1 — Unified LLM Client (Ollama + Gemini)

**Phase**: P5 (RAG Pipeline) | **Status**: done | **Date**: 2026-03-11

### What
Provider-abstracted LLM client supporting both local (Ollama) and cloud (Gemini) inference. Defines a common `LLMProvider` protocol so callers never couple to a specific backend.

### How
- `LLMProvider` protocol (runtime_checkable) with `generate()`, `generate_stream()`, `health_check()`, `get_langchain_model()`, `close()`
- `OllamaProvider` — wraps Ollama HTTP API via httpx, NDJSON streaming, maps errors to LLMConnectionError/LLMTimeoutError/LLMServiceError
- `GeminiProvider` — wraps google.genai SDK, synchronous API calls run in async context, ConfigurationError on missing API key
- Factory: `create_llm_provider()` selects Gemini if API key set, else Ollama; `create_llm_providers()` returns dict of all available providers
- Response models: `LLMResponse` (text, model, provider, usage), `UsageMetadata` (tokens, latency), `HealthStatus`

### Why
Every downstream consumer (RAG chain, agents, analysis, chat) needs LLM access. The provider abstraction means switching from local dev (Ollama) to production (Gemini) requires zero code changes — just set an env var.

### Key Files
- `src/services/llm/provider.py` — protocol + response models
- `src/services/llm/ollama_provider.py` — OllamaProvider
- `src/services/llm/gemini_provider.py` — GeminiProvider
- `src/services/llm/factory.py` — factory functions
- `src/services/llm/__init__.py` — public exports
- `src/dependency.py` — LLMProviderDep added
- `tests/unit/test_llm_provider.py` — 27 tests
- `notebooks/specs/S5.1_llm_client.ipynb` — interactive demo

### Dependencies Unlocked
- **S4b.2** (HyDE retrieval) — needs LLM to generate hypothetical documents
- **S4b.3** (Multi-query retrieval) — needs LLM for query expansion
- **S5.2** (RAG chain) — needs LLM for answer generation
- **S6.1** (Agent state) — needs LLM provider reference
- **S8.2-S8.4** (Paper analysis) — needs LLM for summaries, highlights, methodology

---

## S9.7 — Reading Lists & Collections

### What Was Done
Built a complete collections/reading lists feature for the Next.js frontend. Users can create named collections, add/remove papers, reorder via drag-and-drop, and generate shareable links. All data persists in `localStorage` (no backend required for MVP).

### How It Was Done
- **Storage layer** (`collections.ts`): Pure functions wrapping `localStorage` with JSON serialization. CRUD operations for collections, add/remove/reorder for papers, base64-encoded share link generation and parsing. All functions throw on invalid input (empty name, non-existent ID).
- **Components**: `CollectionCard` (grid card with name, description, paper count, delete button), `AddToCollection` (popover dropdown listing existing collections + "Create New"), `PaperList` (list with HTML5 drag-and-drop via `draggable` attribute + `onDragStart/Over/Drop` handlers), `CreateCollectionDialog` (modal form with name + description inputs).
- **Pages**: `/collections` (list page with empty state CTA, grid layout, delete confirmation dialog) and `/collections/[id]` (detail page with paper list, share button using `navigator.clipboard`, back navigation).
- **Testing**: 57 tests across 6 test files using Vitest + React Testing Library. localStorage mocked with in-memory store. `next/navigation` mocked for routing.

### Why It Matters
Collections enable users to organize research papers into topical reading lists — a core workflow for researchers. The localStorage-based MVP provides immediate value without backend dependencies. The share link feature enables collaboration. The `AddToCollection` component can be integrated into search results and paper detail pages to make paper organization seamless from any context.

### Core Features
- Collection CRUD: create (with name + description), rename, delete (with confirmation dialog)
- Add/remove papers to/from collections via popover dropdown
- Drag-and-drop paper reordering (HTML5 API, no extra deps)
- Shareable collection links (base64-encoded URL with paper IDs)
- Empty states with CTAs for both collection list and detail views
- localStorage persistence across page refreshes
- 57 co-located Vitest tests (24 storage + 7 card + 5 popover + 8 paper list + 5 list page + 8 detail page)

### Key Files
- `frontend/src/types/collection.ts` — Collection, ShareData interfaces
- `frontend/src/lib/collections.ts` — localStorage CRUD + share link logic (161 lines)
- `frontend/src/lib/collections.test.ts` — 24 storage layer tests
- `frontend/src/components/collections/collection-card.tsx` — Collection card component
- `frontend/src/components/collections/add-to-collection.tsx` — Add-to-collection popover
- `frontend/src/components/collections/paper-list.tsx` — Draggable paper list
- `frontend/src/components/collections/create-collection-dialog.tsx` — Create/edit dialog
- `frontend/src/components/collections/index.ts` — Barrel export
- `frontend/src/app/collections/page.tsx` — Collections list page
- `frontend/src/app/collections/[id]/page.tsx` — Collection detail page

### Dependencies Unlocked
- No specs directly depend on S9.7, but the `AddToCollection` component is reusable in S9.3 (search) and S9.6 (paper detail) for cross-feature integration

---

## S4b.2 — HyDE (Hypothetical Document Embeddings)

**What**: Advanced retrieval technique that improves search recall by generating a hypothetical answer passage via LLM, embedding it, and using that embedding for KNN vector search against real documents.

**How**: `HyDEService` takes a user query, generates a 150-200 word academic passage via the LLM provider (temperature=0.3), embeds it using Jina embeddings, then runs KNN search on OpenSearch. If any step fails, it gracefully falls back to standard query embedding.

**Why**: Bridges the vocabulary gap between short user queries and long indexed document passages. A hypothetical answer is semantically closer to relevant documents than the original question, improving recall for complex research queries.

**Core Features**:
- `generate_hypothetical_document()` — LLM generates academic-style passage
- `retrieve_with_hyde()` — Full pipeline: generate → embed → KNN search → SearchHit results
- Graceful fallback at every stage (LLM failure, embedding failure)
- `HyDESettings` with `HYDE__` env prefix (enabled, max_tokens, temperature, timeout)
- Factory function + `HyDEDep` for FastAPI dependency injection
- 19 unit tests covering happy path, fallbacks, edge cases, prompt format

### Key Files
- `src/services/retrieval/hyde.py` — HyDEService + HyDEResult dataclass
- `src/services/retrieval/factory.py` — create_hyde_service factory
- `src/services/retrieval/__init__.py` — Package init
- `src/config.py` — HyDESettings (enabled, max_tokens, temperature, timeout)
- `src/dependency.py` — HyDEDep + get_hyde_service
- `tests/unit/test_hyde.py` — 19 tests
- `notebooks/specs/S4b.2_hyde.ipynb` — Interactive verification

### Dependencies Unlocked
- S4b.5 (Unified retrieval pipeline) — HyDE is one of the query expansion strategies

---

## S9.8 — Trends Dashboard

**Phase:** P9 (Frontend — Next.js) | **Status:** done

### What Was Done
Full research trends and analytics dashboard at `/dashboard` with interactive charts, stat cards, hot papers list, and trending topics widget. Uses **recharts** for data visualization with mock data fallback when the backend API is unavailable.

### How It Was Done
- Added `recharts` dependency to the frontend
- Created TypeScript types for dashboard data (`DashboardStats`, `CategoryCount`, `MonthlyCount`, `TrendingTopic`, `DashboardData`)
- Built API client (`lib/api/dashboard.ts`) with typed `getDashboardData()` function that tries the backend API and falls back to mock data on error
- Implemented 5 dashboard sub-components following existing shadcn/ui + Tailwind patterns:
  - `StatsCards` — 4 stat cards (Total Papers, Papers This Week, Categories, Most Active) with skeleton loading
  - `CategoryChart` — Recharts donut/pie chart with top 8 categories + "Other" grouping, legend, tooltips
  - `TimelineChart` — Recharts area chart with monthly publication counts, gradient fill, formatted month labels
  - `HotPapers` — Paper list with title links (internal + arXiv), authors (et al. truncation), category badges, dates
  - `TrendingTopics` — Tag cloud with relative sizing based on frequency counts
- Dashboard page shell (`app/dashboard/page.tsx`) orchestrates all widgets in a responsive grid with `useEffect` data fetching, loading states, and a "Using sample data" banner when mock data is active
- All components support dark mode via Tailwind CSS variables
- Tests mock recharts (jsdom can't render SVG) and verify rendering, loading, empty, and error states

### Why It Matters
- Provides researchers with at-a-glance analytics about the indexed paper collection
- Demonstrates the frontend's data visualization capabilities
- Navigation was already wired (S9.2) — this spec fills in the dashboard page content
- Mock data fallback enables frontend development independent of backend API readiness

### Core Features
- Responsive grid layout (1-col mobile, 2-col desktop)
- 4 stat overview cards with loading skeleton states
- Interactive donut chart for arXiv category distribution (top 8 + "Other")
- Area chart for monthly publication volume timeline
- Hot papers list with arXiv links and internal paper detail links
- Trending topics tag cloud with frequency-based sizing
- Automatic mock data fallback with "Using sample data" banner
- Dark mode support on all chart/widget components
- 42 new Vitest tests across 7 test files

### Key Files
- `frontend/src/app/dashboard/page.tsx` — Dashboard page
- `frontend/src/components/dashboard/stats-cards.tsx` — Stats overview cards
- `frontend/src/components/dashboard/category-chart.tsx` — Category donut chart
- `frontend/src/components/dashboard/timeline-chart.tsx` — Publication timeline chart
- `frontend/src/components/dashboard/hot-papers.tsx` — Hot papers list
- `frontend/src/components/dashboard/trending-topics.tsx` — Trending topics widget
- `frontend/src/lib/api/dashboard.ts` — API client with mock fallback
- `frontend/src/types/dashboard.ts` — TypeScript types

### Dependencies Unlocked
- S9.9 (Export & Citations) — does not depend on S9.8, but completes P9 alongside it

---

## S9.9 — Export & Citations
**Status:** done | **Phase:** P9 (Frontend) | **Completed:** 2026-03-11

### What Was Done
Client-side export system that lets users export paper citations in BibTeX, Markdown, and slide snippet formats. Supports both single-paper export (from paper detail view) and bulk export (from collection pages). Each format offers copy-to-clipboard and file download actions.

### How It Was Done
- **Pure utility functions** in `lib/export/formatters.ts` — no external dependencies, all formatting is client-side
- **BibTeX**: Generates `@article{}` entries with proper field formatting, auto-generated cite keys (first author last name + year + title word), special character escaping (`&`, `%`, `{`, `}`)
- **Markdown**: Structured output with heading, metadata block, abstract, and arXiv link
- **Slide Snippet**: Concise format with bold title, last names (truncated to 3 + "et al."), first sentence of abstract as key point
- **Clipboard/Download**: `clipboard.ts` wraps `navigator.clipboard.writeText` with fallback, `downloadFile` uses Blob + object URL pattern
- **ExportButton component**: Dropdown menu with icons per format, copy (with "Copied!" feedback) and download actions per row
- **TDD**: 28 tests across 3 test files (formatters, clipboard, component) — all written before implementation

### Why It Matters
Completes Phase 9 (Frontend). Export is essential for researchers who need to cite papers in LaTeX documents, Markdown notes, or presentation slides. Bulk export from collections enables batch reference list generation.

### Core Features
- `formatBibtex(paper)` — valid BibTeX with escaped special chars
- `formatMarkdown(paper)` — structured Markdown citation
- `formatSlideSnippet(paper)` — concise presentation-ready snippet
- `formatBulkBibtex(papers)` / `formatBulkMarkdown(papers)` — multi-paper export
- `copyToClipboard(text)` — clipboard with success/failure return
- `downloadFile(content, filename)` — blob-based file download (.bib, .md, .txt)
- `ExportButton` — reusable dropdown component with copy + download per format
- All exports include arXiv URLs

### Key Files
- `frontend/src/lib/export/formatters.ts` — Format functions (BibTeX, Markdown, Slide)
- `frontend/src/lib/export/clipboard.ts` — Copy-to-clipboard and file download utilities
- `frontend/src/lib/export/index.ts` — Public API barrel export
- `frontend/src/components/export/export-button.tsx` — ExportButton dropdown component
- `frontend/src/components/export/index.ts` — Component barrel export
- `frontend/src/components/paper/paper-header.tsx` — Modified: added ExportButton
- `frontend/src/app/collections/[id]/page.tsx` — Modified: added bulk ExportButton

### Dependencies Unlocked
- Completes Phase 9 (all 9 frontend specs done)
- No downstream specs depend on S9.9

---

## S4b.3 — Multi-Query Retrieval

### What Was Done
Implemented multi-query retrieval service that generates diverse query reformulations via LLM, runs parallel hybrid searches across all variations, deduplicates results by chunk_id, and fuses rankings using Reciprocal Rank Fusion (RRF).

### How It Was Done
- **Query generation**: LLM generates N (default 3) alternative formulations with temperature 0.7 for creative diversity; parses numbered or bulleted list output
- **Parallel search**: `asyncio.gather` runs hybrid search (BM25 + KNN + RRF) concurrently for all variations via `JinaEmbeddingsClient` + `OpenSearchClient`
- **Deduplication**: Tracks unique chunks by `chunk_id`, keeping the hit with the highest original score for metadata
- **RRF fusion**: Computes `score = Σ 1/(k + rank)` across all query result lists (k=60), sorts descending, returns top-K
- **Fallback**: On LLM failure or all search failures, falls back to single original query search
- **Configuration**: `MultiQuerySettings` with `MULTI_QUERY__` env prefix (enabled, num_queries, temperature, max_tokens, rrf_k)
- **Factory**: `create_multi_query_service()` in `src/services/retrieval/factory.py`

### Why It Matters
Multi-query retrieval addresses vocabulary mismatch — a single query may miss relevant documents that use different terminology. By searching with multiple reformulations and fusing results, recall improves significantly. This is stage 1 of the 4-stage advanced RAG pipeline (multi-query → hybrid search → re-rank → parent expansion).

### Core Features
- `generate_query_variations(query)` — LLM-powered query expansion (3-5 variations)
- `retrieve_with_multi_query(query, top_k)` — full pipeline: generate → parallel search → dedup → RRF → top-K
- Robust parsing of numbered lists, bullet points, and mixed formats
- Graceful fallback at every stage (LLM failure, partial search failure, embedding failure)
- `MultiQuerySettings` with configurable num_queries, temperature, rrf_k
- 22 unit tests covering all FRs, edge cases, and RRF math

### Key Files
- `src/services/retrieval/multi_query.py` — `MultiQueryService`, `MultiQueryResult`
- `src/config.py` — `MultiQuerySettings` (MULTI_QUERY__ env prefix)
- `src/services/retrieval/factory.py` — `create_multi_query_service()`
- `tests/unit/test_multi_query.py` — 22 unit tests
- `notebooks/specs/S4b.3_multi_query.ipynb` — Interactive verification

### Dependencies Unlocked
- **S4b.5** (Unified retrieval pipeline) — depends on S4b.1-S4b.4; S4b.3 is now done

---

## S4b.4 — Parent-Child Chunk Retrieval

**Phase**: P4b (Advanced RAG Retrieval) | **Status**: done | **Tests**: 25 passed

### What Was Done
Two-tier chunking strategy where papers are indexed as small child chunks (200 words) for precise retrieval, but at query time the parent chunk (600 words) is returned to provide richer LLM context. This improves retrieval precision (small chunks match better) while maintaining generation quality (large chunks give more context).

### How It Was Done
- `ParentChildChunker` reuses the existing `TextChunker` (S4.2) to create 600-word parent chunks, then splits each parent into overlapping 200-word child chunks (50-word overlap)
- Each `ChildChunk` stores a `parent_chunk_index` linking it to its parent
- `expand_to_parents()` takes child search results, groups by `parent_chunk_id`, fetches unique parents from OpenSearch, deduplicates, and preserves the best score per parent
- `prepare_for_indexing()` converts children into OpenSearch-ready dicts with deterministic `parent_chunk_id` format (`{arxiv_id}_parent_{chunk_index}`)
- Graceful fallback: if parent fetch fails during expansion, the child is returned as-is; orphan children without `parent_chunk_id` pass through unchanged
- Added `ChildChunk` and `ParentChildResult` Pydantic models to `src/schemas/indexing.py`
- TDD: 25 unit tests covering initialization validation, parent splitting, full paper chunking, section-based chunking, parent expansion with dedup, indexing preparation, and edge cases

### Why It Matters
Stage 4 of the 4-stage advanced RAG pipeline. Small chunks are better at matching specific queries (higher precision), but they lack the surrounding context needed for high-quality LLM generation. Parent expansion solves this by retrieving the full parent section at query time, giving the LLM 3x more context per retrieved result without sacrificing retrieval precision.

### Core Features
- `ParentChildChunker(parent_chunk_size, child_chunk_size, child_overlap, min_chunk_size)` — configurable two-tier chunking
- `create_parent_child_chunks()` — full paper → ParentChildResult (parents + children)
- `split_parent_into_children()` — single parent → list of ChildChunk with overlap
- `expand_to_parents()` — child search results → deduplicated parent chunks (best score preserved)
- `prepare_for_indexing()` — children → OpenSearch-ready dicts with `parent_chunk_id`
- Section-aware: inherits TextChunker's section-based chunking for parents
- Input validation: child_chunk_size < parent_chunk_size, child_overlap < child_chunk_size

### Key Files
- `src/services/indexing/parent_child.py` — ParentChildChunker class
- `src/schemas/indexing.py` — ChildChunk, ParentChildResult models (added)
- `src/services/indexing/__init__.py` — updated exports
- `tests/unit/test_parent_child.py` — 25 tests
- `notebooks/specs/S4b.4_parent_child.ipynb` — interactive verification

### Dependencies Unlocked
- **S4b.5** (Unified retrieval pipeline) — all S4b.1-S4b.4 now done, S4b.5 can proceed

---

## S4b.5 — Unified Advanced Retrieval Pipeline

**Status**: done | **Date**: 2026-03-11 | **Tests**: 21 passed

### What
Single entry point (`RetrievalPipeline`) that orchestrates the full advanced retrieval pipeline: multi-query expansion → hybrid search → re-ranking → parent chunk expansion → top-K. This is what downstream consumers (RAG chain, agents) call for document retrieval.

### How
- `RetrievalPipeline` class composes all S4b sub-services via dependency injection
- 5-stage pipeline: (1) parallel query expansion (multi-query + HyDE), (2) baseline hybrid search, (3) merge & deduplicate by chunk_id, (4) cross-encoder re-ranking, (5) parent chunk expansion
- Each stage independently enabled/disabled via `RetrievalPipelineSettings`
- Graceful degradation: any stage can fail without killing the pipeline — always returns results
- `RetrievalResult` dataclass captures results, metadata, expanded queries, timings, and stages executed
- Factory function `create_retrieval_pipeline()` wires all dependencies

### Why
- Provides a clean abstraction over 4 complex retrieval strategies
- RAG chain and agents don't need to know about individual retrieval components
- Configurable per-deployment: disable expensive stages in dev, enable all in production
- Graceful degradation ensures the system never fails silently — worst case is basic hybrid search

### Core Features
- **Parallel expansion**: Multi-query + HyDE run concurrently via `asyncio.gather`
- **Deduplication**: Merges results from multiple sources, keeps highest score per chunk_id
- **Timing metadata**: Per-stage timing in seconds for performance monitoring
- **Stage tracking**: `stages_executed` list shows exactly which stages ran
- **Configurable top-K**: `retrieval_top_k` (pre-rerank, default 20) and `final_top_k` (post-rerank, default 5)

### Files
- `src/services/retrieval/pipeline.py` — RetrievalPipeline, RetrievalResult
- `src/services/retrieval/factory.py` — create_retrieval_pipeline (added)
- `src/services/retrieval/__init__.py` — updated exports
- `src/config.py` — RetrievalPipelineSettings (added)
- `tests/unit/test_retrieval_pipeline.py` — 21 tests
- `notebooks/specs/S4b.5_retrieval_pipeline.ipynb` — interactive verification

### Dependencies Unlocked
- **S5.2** (RAG chain) — can now use the unified retrieval pipeline
- **S6.3** (Agent retrieval node) — can call `RetrievalPipeline.retrieve()`

---

## S5.2 — RAG Chain

### What
RAG (Retrieval-Augmented Generation) chain that orchestrates the full pipeline: retrieve relevant paper chunks via the advanced retrieval pipeline, build citation-enforcing prompts, and generate answers using the unified LLM client. Every response includes inline citations [1],[2] and a source list with paper titles, authors, and arXiv links.

### How
- **`RAGChain`** class with two main methods:
  - `aquery(query)` — single-shot: retrieve → prompt → generate → `RAGResponse`
  - `aquery_stream(query)` — streaming: retrieve → prompt → stream tokens → yield `[SOURCES]` JSON
- **Citation-enforcing prompts** (`prompts.py`): system prompt instructs LLM to use [N] notation; user prompt formats numbered context chunks with paper metadata (title, authors, arXiv ID)
- **Response models** (`models.py`): `RAGResponse` with `SourceReference` list, `RetrievalMetadata`, `LLMMetadata`
- **Empty results handling**: when no documents are retrieved, returns graceful "no relevant papers" message without calling the LLM
- **Factory function** (`factory.py`): `create_rag_chain(llm_provider, retrieval_pipeline)`
- **DI wiring**: `RAGChainDep` in `dependency.py` with full dependency chain (LLM → retrieval pipeline → all sub-services)

### Why
This is the core intelligence layer — the bridge between document retrieval and LLM generation. Without the RAG chain, the system cannot answer research questions with cited sources. Citation enforcement ensures every answer is grounded in real papers, which is the fundamental identity of PaperAlchemy.

### Core Features
- Retrieve → prompt → generate pipeline with citation enforcement
- Streaming support (SSE-ready via `aquery_stream`)
- Graceful empty-result handling (no LLM call when no papers found)
- Parameter forwarding (categories → retrieval, temperature → LLM)
- Fully injectable dependencies (LLM + retrieval pipeline)

### Key Files
- `src/services/rag/chain.py` — RAGChain class
- `src/services/rag/prompts.py` — SYSTEM_PROMPT + build_user_prompt
- `src/services/rag/models.py` — RAGResponse, SourceReference, RetrievalMetadata, LLMMetadata
- `src/services/rag/factory.py` — create_rag_chain
- `src/services/rag/__init__.py` — exports
- `src/dependency.py` — RAGChainDep, RetrievalPipelineDep (added)
- `tests/unit/test_rag_chain.py` — 14 tests
- `notebooks/specs/S5.2_rag_chain.ipynb` — interactive verification

### Dependencies Unlocked
- **S5.3** (Streaming responses) — can use RAGChain.aquery_stream() for SSE endpoint
- **S5.4** (Response caching) — can cache RAGResponse objects
- **S5.5** (Citation enforcement) — can parse/validate citations from RAGChain output
- **S6.6** (Generation node) — can use RAGChain for agent answer generation
- **S10.1** (Eval dataset) — can evaluate RAG chain quality

---

## S5.3 — Streaming Responses (SSE)

**What**: SSE streaming endpoint (`POST /api/v1/ask`) that exposes the RAG chain's `aquery_stream()` as a real-time Server-Sent Events stream, plus a non-streaming JSON fallback.

**How**:
- `src/schemas/api/ask.py` — `AskRequest` (query, top_k, categories, temperature, stream) with Pydantic validation; `AskResponse` for non-streaming JSON mode
- `src/routers/ask.py` — Single `POST /ask` endpoint that branches on `stream` flag:
  - **stream=true** (default): Returns `StreamingResponse` with `text/event-stream` content type. Parses RAGChain's `[SOURCES]` marker to split token events from source metadata
  - **stream=false**: Calls `RAGChain.aquery()` and returns JSON `AskResponse`
- SSE event types: `token` (text chunks), `sources` (paper metadata array), `done` (completion signal), `error` (on failures)
- Error handling: LLM/retrieval errors emit `event: error` in streaming mode, return HTTP 503 in non-streaming mode
- Uses existing `RAGChainDep` from dependency injection — fully mockable

**Why**: Enables responsive real-time UX in the Next.js frontend (S9.4 chat interface) and future chat API (S7.3). Token-by-token streaming improves perceived latency for research questions.

**Core Features**:
- Token-by-token SSE streaming with proper `event: type\ndata: json\n\n` format
- Source metadata (arxiv_id, title, authors, arxiv_url) sent as final `sources` event
- Non-streaming JSON fallback for simpler integrations
- Input validation: min 1 char query, temperature 0.0-2.0, positive top_k
- Graceful error handling: LLM failures as error events, empty results as message + empty sources
- No-cache headers + X-Accel-Buffering: no for proxy compatibility

### Key Files
- `src/routers/ask.py` — SSE streaming + non-streaming endpoint
- `src/schemas/api/ask.py` — AskRequest / AskResponse schemas
- `tests/unit/test_ask_router.py` — 22 tests (streaming, non-streaming, validation, errors, DI)
- `notebooks/specs/S5.3_streaming.ipynb` — interactive verification

### Dependencies Unlocked
- **S7.3** (Chat API) — can build on SSE streaming pattern for chat endpoint
- **S9.4** (Chat interface) — frontend can consume SSE events from `/api/v1/ask`

---

## S5.4 — Response Caching (Redis)

**Phase**: P5 — RAG Pipeline | **Status**: done | **Date**: 2026-03-11

### What Was Done
Redis-backed exact-match caching for RAG responses. Repeated identical queries return instantly from cache instead of re-running the full RAG pipeline (retrieve → prompt → generate), providing ~150-400x speedup on cache hits. Graceful degradation ensures the system works normally when Redis is unavailable.

### How It Was Done
- **CacheClient** class wraps `redis.asyncio.Redis` with SHA256-based deterministic key generation from normalized query parameters (query, model, top_k, categories)
- Keys are generated by JSON-serializing normalized params → SHA256 hash → `rag:response:{hash}` prefix
- `RAGResponse` Pydantic models are serialized/deserialized via `model_dump_json()` / `model_validate_json()`
- All Redis operations wrapped in try/except — failures are logged but never propagate (graceful degradation)
- Factory functions (`make_redis_client`, `make_cache_client`) create clients with ping verification, returning `None` if Redis is unavailable
- `CacheDep` Annotated type added to dependency injection for use in routers
- TTL configurable via `REDIS__TTL_HOURS` env var (default 24h = 86400s)

### Why It Matters
- Dramatically reduces latency for repeated queries (sub-ms vs seconds)
- Reduces load on LLM providers (Ollama/Gemini) and OpenSearch
- Graceful fallback means Redis outages never break the system
- Enables **S7.1** (Conversation Memory) which also uses Redis
- Foundation for any future caching needs (embeddings cache, search result cache)

### Core Features
- Deterministic SHA256 cache key generation with query normalization (strip + lowercase, sorted categories)
- `find_cached_response()` — cache lookup, returns `RAGResponse | None`
- `store_response()` — store with configurable TTL, fire-and-forget
- `invalidate()` — delete specific cache entry by query params
- `invalidate_all()` — scan and delete all `rag:response:*` keys
- `get_stats()` — return key count and memory usage
- `make_redis_client()` — async Redis client with ping check, returns `None` on failure
- `make_cache_client()` — CacheClient factory, returns `None` on failure

### Key Files
- `src/services/cache/__init__.py` — public exports
- `src/services/cache/client.py` — CacheClient class (key gen, lookup, store, invalidate, stats)
- `src/services/cache/factory.py` — make_redis_client, make_cache_client factories
- `src/dependency.py` — CacheDep, get_cache_client, set_cache_client
- `tests/unit/test_cache_client.py` — 19 tests (key gen, lookup, store, invalidation, stats, error paths)
- `tests/unit/test_cache_factory.py` — 6 tests (Redis factory, CacheClient factory, settings)
- `notebooks/specs/S5.4_caching.ipynb` — interactive verification

### Dependencies Unlocked
- **S7.1** (Conversation Memory) — can use Redis for session-based chat history

---

## S5.5 — Citation Enforcement

**Completed**: 2026-03-12 | **Phase**: P5 (RAG Pipeline) | **Depends on**: S5.2

### What
Post-processing layer that parses, validates, and formats citations in every RAG response. Ensures inline `[N]` references are mapped to real papers with title, authors, and arXiv links. Strips any LLM-generated "Sources:" sections and replaces them with a standardized format.

### How
- **`parse_citations(text)`** — Regex-based extraction of `[N]` indices from text; handles nested brackets `[[1]]`, deduplicates, ignores invalid (0, negative, non-numeric) and range-style `[1-3]`
- **`validate_citations(cited, sources)`** — Validates cited indices against source list; reports valid/invalid/uncited citations, coverage ratio, and overall validity
- **`format_source_list(sources)`** — Generates markdown source list with arXiv links; author formatting: ≤3 listed, >3 uses "et al."; year extracted from arxiv_id prefix
- **`enforce_citations(response)`** — Main entry point: parse → validate → strip LLM sources → format → append; returns `CitationResult` with formatted answer + validation metadata
- **`stream_with_citations(tokens, sources)`** — Async iterator wrapper for streaming; buffers tokens, strips LLM source sections, appends standardized sources at end; validation available after stream consumed
- **Integration**: Wired into `RAGChain.aquery()` (post-processes response) and `RAGChain.aquery_stream()` (wraps LLM stream)

### Why
PaperAlchemy's core identity is a research assistant that **always cites sources**. LLMs don't always follow citation prompts reliably, so a post-processing enforcement layer guarantees citation quality regardless of LLM compliance. This is the foundation for S6.6 (agent answer generation) which also requires citation enforcement.

### Core Features
- Citation parsing with edge case handling (nested brackets, duplicates, invalid indices)
- Citation validation with coverage metrics and hallucination detection
- Standardized markdown source list with arXiv links and author formatting
- LLM-generated "Sources:" section stripping and replacement
- Streaming support with lazy validation
- Data models: `CitationValidation`, `CitationResult`

### Files
- `src/services/rag/citation.py` — All citation enforcement logic (5 public functions + 2 data models)
- `src/services/rag/chain.py` — Updated to use citation enforcement in both aquery and aquery_stream
- `tests/unit/test_citation.py` — 36 tests (parser, validator, formatter, enforcer, streaming)
- `notebooks/specs/S5.5_citations.ipynb` — interactive verification

### Dependencies Unlocked
- **S6.6** (Generation Node) — Agent answer generation uses citation enforcement
- **Phase 5 complete** — All 5 RAG pipeline specs (S5.1-S5.5) are now done

---

## S6.1 — Agent State & Runtime Context

**Date**: 2026-03-12 | **Status**: done | **Tests**: 44 passing

### What
Foundational data structures for the LangGraph-based agentic RAG system: AgentState TypedDict, AgentContext dataclass, and 5 structured output Pydantic models.

### How
- **AgentState** (`state.py`): `TypedDict` with `total=False` so nodes return partial dicts. Uses `Annotated[list[AnyMessage], add_messages]` reducer for append-only message history. 10 fields: messages, original_query, rewritten_query, retrieval_attempts, guardrail_result, routing_decision, sources, grading_results, relevant_sources, metadata.
- **AgentContext** (`context.py`): `@dataclass(slots=True)` for per-request dependency injection. Holds live service clients (LLMProvider, RetrievalPipeline, CacheClient) and request-scoped config (model_name, temperature, top_k, guardrail_threshold, etc.). Optional services gracefully degrade when None.
- **Structured models** (`models.py`): GuardrailScoring (score 0-100), GradeDocuments (yes/no Literal), GradingResult (per-document), SourceItem (citation metadata), RoutingDecision (4-way Literal). All use Pydantic field constraints for validation.
- Factory functions: `create_initial_state(query)` and `create_agent_context(llm_provider=, **overrides)`.

### Why
Every Phase 6 agent node (S6.2–S6.8) depends on these types. The state schema defines how data flows through the LangGraph workflow, the context enables testable dependency injection, and the structured output models ensure validated LLM responses.

### Key Files
- `src/services/agents/models.py` — 5 Pydantic models for structured LLM output
- `src/services/agents/state.py` — AgentState TypedDict + create_initial_state()
- `src/services/agents/context.py` — AgentContext dataclass + create_agent_context()
- `src/services/agents/__init__.py` — Public re-exports
- `tests/unit/test_agent_models.py` — 21 tests (validation, boundaries, edge cases)
- `tests/unit/test_agent_state.py` — 12 tests (TypedDict structure, factory, reducer)
- `tests/unit/test_agent_context.py` — 11 tests (dataclass, defaults, overrides, protocol)
- `notebooks/specs/S6.1_agent_state.ipynb` — Interactive verification

### Dependencies Unlocked
- **S6.2** (Guardrail Node) — Uses AgentState, AgentContext, GuardrailScoring
- **S6.3** (Retrieval Node) — Uses AgentState, AgentContext, SourceItem
- **S6.4** (Grading Node) — Uses AgentState, GradeDocuments, GradingResult
- **S6.5** (Rewrite Node) — Uses AgentState, AgentContext
- **S6.6** (Generation Node) — Uses AgentState, AgentContext, SourceItem

---

## S6.2 — Guardrail Node

**Phase:** P6 (Agent System) | **Status:** done

### What Was Done
Domain relevance guardrail node — the first node in the agentic RAG LangGraph workflow. Scores user queries on a 0-100 scale using structured LLM output, then a conditional edge routes the graph to retrieval (on-topic) or rejection (off-topic).

### How It Was Done
- Created `src/services/agents/nodes/guardrail_node.py` with three components:
  - `get_latest_query()` — extracts last HumanMessage content from message list, raises ValueError if none found
  - `ainvoke_guardrail_step()` — async node that calls LLM with `with_structured_output(GuardrailScoring)` at temperature=0.0 for deterministic scoring. Falls back to score=50 on LLM failure (above default threshold=40, so queries proceed gracefully)
  - `continue_after_guardrail()` — sync conditional edge that compares score vs `context.guardrail_threshold`, returns "continue" or "out_of_scope"
- `GUARDRAIL_PROMPT` template rates queries for academic/scientific research relevance with a detailed scoring guide (0-19 off-topic, 80-100 directly about research)
- Uses `AgentContext` for dependency injection (llm_provider, model_name, guardrail_threshold)
- Created `src/services/agents/nodes/__init__.py` re-exporting all public symbols

### Why It Matters
Cheapest possible check before expensive retrieval operations. Prevents wasting compute on guaranteed-off-topic queries (pizza, sports, personal advice). Deterministic temperature=0.0 ensures consistent routing. Graceful fallback prevents pipeline crashes.

### Core Features
- Structured LLM output via `GuardrailScoring(score: int[0-100], reason: str)`
- Temperature=0.0 for deterministic, reproducible scoring
- Fallback score=50 on any LLM failure (conservative: above threshold, queries proceed)
- Configurable threshold (default=40) via AgentContext
- Conditional edge returns `"continue"` or `"out_of_scope"` for LangGraph routing
- Missing guardrail_result defaults to "continue" (never silently drops valid queries)

### Key Files
- `src/services/agents/nodes/guardrail_node.py` — node, edge, helper, prompt
- `src/services/agents/nodes/__init__.py` — public exports
- `tests/unit/test_guardrail_node.py` — 19 tests (all passing)
- `notebooks/specs/S6.2_guardrail.ipynb` — interactive verification

### Dependencies Unlocked
- **S6.7** (Agent Orchestrator) — Can wire guardrail as first node with conditional edge
- **S6.3** (Retrieval Node) — Next node in the graph after guardrail passes

---

## S6.4 — Document Grading Node

### What Was Done
Binary relevance grading node for the agentic RAG LangGraph workflow. Evaluates each retrieved document against the user's query using structured LLM output (`GradeDocuments`), filtering relevant documents for answer generation and routing to query rewriting when no relevant documents are found.

### How It Was Done
- Follows the same node pattern as S6.2 (guardrail): async `ainvoke_*` function + sync `continue_after_*` conditional edge
- Uses `llm.with_structured_output(GradeDocuments)` for binary "yes"/"no" relevance grading with reasoning
- Sequential document grading (respects LLM rate limits, maintains deterministic ordering)
- Per-document error handling: LLM failures for individual documents mark them as not-relevant rather than crashing the pipeline
- `continue_after_grading()` implements three-way routing: generate (has relevant docs), rewrite (no relevant docs, retries left), or force-generate (retries exhausted)
- Reuses `get_latest_query()` pattern from guardrail node for extracting the user query

### Why It Matters
Critical quality gate between retrieval and generation. Without grading, the LLM would attempt to generate answers from irrelevant chunks, producing hallucinated or off-topic responses. The rewrite loop (grade → rewrite → re-retrieve → grade) enables self-correcting retrieval with bounded retries.

### Core Features
- `ainvoke_grade_documents_step()` — grades each source via `GradeDocuments` structured output, returns `grading_results` + `relevant_sources`
- `continue_after_grading()` — conditional edge: "generate" if relevant sources exist, "rewrite" if empty + retries remain, "generate" if retries exhausted
- `GRADING_PROMPT` template with `{query}` and `{document}` placeholders
- Empty sources short-circuit (no LLM calls)
- Per-document LLM failure graceful degradation (marked not-relevant, pipeline continues)
- `GradingResult` with `document_id`, `is_relevant`, `score` (1.0/0.0), `reasoning`

### Key Files
- `src/services/agents/nodes/grade_documents_node.py` — node, edge, prompt
- `src/services/agents/nodes/__init__.py` — updated exports
- `tests/unit/test_grade_documents_node.py` — 17 tests (all passing)
- `notebooks/specs/S6.4_grading.ipynb` — interactive verification

### Dependencies Unlocked
- **S6.7** (Agent Orchestrator) — Can wire grading node after retrieval with conditional edge to rewrite or generate

---

## S6.3 — Retrieval Node (Tool-Based Document Retrieval)

### What Was Done
Retrieval node for the agentic RAG LangGraph workflow. Invokes the advanced retrieval pipeline (S4b.5) to fetch relevant documents for the user's query, converts `SearchHit` results into `SourceItem` objects, and populates the agent state with sources, attempt tracking, and pipeline metadata.

### How It Was Done
- Follows the established node pattern: async `ainvoke_retrieve_step(state, context)` returns a partial state dict for LangGraph merge
- `convert_search_hits_to_sources()` maps `SearchHit` fields to `SourceItem` with URL construction from `pdf_url` or `arxiv_id`
- Query priority: `rewritten_query` > `original_query` > last `HumanMessage` (supports rewrite-retry loop)
- Three-tier error handling: None pipeline, pipeline exception, and zero results — all return empty sources with descriptive error metadata
- Retrieval attempts incremented on every invocation (even failures) for bounded retry tracking
- Pipeline metadata (stages_executed, total_candidates, timings, query_used, num_results) stored under `state["metadata"]["retrieval"]`

### Why It Matters
The retrieval node is the MANDATORY tool call that ensures every research question is grounded in real papers from the knowledge base. Without it, the agent would answer from LLM memory alone, violating PaperAlchemy's core principle of citation-backed responses. The node bridges the advanced retrieval pipeline (multi-query + HyDE + re-rank + parent expansion) with the agent state machine.

### Core Features
- `ainvoke_retrieve_step()` — invokes `context.retrieval_pipeline.retrieve()` with current query and `top_k`
- `convert_search_hits_to_sources()` — maps SearchHit → SourceItem with arxiv_id-based URL construction, skips empty arxiv_id
- Retrieval attempt tracking (incremented per call, supports max 3 retries via orchestrator)
- Pipeline metadata enrichment for observability (stages, timings, candidates)
- Graceful degradation on None pipeline, exception, or empty results

### Key Files
- `src/services/agents/nodes/retrieve_node.py` — node implementation
- `src/services/agents/nodes/__init__.py` — updated exports
- `tests/unit/test_retrieve_node.py` — 15 tests (all passing)
- `notebooks/specs/S6.3_retrieval.ipynb` — interactive verification

### Dependencies Unlocked
- **S6.7** (Agent Orchestrator) — Can wire retrieval node after guardrail, feeding sources to grading node

---

## S6.6 — Answer Generation Node

**Status**: done | **Date**: 2026-03-12

### What Was Done
LangGraph agent node that generates citation-backed answers from relevant sources. Takes graded/relevant documents, constructs a citation-enforcing prompt, invokes the LLM, and post-processes output through S5.5 citation enforcement to ensure inline [N] references mapped to real papers with title, authors, and arXiv links.

### How It Was Done
- **Prompt engineering**: GENERATION_PROMPT template includes numbered source chunks with arXiv IDs and explicit citation instructions ([N] notation)
- **Citation pipeline**: SourceItem → SourceReference conversion (1-based indexing), then RAGResponse → `enforce_citations()` from S5.5 for validation and formatting
- **Graceful degradation**: No-sources fallback returns honest "I don't have papers on that topic" without calling LLM; LLM failures return user-friendly error message
- **Query selection**: Prefers `rewritten_query` over `original_query` over last HumanMessage, supporting the rewrite-re-retrieve loop
- **State contract**: Returns partial AgentState dict with AIMessage (formatted answer + source list) and citation validation metadata

### Why It Matters
This is the terminal node in the agentic RAG workflow — it produces the final user-facing answer. Citation enforcement ensures PaperAlchemy's core identity: every answer grounded in real papers with proper references. Enables S6.7 (Agent Orchestrator) to complete the full guardrail → retrieve → grade → rewrite → generate pipeline.

### Core Features
- `ainvoke_generate_answer_step()` — async node function following established agent node pattern
- `build_generation_prompt()` — constructs prompt with numbered source chunks and citation instructions
- `source_items_to_references()` — converts agent SourceItems to RAG SourceReferences with 1-based indices
- No-sources fallback (skips LLM, returns honest message)
- LLM error handling (returns user-friendly error, no crash)
- Citation validation metadata in state (is_valid, valid/invalid citations, coverage)
- Full integration with S5.5 `enforce_citations()` for post-processing

### Key Files
- `src/services/agents/nodes/generate_answer_node.py` — node implementation
- `src/services/agents/nodes/__init__.py` — updated exports
- `tests/unit/test_generate_answer_node.py` — 14 tests (all passing)
- `notebooks/specs/S6.6_generation.ipynb` — interactive verification

### Dependencies Unlocked
- **S6.7** (Agent Orchestrator) — All 5 nodes now complete (guardrail, retrieve, grade, rewrite, generate); can compile the full LangGraph StateGraph

---

## S6.5 — Query Rewrite Node

### What Was Done
Implemented the query rewrite agent node (`ainvoke_rewrite_query_step`) that optimizes user queries when document grading finds insufficient relevant results. Uses structured LLM output to expand abbreviations, add synonyms, and enrich queries with academic terminology, enabling a rewrite -> re-retrieve loop.

### How It Was Done
- **Structured LLM output**: `QueryRewriteOutput` Pydantic model (local to node) with `rewritten_query` and `reasoning` fields, used with `llm.with_structured_output()`
- **Semantic drift prevention**: Reads `state["original_query"]` instead of latest message, so repeated rewrites don't compound previous expansions
- **Temperature 0.3**: Controlled creativity for synonym/term expansion (between deterministic 0.0 and generative 0.7)
- **Graceful fallback**: On LLM failure, appends "research paper arxiv" to original query (simple keyword expansion)
- **Message integration**: Appends rewritten query as `HumanMessage` so retrieval node picks it up on next loop iteration
- **Metadata tracking**: Records original query, rewritten query, reasoning, and attempt number in `state["metadata"]["rewrite"]`

### Why It Matters
This node closes the feedback loop in the agentic RAG workflow. When initial retrieval doesn't find relevant papers, the grading node routes to rewrite, which optimizes the query and triggers re-retrieval. Combined with `max_retrieval_attempts=3` in AgentContext, this prevents both insufficient results and infinite loops. Enables S6.7 (Agent Orchestrator) to wire up the complete rewrite -> retrieve -> grade cycle.

### Core Features
- `ainvoke_rewrite_query_step()` — async node function following established agent node pattern
- `QueryRewriteOutput` — Pydantic structured output model for validated LLM responses
- `REWRITE_PROMPT` — academic-focused prompt template with synonym/abbreviation expansion instructions
- Original query preservation (prevents semantic drift on multi-rewrite)
- Keyword expansion fallback on LLM failure
- Metadata enrichment for observability

### Key Files
- `src/services/agents/nodes/rewrite_query_node.py` — node implementation
- `src/services/agents/nodes/__init__.py` — updated exports
- `tests/unit/test_rewrite_query_node.py` — 17 tests (all passing)
- `notebooks/specs/S6.5_rewrite.ipynb` — interactive verification

### Dependencies Unlocked
- **S6.7** (Agent Orchestrator) — All 5 agent nodes now complete (guardrail, retrieve, grade, rewrite, generate); ready to compile the full LangGraph StateGraph with conditional routing

---

## S6.7 — Agent Orchestrator (LangGraph)

### What Was Done
Built `AgenticRAGService` — a LangGraph `StateGraph` orchestrator that wires all 5 agent nodes (guardrail, retrieval, grading, rewrite, generation) into a complete agentic RAG workflow. The graph is compiled once at startup and reused for every request, exposing a single `ask()` entry point.

### How It Was Done
- **LangGraph 1.0 StateGraph**: Nodes registered as wrapper functions that extract `AgentContext` from `RunnableConfig["configurable"]["context"]`, then delegate to the existing node implementations (S6.2–S6.6)
- **Conditional edges**: `continue_after_guardrail` routes to retrieve or out_of_scope; `continue_after_grading` routes to generate or rewrite (loop back)
- **AgenticRAGResponse** Pydantic model: answer, sources (as `SourceReference`), reasoning_steps, metadata
- **Result extraction**: Static helpers `_extract_answer`, `_extract_sources`, `_extract_reasoning_steps` parse the final LangGraph state into structured output
- **Factory pattern**: `create_agentic_rag_service()` validates llm_provider and constructs the service for DI

### Why It Matters
This is the **keystone spec** of Phase 6 — it connects all individual agent nodes into a working pipeline. Without the orchestrator, nodes are isolated functions; with it, PaperAlchemy has a complete agentic RAG system that can: validate queries, retrieve papers, grade relevance, rewrite queries on failure, and generate citation-backed answers. Unlocks S6.8 (specialized agents) and is the foundation for the chat API (S7.x).

### Core Features
- Compiled LangGraph with 6 nodes: guardrail, out_of_scope, retrieve, grade_documents, rewrite_query, generate_answer
- Conditional routing: guardrail pass/reject, grade → generate vs rewrite loop
- Max retrieval attempts enforcement (prevents infinite rewrite loops)
- Out-of-scope handler with polite rejection message
- Per-request `AgentContext` with model, top_k, threshold overrides
- `AgenticRAGResponse` with answer, sources, reasoning_steps, metadata (including elapsed time)
- Graph compiled once at `__init__`, reused across all requests
- Graceful error handling: graph failures return error response instead of crashing

### Key Files
- `src/services/agents/agentic_rag.py` — `AgenticRAGService`, `AgenticRAGResponse`, node wrappers
- `src/services/agents/factory.py` — `create_agentic_rag_service()` factory
- `src/services/agents/__init__.py` — updated exports
- `tests/unit/test_agent_orchestrator.py` — 21 tests (all passing)
- `notebooks/specs/S6.7_orchestrator.ipynb` — interactive verification

### Dependencies Unlocked
- **S6.8** (Specialized Agents) — Can now build Summarizer, Fact-Checker, Trend Analyzer, Citation Tracker on top of the orchestrator
- **S7.1–S7.3** (Chatbot & Conversations) — Chat API can now invoke `AgenticRAGService.ask()` for research Q&A

---

## S7.1 — Conversation Memory

**Phase**: P7 (Chatbot & Conversations) | **Status**: done | **Tests**: 29 passing

### What
Session-based conversation history backed by Redis. Each chat session stores messages as a JSON list with a sliding window (last 20 messages by default) and 24-hour TTL refreshed on every interaction.

### How
- **ChatMessage** Pydantic model with `Literal["user", "assistant"]` role validation, auto-generated UTC timestamp, and optional metadata dict
- **ConversationMemory** class wrapping `redis.asyncio.Redis` — uses `RPUSH` for appends, `LRANGE` for reads, `LTRIM` for sliding window enforcement, and `EXPIRE` for TTL management
- **Redis key format**: `chat:session:{session_id}` — one list per session
- **Graceful degradation**: every Redis operation wrapped in try/except — failures log warnings but never propagate exceptions
- **Factory**: `make_conversation_memory()` reuses `make_redis_client()` from S5.4, returns `None` if Redis unavailable
- **DI**: `ConversationMemoryDep` Annotated type, `set_conversation_memory()`/`get_conversation_memory()` singleton pattern (same as CacheDep)

### Why
Enables follow-up Q&A in the chatbot. Without conversation memory, every question is treated as independent. With it, S7.2 (Follow-up Handler) can resolve coreferences ("What about its limitations?") by looking at prior context.

### Core Features
- `add_message(session_id, role, content, metadata)` — append + trim + TTL refresh
- `get_history(session_id, limit)` — retrieve full or last-N messages, skip corrupted entries
- `clear_session(session_id)` — delete all messages for a session
- `list_sessions()` — return all active session IDs via `SCAN`
- Sliding window: configurable `max_messages` (default 20), enforced via `LTRIM` after each add
- TTL: configurable `ttl_seconds` (default 86400 = 24h), refreshed on `add_message` and `get_history`

### Key Files
- `src/services/chat/memory.py` — `ChatMessage`, `ConversationMemory`, `make_conversation_memory()`
- `src/services/chat/__init__.py` — public exports
- `src/dependency.py` — `ConversationMemoryDep`, `set_conversation_memory()`, `get_conversation_memory()`
- `tests/unit/test_conversation_memory.py` — 29 tests (all passing)
- `notebooks/specs/S7.1_chat_memory.ipynb` — interactive verification

### Dependencies Unlocked
- **S7.2** (Follow-up Handler) — Can now access conversation history for coreference resolution
- **S7.3** (Chat API) — Can wire ConversationMemoryDep into chat endpoint

---

## S7.2 — Follow-up Handler

**What**: Context-aware follow-up Q&A handler that resolves coreferences in conversational queries using LLM + conversation history, then re-retrieves from the knowledge base every time.

**How**: Three-layer design:
1. **Heuristic detection** (`is_follow_up()`) — fast, no LLM call. Checks for coreference pronouns (it, its, they, this, that, etc.), continuation prefixes ("What about", "How about", "Also", "But", "Can you"), and short queries (< 5 words) with history present. Empty history → always False.
2. **LLM query rewriting** (`rewrite_query()`) — sends conversation history + follow-up query to LLM with a rewrite prompt. Returns self-contained query. Graceful fallback to original on LLM failure or empty response. Trims history to last N messages (default 10).
3. **Orchestration** (`FollowUpHandler`) — detect → rewrite (if needed) → RAGChain.aquery/aquery_stream → store user + assistant messages in ConversationMemory. Works without memory (treats all as standalone). Supports both non-streaming (`handle()`) and streaming (`handle_stream()`) modes.

**Why**: Enables natural multi-turn conversation. Without this, questions like "What about its limitations?" would fail because the RAG pipeline wouldn't know what "its" refers to. Coreference resolution transforms follow-ups into self-contained queries that the retrieval pipeline can handle.

### Core Features
- `is_follow_up(query, history)` — heuristic detection, no LLM overhead
- `rewrite_query(query, history, llm)` — LLM-based coreference resolution
- `FollowUpHandler.handle()` — full pipeline: detect → rewrite → RAG → store
- `FollowUpHandler.handle_stream()` — streaming variant, stores messages after stream consumed
- `FollowUpResult` model — tracks original_query, rewritten_query, is_follow_up, response
- Graceful degradation: works without ConversationMemory, LLM failures fall back to original query
- Always re-retrieves from knowledge base (never reuses old results)

### Key Files
- `src/services/chat/follow_up.py` — `is_follow_up()`, `rewrite_query()`, `FollowUpHandler`, `FollowUpResult`
- `src/services/chat/__init__.py` — updated exports
- `src/dependency.py` — `FollowUpHandlerDep`, `get_follow_up_handler()`
- `tests/unit/test_follow_up_handler.py` — 30 tests (all passing)
- `notebooks/specs/S7.2_follow_up.ipynb` — interactive verification

### Dependencies Unlocked
- **S7.3** (Chat API) — Can wire FollowUpHandlerDep into chat endpoint for multi-turn conversations

---

## S7.3 — Chat API

**Phase**: P7 (Chatbot & Conversations) | **Status**: done | **Date**: 2026-03-12

### What
Chat endpoint with session management for conversational research Q&A. Provides `POST /api/v1/chat` with session-based conversation history, follow-up detection, coreference resolution, and citation-backed responses in both streaming SSE and JSON modes.

### How
- **ChatRequest** model validates session_id (1-128 chars), query (1-500 chars, stripped), streaming flag, and optional RAG params (top_k, categories, temperature)
- **ChatResponse** includes answer, sources, session_id, is_follow_up flag, rewritten_query, and original query
- **Streaming mode** (default): Returns SSE with `metadata` event first (session info + follow-up status), then `token` events, then `sources`, then `done`. Errors emit `event: error`.
- **JSON mode**: Calls `FollowUpHandler.handle()` and returns `ChatResponse` directly
- **Session management**: `GET /chat/sessions/{id}/history` retrieves messages, `DELETE /chat/sessions/{id}` clears session (idempotent)
- **Graceful degradation**: All endpoints work when Redis/ConversationMemory is None — chat functions without memory, history returns empty, clear returns false

### Why
This is the primary API consumed by the Next.js chat interface (S9.4). It completes Phase 7 by exposing the follow-up handler (S7.2) and conversation memory (S7.1) as HTTP endpoints with proper SSE streaming for real-time token delivery.

### Core Features
- Streaming SSE with metadata → token → sources → done event sequence
- Follow-up detection with query rewrite metadata in responses
- Session history retrieval and clearing
- Graceful degradation when Redis unavailable
- Citation-backed responses (every answer includes paper citations)

### Files
- `src/routers/chat.py` — 3 endpoints (POST /chat, GET /history, DELETE /clear)
- `src/schemas/api/chat.py` — ChatRequest, ChatResponse, SessionHistoryResponse, SessionClearResponse, ChatMessageOut
- `src/main.py` — chat_router registered at `/api/v1`
- `tests/unit/test_chat_api.py` — 32 tests (all passing)
- `notebooks/specs/S7.3_chat_api.ipynb` — interactive verification

### Dependencies Unlocked
- **S12.2** (Telegram RAG Integration) — Can use chat API patterns for Telegram message handling

---

## S6.8 — Specialized Agents

### What Was Done
Four specialized agents for paper analysis — **Summarizer**, **Fact-Checker**, **Trend Analyzer**, and **Citation Tracker** — plus an **AgentRegistry** for dispatch. Each agent implements a common `SpecializedAgentBase` ABC with `run()` and `name`, uses the retrieval pipeline to find relevant papers, and calls LLMs with structured output for deterministic results.

### How It Was Done
- **Abstract base class** (`SpecializedAgentBase`) defines the protocol: `run(query, context, papers?)` returns a typed result model. Includes a shared `_retrieve_papers()` helper that converts `SearchHit` → `SourceItem`.
- **Pydantic structured output** — Each agent has its own result model (`SummarizerResult`, `FactCheckResult`, `TrendAnalysisResult`, `CitationTrackResult`) used with `llm.with_structured_output(Model)` for type-safe LLM responses.
- **Prompt engineering** — Domain-specific prompts instruct the LLM to analyze provided papers and produce structured analysis (e.g., 5-section summaries, claim verdicts, trend directions).
- **Graceful degradation** — Every agent handles: no papers available (retrieves its own), LLM failures (fallback responses with error messages), empty results.
- **AgentRegistry** — Simple name→instance mapping with `get(name)` dispatch and `agent_names` listing.
- **TDD** — 34 tests written first (Red), then implementation (Green), then lint/format (Refactor).

### Why It Matters
Completes Phase 6 (Agent System). Enables multi-agent collaboration where the orchestrator can delegate to specialized agents for different analysis tasks. Downstream specs (P8: Paper Upload & Analysis) can reuse the SummarizerAgent and other agents for paper analysis features.

### Core Features
- `SummarizerAgent`: Structured summaries (objective, methodology, key findings, contributions, limitations)
- `FactCheckerAgent`: Claim verification with verdicts (supported/contradicted/insufficient_evidence)
- `TrendAnalyzerAgent`: Research trend identification with direction (rising/stable/declining) and emerging topics
- `CitationTrackerAgent`: Citation relationship mapping (cited_by, references, influence summary)
- `AgentRegistry`: Name-based dispatch to any specialized agent
- All agents cite sources with arXiv links
- All agents use the retrieval pipeline (never answer from LLM memory alone)

### Files
- `src/services/agents/specialized/__init__.py` — Exports + AgentRegistry
- `src/services/agents/specialized/base.py` — SpecializedAgentBase ABC + SpecializedAgentResult
- `src/services/agents/specialized/summarizer.py` — SummarizerAgent + SummarizerResult
- `src/services/agents/specialized/fact_checker.py` — FactCheckerAgent + FactCheckResult + ClaimVerification
- `src/services/agents/specialized/trend_analyzer.py` — TrendAnalyzerAgent + TrendAnalysisResult + TrendItem
- `src/services/agents/specialized/citation_tracker.py` — CitationTrackerAgent + CitationTrackResult
- `src/services/agents/__init__.py` — Updated to export AgentRegistry
- `tests/unit/test_specialized_agents.py` — 34 tests (all passing)
- `notebooks/specs/S6.8_specialized.ipynb` — Interactive verification

### Dependencies Unlocked
- Phase 6 complete — all 8 agent specs done
- **P8** (Paper Upload & Analysis) — Can reuse SummarizerAgent for S8.2 paper summaries

---

## S8.1 — PDF Upload Endpoint

**Date**: 2026-03-12
**Status**: done
**Tests**: 21 (8 router + 13 service)

### What
`POST /api/v1/upload` endpoint that accepts PDF papers via multipart form, validates, parses with Docling, stores in PostgreSQL, chunks content, embeds via Jina, and indexes in OpenSearch.

### How
- **UploadService** (`src/services/upload/service.py`): Orchestrates full pipeline — validate → parse → save → chunk → embed → index
- **Upload Router** (`src/routers/upload.py`): FastAPI endpoint with DI for all services (PDFParser, PaperRepo, TextChunker, Embeddings, OpenSearch)
- **UploadResponse** (`src/schemas/api/upload.py`): Structured response with paper_id, chunks_indexed, warnings, indexing_status
- **Graceful degradation**: Paper is saved even if indexing fails (returns warning)

### Why
Entry point for Phase 8 (Paper Upload & Analysis). Users can upload their own PDFs, which get parsed, stored, and made searchable alongside arXiv papers. All downstream analysis specs (S8.2-S8.5) depend on this.

### Core Features
- PDF validation: extension check, size limit (50MB), magic bytes (`%PDF`)
- Metadata extraction: title from sections/filename, abstract from "Abstract" section or raw_text fallback
- Paper record: `arxiv_id=upload_<uuid>`, `source=upload`, full text + sections stored
- Chunking + embedding + indexing pipeline with graceful failure handling
- 413 for oversized, 422 for invalid PDF/parse errors

### Files
- `src/routers/upload.py` — POST /api/v1/upload endpoint
- `src/services/upload/service.py` — UploadService (validate, extract, process)
- `src/schemas/api/upload.py` — UploadResponse schema
- `tests/unit/test_upload_router.py` — 8 router tests
- `tests/unit/test_upload_service.py` — 13 service tests
- `notebooks/specs/S8.1_upload.ipynb` — Interactive verification

### Dependencies Unlocked
- **S8.2** (Paper Summary) — can generate AI summaries of uploaded papers
- **S8.3** (Key Highlights) — can extract highlights from uploaded papers
- **S8.4** (Methodology Analysis) — can analyze methodology of uploaded papers

---

## S8.3 — Key Highlights Extraction

### What
Extracts structured key highlights and insights from academic papers using the LLM client. Returns 5 categories: novel contributions, important findings, practical implications, limitations, and keywords.

### How
- **HighlightsService** takes an `LLMProvider` and `PaperRepository`, fetches the paper, prepares content (prioritizing abstract, results, conclusion), and sends a structured JSON prompt to the LLM
- **Content preparation** prioritizes high-value sections (results, conclusion, discussion) and truncates to ~4000 words
- **JSON parsing** with fallback: if LLM returns malformed output, generates placeholder highlights with a warning
- **API endpoint**: `POST /api/v1/papers/{paper_id}/highlights` with `?force=true` for cache bypass
- Handles edge cases: paper not found (404), insufficient content (422), LLM failure (503), abstract-only papers (warning)

### Why
Paper highlights give researchers a quick overview of a paper's key contributions without reading the full text. Combined with the summary (S8.2), this provides a comprehensive at-a-glance analysis.

### Core Features
- Structured output: 5 fields (novel_contributions, important_findings, practical_implications, limitations, keywords)
- Priority-aware content preparation (results/conclusion first)
- Malformed LLM output fallback with warning
- Abstract-only mode for papers without parsed sections
- Force regeneration via query parameter

### Files
- `src/services/analysis/highlights.py` — HighlightsService
- `src/schemas/api/analysis.py` — PaperHighlights + HighlightsResponse schemas
- `src/routers/analysis.py` — POST /api/v1/papers/{paper_id}/highlights endpoint
- `tests/unit/test_analysis_highlights.py` — 15 tests (service, schemas, endpoint)
- `notebooks/specs/S8.3_highlights.ipynb` — Interactive verification

### Dependencies Unlocked
- **S8.5** (Paper Comparison) — can compare highlights across papers

---

## S8.2 — AI-Generated Paper Summary

**What**: `SummarizerService` that generates structured paper summaries using the LLM provider. Given a paper's abstract and sections, produces a JSON-structured summary with five fields: objective, method, key_findings, contribution, and limitations.

**How**:
- `SummarizerService.extract_content()` pulls title, authors, abstract, and priority sections (intro, methodology, results, conclusion), truncating to ~4000 words for context window fit.
- `SummarizerService.summarize()` builds a system prompt requesting JSON output, calls `LLMProvider.generate()`, and parses the structured response via `_parse_summary()` with fallback for malformed LLM output.
- `POST /api/v1/papers/{paper_id}/summary` endpoint wired via `src/routers/analysis.py`, with proper error mapping (404, 422, 503).
- Added `InsufficientContentError` and `AnalysisError` to the exception hierarchy.

**Why**: Core building block for paper analysis features (S8.3-S8.5). Provides actionable, structured summaries instead of raw text, enabling the UI to render each section distinctly.

### Key Files
- `src/services/analysis/summarizer.py` — `SummarizerService` class
- `src/schemas/api/analysis.py` — `PaperSummary`, `SummaryResponse` models
- `src/routers/analysis.py` — Analysis endpoint router
- `src/exceptions.py` — `AnalysisError`, `InsufficientContentError`
- `tests/unit/test_analysis_summarizer.py` — 11 tests (service, schemas, endpoint, edge cases)
- `notebooks/specs/S8.2_summary.ipynb` — Interactive verification

### Dependencies Unlocked
- **S8.5** (Paper Comparison) — depends on S8.2

---

## S8.4 — Methodology & Findings Deep-Dive

**Date completed**: 2026-03-12

### What
`MethodologyService` that extracts structured methodology and findings analysis from academic papers using an LLM. Produces: research design, datasets used, baselines compared, key results with metrics, statistical significance, and reproducibility notes.

### How
- `MethodologyService` class (`src/services/analysis/methodology.py`) following the same pattern as S8.2/S8.3
- Pydantic models: `DatasetInfo`, `ResultEntry`, `MethodologyAnalysis`, `MethodologyResponse` in `src/schemas/api/analysis.py`
- LLM prompt enforces structured JSON output with 6 fields
- Content preparation prioritizes methodology/experiments/results sections, truncates to ~4000 words
- Fallback parsing for malformed LLM output with warning
- `POST /api/v1/papers/{paper_id}/methodology` endpoint with `?force=true` support

### Why
Enables deep-dive analysis of research methodology — essential for researchers evaluating paper rigor, comparing experimental setups, and understanding reproducibility. Complements S8.2 (summary) and S8.3 (highlights) to provide comprehensive paper analysis.

### Core Features
- 6-field structured output: research_design, datasets, baselines, key_results, statistical_significance, reproducibility_notes
- Handles theoretical papers (empty datasets/results) gracefully
- Abstract-only analysis with warning when sections unavailable
- JSON fallback parsing for malformed LLM output

### Key Files
- `src/services/analysis/methodology.py` — Service implementation
- `src/schemas/api/analysis.py` — DatasetInfo, ResultEntry, MethodologyAnalysis, MethodologyResponse
- `src/routers/analysis.py` — POST endpoint
- `tests/unit/test_analysis_methodology.py` — 16 tests (service, schemas, endpoint, edge cases)
- `notebooks/specs/S8.4_methodology.ipynb` — Interactive verification

### Dependencies Unlocked
- None directly — S8.5 (Paper Comparison) depends on S8.2, not S8.4

---

## S8.5 — Side-by-Side Paper Comparison

**Status**: done
**Date completed**: 2026-03-12

### What
`ComparatorService` that generates structured side-by-side comparisons of 2-5 academic papers using an LLM. Produces: methods comparison, results comparison, contributions comparison, limitations comparison, common themes, key differences, and an overall verdict.

### How
- `ComparatorService` class (`src/services/analysis/comparator.py`) following the S8.2/S8.3 analysis pattern
- Pydantic models: `ComparedPaper`, `PaperComparison`, `ComparisonRequest` (min 2, max 5 IDs), `ComparisonResponse` in `src/schemas/api/analysis.py`
- Multi-paper content extraction with per-paper labelling ("Paper 1: ...", "Paper 2: ...") and proportional truncation (~6000 words total)
- LLM prompt enforces structured JSON output with 7 comparison fields
- Fallback parsing for malformed LLM output with warning
- Input validation: deduplicates paper IDs, rejects <2 or >5 unique IDs
- `POST /api/v1/papers/compare` endpoint with JSON body `{"paper_ids": [...]}` and `?force=true` support

### Why
Enables researchers to systematically compare multiple papers — identifying methodological differences, complementary contributions, and shared limitations. This is a core feature for literature review workflows and the foundation for S19.1 (Multi-Paper Q&A).

### Core Features
- 7-field structured comparison: methods, results, contributions, limitations, common_themes, key_differences, verdict
- Supports 2-5 papers per comparison with automatic deduplication
- Proportional content truncation per paper to fit LLM context window
- Markdown-fenced JSON stripping and fallback parsing
- Full error handling: paper not found, insufficient content, LLM failure

### Key Files
- `src/services/analysis/comparator.py` — Service implementation
- `src/schemas/api/analysis.py` — ComparedPaper, PaperComparison, ComparisonRequest, ComparisonResponse
- `src/routers/analysis.py` — POST /api/v1/papers/compare endpoint
- `tests/unit/test_analysis_comparator.py` — 18 tests (service, schemas, endpoint, edge cases)
- `notebooks/specs/S8.5_comparison.ipynb` — Interactive verification (10 sections)

### Dependencies Unlocked
- **S19.1** (Multi-Paper Q&A) — depends on S6.7 + S8.5

---

## S9b.2 — Platform Dependency Declaration

**Phase**: P9b (Platform Foundation) | **Status**: done | **Date**: 2026-03-13

### What
Added 6 new Python dependency groups to `pyproject.toml` and corresponding environment variables to `.env.example`, enabling phases P14 (auth), P19 (slides), P22 (code gen), and P23 (audio/podcast).

### How
- Added `anthropic>=0.52.0` for Claude API access (P22 paper-to-code generation)
- Added `python-jose[cryptography]>=3.3.0` + `passlib[bcrypt]>=1.7.4` for JWT auth and password hashing (P14 user auth)
- Added `websockets>=14.0` for real-time comment transport (P14 discussions)
- Added `edge-tts>=7.0.0` for free Microsoft Edge text-to-speech (P23 podcast generation)
- Added `python-pptx>=1.0.0` for PowerPoint slide generation (P19 advanced AI)
- Updated `.env.example` with `ANTHROPIC__*`, `AUTH__*`, and `TTS__*` config sections

### Why
P9b is the "platform foundation" bridge phase. Before implementing features in P14-P23, all required dependencies must be declared and resolvable together. This ensures no version conflicts surface mid-implementation and that all downstream specs can import what they need.

### Core Features
- 6 new dependency groups (anthropic, auth, websockets, tts, slides)
- 3 new `.env.example` config sections (Anthropic, Auth, TTS)
- All dependencies resolve cleanly with `uv sync` (330 packages total)
- Note: passlib has compatibility issues with bcrypt 5.x — tests use `bcrypt` directly for hashing roundtrips

### Key Files
- `pyproject.toml` — Updated with new dependencies
- `.env.example` — Updated with new environment variables
- `tests/unit/test_platform_deps.py` — 9 tests (import checks + functional roundtrips)
- `notebooks/specs/S9b.2_platform_deps.ipynb` — Interactive verification

### Dependencies Unlocked
- **S5b.1** (Anthropic Provider) — depends on S5.1 + S9b.2
- **S14.1** (User Auth) — depends on S2.4 + S9b.1 + S9b.2 + S9b.4
- **S22.2** (Code Generation Agent) — depends on S22.1 + S6.7 + S9b.2
- **S23.2** (Text-to-Speech) — depends on S23.1 + S9b.2 + S9b.7

---

## S9b.1 — Alembic Migration Setup

**Phase**: P9b (Platform Foundation) | **Date**: 2026-03-13

### What
Initialized Alembic for async PostgreSQL database migrations. Created the initial migration for the `papers` table and added Makefile convenience commands.

### How
- **alembic.ini**: Standard config with `script_location = alembic`, DB URL set programmatically (not hardcoded)
- **alembic/env.py**: Custom async env.py using `async_engine_from_config` + `asyncpg`, imports `Base.metadata` from `src.db.base`, imports all models via `src.models` for autogenerate detection, reads DB URL from `src.config.get_settings()`
- **Initial migration**: `7fa2bf4a9f17_create_papers_table.py` — creates `papers` table with all 15 columns (id, arxiv_id, title, authors, abstract, categories, published_date, updated_date, pdf_url, pdf_content, sections, parsing_status, parsing_error, created_at, updated_at) plus 3 indexes
- **Makefile targets**: `db-migrate msg="..."` (autogenerate), `db-upgrade` (apply head), `db-downgrade` (revert -1)

### Why
Required for all future ORM models (User, Comment, Vote, Note, Avatar, Collection, etc.) in P14+. Replaces the `create_tables()` approach with proper version-controlled migrations.

### Key Files
- `alembic.ini` — Alembic config
- `alembic/env.py` — Async migration environment
- `alembic/versions/7fa2bf4a9f17_create_papers_table.py` — Initial migration
- `Makefile` — db-migrate, db-upgrade, db-downgrade targets
- `tests/unit/test_alembic_setup.py` — 24 tests (structural/config validation)
- `notebooks/specs/S9b.1_alembic.ipynb` — Interactive verification

### Dependencies Unlocked
- **S14.1** (User Auth) — depends on S2.4 + S9b.1 + S9b.2 + S9b.4

---

## S9b.3 — Frontend Infrastructure Dependencies

### What
Added 10 npm packages to the Next.js frontend required by P13–P23 feature phases: markdown rendering, animations, global state, toasts, command palette, and form validation. Also added jsdom polyfills for test environment compatibility.

### How
- Installed `react-markdown` + `remark-gfm` + `rehype-highlight` (markdown rendering for chat)
- Installed `framer-motion` (page transitions, micro-interactions)
- Installed `zustand` (lightweight global state management)
- Installed `sonner` (toast notifications)
- Installed `cmdk` (Cmd+K command palette)
- Installed `react-hook-form` + `@hookform/resolvers` + `zod` (type-safe form validation)
- Added `ResizeObserver` and `Element.scrollIntoView` polyfills to Vitest setup for jsdom compatibility (needed by cmdk, recharts)
- TDD: wrote 16 smoke tests verifying imports, rendering, store creation, and schema validation before installing packages

### Why
This is the frontend dependency foundation for all P13–P23 UI features. Without these packages, no downstream spec (chat polish, animations, auth forms, command palette, etc.) can begin implementation. The jsdom polyfills also fix test environment issues that would block component testing for cmdk and similar libraries.

### Core Features
- Markdown rendering pipeline (react-markdown + GFM + syntax highlighting)
- Framer Motion animation library for React 19 + Next.js 16
- Zustand state management (SSR-compatible, minimal boilerplate)
- Sonner toast notifications
- cmdk command palette component
- React Hook Form + Zod type-safe validation
- jsdom polyfills (ResizeObserver, scrollIntoView) in test setup

### Key Files
- `frontend/package.json` — 10 new dependencies added
- `frontend/pnpm-lock.yaml` — lockfile updated
- `frontend/src/test/setup.ts` — jsdom polyfills for ResizeObserver + scrollIntoView
- `frontend/src/lib/infra-deps.test.tsx` — 16 smoke tests (import + functional)

### Dependencies Unlocked
- **S9b.4** (Frontend Auth Infrastructure) — auth context, token storage, protected routes
- **S9b.5** (Frontend UI Primitives) — Dialog, Tabs, Toast container, Command palette, Sheet
- **S13.3** (Chat UX Polish) — markdown rendering, citation cards, typing indicator (also needs S9.4)
- **S13.6** (Micro-Interactions) — page transitions, skeleton loaders, toast animations (also needs S9.1)

---

## S9b.5 — Missing UI Primitives
**Phase**: P9b (Platform Foundation) | **Status**: Done | **Date**: 2026-03-13

### What Was Done
Added 11 shadcn/ui (base-nova style) components to fill the UI primitive gaps needed by P13–P23 feature phases. Each component has a co-located Vitest test file with accessibility and interaction coverage.

### How It Was Done
- Installed 7 Radix UI packages (`@radix-ui/react-dialog`, `@radix-ui/react-dropdown-menu`, `@radix-ui/react-popover`, `@radix-ui/react-tabs`, `@radix-ui/react-checkbox`, `@radix-ui/react-tooltip`, `@radix-ui/react-avatar`)
- Leveraged already-installed `cmdk` and `sonner` packages (from S9b.3)
- All components follow the existing pattern: `"use client"` directive, `data-slot` attributes, `cn()` utility, CVA variants (Sheet), Radix primitives underneath
- TDD approach: wrote 43 tests first (all failing), then implemented components to pass
- Used `@testing-library/react` + `@testing-library/user-event` for interaction tests

### Why It Matters
These are the building blocks for every upcoming UI feature. Without Dialog, Tabs, Command, Sheet, and Tooltip, P13 (UI Enhancement) and P14 (Community) cannot build their interfaces. The Command component specifically enables the Cmd+K palette (S13.5), Sheet enables mobile drawers (S13.7), and Avatar/Tooltip/DropdownMenu enable the community features (S14.6).

### Core Features
- **Dialog/Modal** — accessible overlay with focus trap, escape-to-close, header/footer sections
- **Dropdown Menu** — items, checkboxes, radio groups, sub-menus, separators, keyboard navigation
- **Popover** — floating anchored content with auto-placement
- **Tabs** — tabbed content panels with keyboard navigation and active state styling
- **Textarea** — styled multi-line input with consistent design tokens
- **Checkbox** — checked/unchecked/indeterminate states with Radix accessibility
- **Tooltip** — hover/focus tooltips with delay and collision avoidance
- **Avatar** — image with initials fallback on load failure
- **Toast (Sonner)** — notification container with themed toast options
- **Command (cmdk)** — command palette with search filtering, groups, empty state
- **Sheet** — slide-in panel from any edge (top/right/bottom/left) for mobile navigation

### Key Files
- `frontend/src/components/ui/dialog.tsx` + `dialog.test.tsx` (4 tests)
- `frontend/src/components/ui/dropdown-menu.tsx` + `dropdown-menu.test.tsx` (3 tests)
- `frontend/src/components/ui/popover.tsx` + `popover.test.tsx` (3 tests)
- `frontend/src/components/ui/tabs.tsx` + `tabs.test.tsx` (4 tests)
- `frontend/src/components/ui/textarea.tsx` + `textarea.test.tsx` (5 tests)
- `frontend/src/components/ui/checkbox.tsx` + `checkbox.test.tsx` (4 tests)
- `frontend/src/components/ui/tooltip.tsx` + `tooltip.test.tsx` (3 tests)
- `frontend/src/components/ui/avatar.tsx` + `avatar.test.tsx` (3 tests)
- `frontend/src/components/ui/sonner.tsx` + `sonner.test.tsx` (2 tests)
- `frontend/src/components/ui/command.tsx` + `command.test.tsx` (3 tests)
- `frontend/src/components/ui/sheet.tsx` + `sheet.test.tsx` (4 tests)
- `frontend/package.json` — 7 new Radix UI dependencies

### Dependencies Unlocked
- **S13.5** (Sidebar Navigation) — needs Command for Cmd+K palette, Tooltip for icon hints (also needs S9.2)
- **S14.6** (Community Frontend) — needs Avatar, Tooltip, DropdownMenu for user profiles and interactions (also needs S14.1-S14.5, S9.6)

---

## S9b.4 — Frontend Auth Infrastructure

**Status**: Done | **Phase**: P9b (Platform Foundation) | **Date**: 2026-03-13

### What Was Done
Built the complete frontend authentication infrastructure: auth types with Zod runtime validation, a zustand-based auth store with login/signup/logout actions, an API client auth interceptor (Bearer token injection + 401 auto-logout), a ProtectedRoute wrapper component, and three page shells for /login, /signup, and /forgot-password — all with react-hook-form + zod validation.

### How It Was Done
- **Auth types** (`types/auth.ts`): Zod schemas for User, LoginRequest/Response, SignupRequest/Response, ForgotPasswordRequest — TypeScript types inferred from schemas via `z.infer<>` for zero drift between runtime validation and compile-time types.
- **Auth store** (`lib/auth/store.ts`): Zustand store with `create<AuthState & AuthActions>()`. Token persisted in `sessionStorage` (SSR-safe — guarded by `typeof window` checks). Login/signup call backend API, parse response with Zod, and atomically update state.
- **Interceptor** (`lib/auth/interceptor.ts`): `createAuthenticatedFetch()` wraps any fetch function, auto-attaching `Authorization: Bearer <token>` header when authenticated, clearing auth state on 401 responses (but NOT on 403 — permission denied ≠ unauthenticated).
- **ProtectedRoute** (`components/auth/protected-route.tsx`): Client component using `useEffect` for SSR-safe redirect. Shows loading spinner during auth check, redirects to `/login?redirect={currentPath}` when unauthenticated, renders children when authenticated.
- **Page shells**: Login, signup, and forgot-password pages using react-hook-form + zodResolver with `noValidate` (Zod handles all validation, not HTML5). Forms show field-level errors, disable submit while loading, and use sonner for error toasts.
- **Header integration**: Sign in/Sign out button added to app header, conditionally rendered based on `isAuthenticated`.

### Why It Matters
This is the **client-side auth foundation** for all authenticated features. P14 (Community — comments, voting, profiles), P15 (Annotations — highlights, notes), P17 (Collaboration — shared collections), and P18 (Integrations — API keys) all require user authentication. Without this infrastructure, none of those phases can build their frontends. The interceptor pattern ensures every API call automatically includes credentials once the user is logged in.

### Core Features
- 7 Zod schemas with runtime validation for all auth API payloads
- Zustand auth store with login/signup/logout/clearAuth/setToken/refreshUser actions
- Token persistence in sessionStorage (SSR-safe)
- API interceptor with auto Bearer token injection and 401 auto-logout
- ProtectedRoute wrapper with loading state and redirect preservation
- Login page with email + password validation
- Signup page with name, email, password (must include number), confirm password, optional affiliation
- Forgot password page with email-only form and success feedback
- Header Sign in / Sign out integration
- 56 Vitest tests across 7 test files

### Key Files
- `frontend/src/types/auth.ts` — Zod schemas + TypeScript types (55 lines)
- `frontend/src/lib/auth/store.ts` — Zustand auth store (131 lines)
- `frontend/src/lib/auth/interceptor.ts` — Authenticated fetch wrapper (21 lines)
- `frontend/src/lib/auth/index.ts` — Barrel export
- `frontend/src/components/auth/protected-route.tsx` — Route guard component (40 lines)
- `frontend/src/app/(auth)/login/page.tsx` — Login page (98 lines)
- `frontend/src/app/(auth)/signup/page.tsx` — Signup page (152 lines)
- `frontend/src/app/(auth)/forgot-password/page.tsx` — Forgot password page (85 lines)
- `frontend/src/components/layout/header.tsx` — Updated with auth UI

### Dependencies Unlocked
- **S14.1** (User Authentication & Profiles) — needs S9b.4 for frontend auth types, store, and page shells to wire backend auth API into

---

## S9b.7 — Docker Platform Services

**Phase:** P9b (Platform Foundation) | **Status:** Done | **Date:** 2026-03-13

### What
Added Docker Compose services for MinIO (S3-compatible object storage) and a Docker-in-Docker sandbox, plus updated the API service with platform environment variables (MinIO, sandbox, Anthropic, TTS, Auth).

### How
- **MinIO** service (profile: `platform`) on ports 9100/9101, auto-creates `paperalchemy` bucket via entrypoint
- **Sandbox** service (Docker-in-Docker `docker:27-dind`, privileged) for sandboxed code execution
- API service env updated with `MINIO__*`, `SANDBOX__*` vars pointing to internal service hostnames
- `.env.example` updated with MinIO and sandbox vars for local development
- Makefile `platform` target (`make platform`) to bring up platform services
- Down/clean targets updated to include `--profile platform`

### Why
P22 (Paper-to-Code) needs a sandboxed execution environment for generated code, and P23 (Audio/Podcast) needs S3-compatible storage for audio files. This spec provides the infrastructure layer that those features will build on.

### Tests
- 27 tests in `tests/unit/test_docker_platform_services.py` — validates compose.yml structure, service configs, API env vars, .env.example vars, Makefile target, and volume declarations

### Key Files
- `compose.yml` — Added `minio` and `sandbox` services, `minio_data`/`sandbox_data` volumes, API env vars
- `.env.example` — Added `MINIO__*` and `SANDBOX__*` sections
- `Makefile` — Added `platform`/`up-platform` targets, updated down/clean with `--profile platform`

### Dependencies Unlocked
- **S11b.1** (Platform Monitoring) — health checks for MinIO, sandbox
- **S11b.2** (Platform Ops Guide) — MinIO admin, sandbox lifecycle docs
- **S11b.3** (Platform Deploy) — Cloud Storage replaces MinIO in prod
- **S22.3** (Code Sandbox) — uses sandbox service
- **S23.2** (Text-to-Speech) — stores audio in MinIO

---

## S9b.6 — Collections Backend API

**Date**: 2026-03-13 | **Phase**: P9b (Platform Foundation)

### What
Backend API for paper collections — migrating from frontend localStorage to a proper database-backed system with full CRUD operations and paper management.

### How
- **Collection model** (`src/models/collection.py`): UUID PK, name, description, nullable user_id (pre-auth), timestamps, M2M relationship to Papers via `collection_papers` association table with CASCADE deletes
- **CollectionRepository** (`src/repositories/collection.py`): Async CRUD (create, get_by_id, list_all, update, delete), paper operations (add_paper idempotent, remove_paper), count with optional user_id filter
- **Pydantic schemas** (`src/schemas/collection.py`): CollectionCreate, CollectionUpdate, CollectionResponse (with paper_count), CollectionDetailResponse (with papers list), CollectionPaperAction
- **REST API** (`src/routers/collections.py`): 7 endpoints under `/api/v1/collections` — GET list, POST create (201), GET detail, PUT update, DELETE (204), POST add paper, DELETE remove paper
- **DI wiring**: CollectionRepoDep in `src/dependency.py`, router registered in `src/main.py`

### Why
P15 (Annotations & Notebooks) and P17 (Shared Collections) both require server-side collection management. Moving collections to the backend enables multi-device sync, sharing, and team collaboration features.

### Tests
- 48 tests in `tests/unit/test_collections.py` — covers model structure, repository CRUD, paper operations (add/remove/idempotent), schema validation, and all 7 API endpoints with error cases

### Key Files
- `src/models/collection.py` — Collection ORM model + `collection_papers` M2M table
- `src/repositories/collection.py` — CollectionRepository (async CRUD + paper ops)
- `src/schemas/collection.py` — Pydantic request/response schemas
- `src/routers/collections.py` — REST API endpoints
- `src/dependency.py` — CollectionRepoDep injection
- `src/main.py` — Router registration

### Dependencies Unlocked
- **S17.1** (Shared Collections) — team-based collection sharing
- **S15.x** (Annotations) — collection-scoped annotations

---

## S5b.1 — Anthropic Claude LLM Provider

**What:** Added `AnthropicProvider` implementing the `LLMProvider` protocol for Anthropic Claude models (Opus/Sonnet/Haiku).

**How:**
- Created `AnthropicSettings` (Pydantic `BaseSettings`, `ANTHROPIC__` env prefix) with api_key, model, temperature, max_tokens, timeout
- Implemented `AnthropicProvider` in `src/services/llm/anthropic_provider.py` using `anthropic.AsyncAnthropic` for async generation
- Non-streaming via `messages.create()`, streaming via `messages.stream()` context manager
- LangChain integration via `ChatAnthropic` from `langchain-anthropic`
- Error mapping: `AuthenticationError` → `ConfigurationError`, `APITimeoutError` → `LLMTimeoutError`, `APIConnectionError` → `LLMConnectionError`
- Updated factory with priority: Anthropic → Gemini → Ollama
- Added `langchain-anthropic` dependency

**Why:** Enables Claude as an LLM provider for agentic code generation (P22) and multi-provider routing (S5b.2). Claude excels at code generation tasks where Gemini is used for research Q&A.

**Core features:** generate, generate_stream, health_check, get_langchain_model, close, usage extraction

**Tests:** 41 tests (11 AnthropicProvider + 4 updated factory tests + 26 existing)

**Files:** `src/services/llm/anthropic_provider.py`, `src/config.py`, `src/services/llm/factory.py`, `src/services/llm/__init__.py`, `tests/unit/test_llm_provider.py`

---

## S5b.2 — Multi-Provider LLM Routing

**What:** `LLMRouter` class that routes LLM calls to optimal providers based on task type. Research Q&A goes to Gemini, code generation to Claude, local dev to Ollama. Supports configurable per-task routing tables, automatic fallback chains (primary fails -> try secondary), and per-provider cost tracking with token counters.

**How:** `TaskType` enum defines 6 task categories (RESEARCH_QA, CODE_GENERATION, SUMMARIZATION, GRADING, QUERY_REWRITE, GENERAL). `LLMRoutingSettings` maps each task type to a provider name via env vars (`LLM_ROUTING__*`). `LLMRouter` wraps multiple `LLMProvider` instances and implements the same protocol, so callers just specify the task type. On failure, the fallback chain tries each provider in `fallback_order` until one succeeds. `ProviderUsageStats` tracks requests, tokens, and failures per provider.

**Why:** Different LLM providers excel at different tasks — Gemini for research Q&A, Claude for code generation, Ollama for zero-cost local dev. Routing optimizes quality, cost, and latency per task. Fallback chains ensure resilience. This is the foundation for S5c.3 (multi-tier routing by complexity) and S10c.2 (A/B testing).

**Core features:** TaskType enum, task-based routing, fallback chain, cost tracking (get_usage_stats/reset_usage_stats), health check aggregation, LangChain model routing, factory + DI integration

**Tests:** 34 tests — routing logic (8), generate delegation (4), stream delegation (2), fallback chain (4), cost tracking (5), health check (2), langchain model (2), close (1), settings (2), stats (1), factory (1), enum (2)

**Files:** `src/services/llm/router.py`, `src/config.py`, `src/services/llm/factory.py`, `src/dependency.py`, `tests/unit/test_llm_router.py`

---

## S13.2 — Design System Overhaul

**What:** Complete overhaul of the default shadcn/ui design system with a premium indigo/violet aesthetic. Replaced the grayscale palette with rich indigo-600 primary and violet-500 accent colors (oklch), added glassmorphism card components, configured Inter + Plus Jakarta Sans typography, gradient accent utilities, a 6-tier elevation/shadow system, accessible focus rings, and 200ms transitions with prefers-reduced-motion support.

**How:** Updated `globals.css` with new oklch-based color tokens for both light and dark modes. Created `GlassCard` component with `backdrop-filter: blur(24px)` and semi-transparent backgrounds. Configured Inter (body) and Plus Jakarta Sans (headings) via `next/font/google` with `font-display: swap`. Defined gradient CSS custom properties (`--gradient-primary`, `--gradient-accent`, `--gradient-surface`) and utility classes (`.bg-gradient-primary`, `.text-gradient`). Shadow scale (xs → 2xl) defined as CSS custom properties mapped to Tailwind via `@theme`. Interactive elements get `transition: all 200ms ease-in-out` with `prefers-reduced-motion: reduce` override. Added `--success` and `--warning` semantic tokens.

**Why:** Establishes PaperAlchemy's visual identity across the entire frontend. All P13 specs (S13.1 landing page, S13.3 chat polish, S13.4 search polish, S13.5 sidebar, S13.6 micro-interactions, S13.7 mobile) consume these design tokens. Without a cohesive design system, each page would look inconsistent. The glassmorphism, gradients, and elevation system create a premium, modern aesthetic that differentiates PaperAlchemy from basic academic tools.

**Core features:** Indigo/violet oklch palette (light + dark), glassmorphism GlassCard component (default + elevated), Inter + Plus Jakarta Sans fonts, gradient utilities (.bg-gradient-primary, .text-gradient), shadow elevation scale (xs → 2xl), success/warning semantic colors, 200ms interactive transitions, prefers-reduced-motion support, WCAG AA contrast-compliant

**Tests:** 28 tests — color palette (5), glassmorphism (4), typography (4), gradients (4), elevation (3), focus rings & transitions (3), component regression (5)

**Files:** `frontend/src/app/globals.css`, `frontend/src/app/layout.tsx`, `frontend/src/components/ui/glass-card.tsx`, `frontend/src/app/design-system.test.tsx`

---

## S13.1 — Premium Landing Page

**What:** Built a premium, animated landing page for PaperAlchemy with five distinct sections: hero with animated mesh gradient, 6-card feature showcase grid, animated stats counter, use cases section, and a navigation footer. Replaces the previous minimal placeholder page.

**How:** Created 5 modular React components in `frontend/src/components/landing/`: HeroSection (mesh gradient via CSS `@keyframes` animation, CTA buttons linking to /chat and /search), FeatureGrid (6 cards with Lucide icons and hover-lift `translateY` + shadow transitions), StatsCounter (client component using `IntersectionObserver` + `requestAnimationFrame` for count-up animation with ease-out cubic easing), UseCases (3 research-focused use case cards), and LandingFooter (Product/Resources/Connect link sections with branding). Composed in `page.tsx` as a Server Component importing client components where needed. Added `animate-mesh-gradient` keyframe animation to `globals.css`.

**Why:** The landing page is the first impression for new users and establishes PaperAlchemy as a premium research tool. It showcases all key features (chat, search, upload, citations, comparison, dashboard) with clear CTAs driving users to the chat and search experiences. This spec is a prerequisite for the broader P13 UI Enhancement phase and provides the public-facing entry point.

**Core features:** Animated mesh gradient hero with CTA buttons, 6-card feature grid with hover-lift effects, scroll-triggered animated number counters (IntersectionObserver), 3 research use case cards, responsive footer with navigation links, fully responsive layout (mobile/tablet/desktop via Tailwind responsive classes)

**Tests:** 17 tests — HeroSection (3: headline, CTAs, gradient element), FeatureGrid (3: card count, content, heading), StatsCounter (2: stat items, number/label structure), UseCases (3: item count, content, heading), LandingFooter (3: branding, sections, links), Home page integration (3: hero, features, CTA links)

**Files:** `frontend/src/app/page.tsx`, `frontend/src/app/page.test.tsx`, `frontend/src/components/landing/hero-section.tsx`, `frontend/src/components/landing/feature-grid.tsx`, `frontend/src/components/landing/stats-counter.tsx`, `frontend/src/components/landing/use-cases.tsx`, `frontend/src/components/landing/landing-footer.tsx`, `frontend/src/components/landing/landing.test.tsx`, `frontend/src/app/globals.css`

---

## S13.4 — Search & Discovery Polish

**What:** Premium search & discovery UX upgrade — autocomplete/typeahead with recent searches stored in localStorage, rich paper cards with colored category chips and bookmark toggle, staggered fade-in animations, active filter pills with remove buttons, enhanced "no results" state with gradient illustration and search suggestions, and shimmer loading skeletons.

**How:** Created `useRecentSearches` hook (`frontend/src/lib/hooks/use-recent-searches.ts`) using `useState` + `localStorage` with max 10 LIFO entries, deduplication, and graceful fallback for corrupted/unavailable storage. Created `FilterPills` component rendering active query/category/sort as removable pill badges (hides default "relevance" sort). Upgraded `PaperCard` with 10-color category chip mapping (`CATEGORY_COLORS` record mapping arXiv prefixes to Tailwind color classes), `Bookmark` icon toggle (visual only), full abstract reveal on hover via `onMouseEnter`/`onMouseLeave` state, and glassmorphism `glass-card` styling with `animationDelay` prop for staggered entry. Upgraded `SearchBar` with recent searches dropdown (on focus, filtered by input, max 5 shown, individual remove via Trash2 icon, "Clear all" button, click-outside dismiss). Upgraded `SearchResults` with shimmer skeletons (CSS `@keyframes shimmer` gradient sweep), `animate-fade-in-up` staggered animation (capped at 10 items × 50ms), and enhanced empty state with gradient orb illustration + 5 clickable search suggestions. Added `@keyframes shimmer` and `@keyframes fade-in-up` animations to `globals.css` with `prefers-reduced-motion` fallbacks. Wired `FilterPills` into search page with URL param removal handlers. Used `vi.stubGlobal("localStorage", ...)` mock pattern for Node 25 compatibility.

**Why:** Transforms the basic S9.3 search interface into a polished, production-quality experience. Recent searches reduce friction for repeat queries. Colored category chips provide visual categorization at a glance. Staggered animations and shimmer loaders give a premium feel. Filter pills make active filters visible and easily removable. The enhanced empty state with suggestions helps users discover content. No downstream specs depend directly on S13.4 but it completes a key part of the P13 UI Enhancement phase.

**Core features:** Recent searches hook (localStorage, max 10, LIFO, deduplication), search bar dropdown (on focus, filtered, individual remove, clear all), 10-color arXiv category chip mapping, bookmark icon toggle (visual), full abstract on hover, staggered fade-in-up animation (50ms × index, capped at 10), active filter pills (query/category/sort with × remove), enhanced empty state (gradient orb + 5 search suggestions), shimmer skeleton loading (CSS gradient sweep), prefers-reduced-motion fallbacks

**Tests:** 75 tests across 8 files — use-recent-searches.test.ts (11: add, LIFO, dedup, max 10, remove, clear, persist, load, empty/whitespace, corrupted), search-bar.test.tsx (12: existing 7 + dropdown focus, no dropdown empty, select recent, save on submit, remove individual), paper-card.test.tsx (15: existing 10 + colored chips, bookmark render/toggle, hover abstract, animation delay), filter-pills.test.tsx (8: category/query/sort pills, remove callbacks, hide default sort, empty state), search-results.test.tsx (11: existing 6 + shimmer glass-card, query in empty state, suggestions render/click, long query truncation), plus unchanged pagination (10), category-filter (5), sort-select (3)

**Files:** `frontend/src/lib/hooks/use-recent-searches.ts`, `frontend/src/lib/hooks/use-recent-searches.test.ts`, `frontend/src/components/search/filter-pills.tsx`, `frontend/src/components/search/filter-pills.test.tsx`, `frontend/src/components/search/paper-card.tsx`, `frontend/src/components/search/paper-card.test.tsx`, `frontend/src/components/search/search-bar.tsx`, `frontend/src/components/search/search-bar.test.tsx`, `frontend/src/components/search/search-results.tsx`, `frontend/src/components/search/search-results.test.tsx`, `frontend/src/components/search/index.ts`, `frontend/src/app/search/page.tsx`, `frontend/src/app/globals.css`

---

## S13.3 — Chat UX Polish

**What was done:** Premium chat experience upgrade — replaced custom markdown parser with react-markdown (remark-gfm + rehype-highlight), added code block copy-to-clipboard, enhanced source citation cards with gradient badges and hover effects, added follow-up suggestion chips, upgraded typing indicator and scroll-to-bottom with framer-motion animations, and enhanced the welcome empty state with gradient illustration.

**How it was done:** Replaced the hand-rolled `renderMarkdown()` in message-bubble.tsx with `react-markdown` + `remark-gfm` + `rehype-highlight`, using custom component overrides for headings, lists, tables, blockquotes, code blocks, and inline code. Created a new `CodeBlock` component with language label and clipboard copy button (Clipboard API with "Copied!" feedback). Enhanced `SourceCard` with gradient number badges, FileText icons, hover transitions, and group-hover effects. Created `FollowUpChips` component for clickable follow-up suggestions after assistant responses. Upgraded `TypingIndicator` and `ScrollToBottom` with framer-motion `motion.*` components and `AnimatePresence` for smooth entrance/exit animations. Enhanced `WelcomeState` with gradient icon container, blurred glow background, and "AI-Powered Research" badge. Added `suggested_followups` field to ChatMessage type. Wired FollowUpChips and AnimatePresence into the chat page.

**Why it matters:** This transforms the chat from a basic text interface into a polished, production-quality experience. Markdown rendering is essential for research answers that include code, tables, and formatted text. Citation cards make sources more discoverable. Follow-up chips improve engagement and discoverability. Animations add perceived quality. These improvements are foundational for the premium UX that P13 aims to deliver.

**Core features:** react-markdown rendering (headings, lists, bold, italic, links, tables, blockquotes, strikethrough via remark-gfm), rehype-highlight syntax highlighting for code blocks, CodeBlock component (language label + copy-to-clipboard with "Copied!" feedback), rich SourceCard (gradient number badge, FileText icon, group-hover effects, hover:shadow-sm), FollowUpChips (Sparkles icon, rounded-full chips, primary color scheme, click-to-send), framer-motion TypingIndicator (motion.div entrance/exit, motion.span animated dots with staggered delay), framer-motion ScrollToBottom (AnimatePresence, scale+opacity animation), gradient WelcomeState (blurred glow, gradient icon, "AI-Powered Research" badge), message timestamps with data-testid, ChatMessage.suggested_followups type field, inline citation badges preserved via processChildren()

**Tests:** 60 tests across 9 files — message-bubble.test.tsx (13: user/assistant rendering, markdown renderer, source cards, citations, error/retry, timestamps, older timestamps), code-block.test.tsx (7: render content, language label, default "text", copy button, clipboard copy, "Copied!" feedback, language class), source-card.test.tsx (10: title, arxiv link, authors/year, truncated authors, number badge, testid, missing fields, no arxiv_id, hover classes, gradient badge), followup-chips.test.tsx (5: render chips, correct text, click callback, empty array, container testid), typing-indicator.test.tsx (3: render, three dots, framer-motion wrapper), welcome-state.test.tsx (6: title/description, suggested questions, click callback, try asking label, gradient icon, AI badge), scroll-to-bottom.test.tsx (3: visible render, hidden render, click), citation-badge.test.tsx (3), message-input.test.tsx (10)

**Files:** `frontend/src/components/chat/message-bubble.tsx`, `frontend/src/components/chat/code-block.tsx` (new), `frontend/src/components/chat/followup-chips.tsx` (new), `frontend/src/components/chat/source-card.tsx`, `frontend/src/components/chat/typing-indicator.tsx`, `frontend/src/components/chat/welcome-state.tsx`, `frontend/src/components/chat/scroll-to-bottom.tsx`, `frontend/src/components/chat/index.ts`, `frontend/src/app/chat/page.tsx`, `frontend/src/types/chat.ts`, plus 9 test files

---

## S13.5 — Premium Navigation & Command Palette

**What:** Upgraded the sidebar, header, and breadcrumbs into a premium navigation experience with gradient branding, active item indicators, keyboard shortcuts, a global command palette, breadcrumb dropdowns, and a notification bell.

**How:** Enhanced existing layout components (Sidebar, SidebarNavItem, Breadcrumbs, Header, AppShell) and created new ones (CommandPalette, NotificationBell, useKeyboardShortcuts hook). Used Radix UI primitives from S9b.5: CommandDialog for Cmd+K palette, Tooltip for collapsed sidebar tooltips, DropdownMenu for breadcrumb sibling navigation and notification dropdown. TDD with 74 total layout tests.

**Why:** Premium navigation polish is essential for a professional research assistant UX — keyboard shortcuts (Cmd+1-6) and command palette (Cmd+K) enable power-user workflows, gradient logo and accent indicators establish visual identity, and notification bell sets up infrastructure for future paper alerts.

**Core features:** Gradient logo mark (bg-gradient-to-br from-primary via-primary/80 to-violet-600), active nav item left border accent (border-l-3 border-primary + bg-primary/10), collapsed sidebar tooltips with shortcut hints (Tooltip primitive), keyboard shortcut hints as kbd elements, Cmd/Ctrl+1-6 page navigation with input field bypass, Cmd/Ctrl+K command palette (CommandDialog with fuzzy search across Pages/Actions groups), breadcrumb dropdown with sibling route navigation (DropdownMenu), notification bell with badge count (0 hides, 99+ truncation), smooth sidebar collapse animation (duration-300 ease-in-out)

**Tests:** 74 tests across 9 layout files — sidebar.test.tsx (9: nav items, branding, collapse toggle, localStorage persistence, gradient logo mark, transition classes), sidebar-nav-item.test.tsx (11: icon/label render, href, collapsed hide, active styling, nested routes, left border accent, inactive border transparent, shortcut hint display, collapsed hint hidden, tooltip wrapper), command-palette.test.tsx (9: closed render, dialog open, search input, Pages group, Actions group, fuzzy filter, empty state, navigation on select, close on Escape), breadcrumbs.test.tsx (10: root path, single/nested breadcrumbs, Home link, intermediate links, last segment text, aria-label, capitalize, dropdown trigger, sibling routes), notification-bell.test.tsx (6: bell button, badge count, zero hides badge, 99+ truncation, dropdown open, placeholder items), keyboard-shortcuts.test.tsx (11: Cmd+1-6 navigation, Ctrl key, input bypass, textarea bypass, Cmd+K callback, no modifier ignored), app-shell.test.tsx (6), header.test.tsx (6), mobile-nav.test.tsx (6)

**Files:** `frontend/src/components/layout/sidebar.tsx` (enhanced), `frontend/src/components/layout/sidebar-nav-item.tsx` (enhanced), `frontend/src/components/layout/breadcrumbs.tsx` (enhanced), `frontend/src/components/layout/header.tsx` (enhanced), `frontend/src/components/layout/app-shell.tsx` (enhanced), `frontend/src/components/layout/nav-items.ts` (enhanced with shortcuts), `frontend/src/components/layout/command-palette.tsx` (new), `frontend/src/components/layout/notification-bell.tsx` (new), `frontend/src/components/layout/use-keyboard-shortcuts.ts` (new), `frontend/src/components/layout/index.ts` (updated exports), plus 9 test files

---

## S13.6 — Animations & Micro-Interactions

**What:** 8 animation/micro-interaction components providing polish-level UX across the frontend — page transitions, skeleton loaders, button press feedback, hover card previews, toast notifications, progress indicators, scroll-triggered fade-in, and animated number counters.

**How:** Built as a dedicated `frontend/src/components/animations/` module with barrel export. Uses framer-motion for page transitions (AnimatePresence + motion.div with fade/slide), CSS animations for shimmer and indeterminate progress, IntersectionObserver for scroll-triggered effects (ScrollFadeIn, AnimatedCounter), and sonner for toast notifications. All components respect `prefers-reduced-motion` via the existing globals.css media query. Added IntersectionObserver polyfill to test setup for jsdom compatibility.

**Why:** Elevates PaperAlchemy from functional to premium — subtle animations create perceived responsiveness, skeleton loaders reduce perceived loading time, scroll-triggered reveals create engagement on the dashboard, and toast notifications provide clear feedback for user actions.

**Core features:** PageTransition (framer-motion fade+slide, 0.25s ease-out), SkeletonCard/Text/Chart/List (animate-pulse + shimmer variants, configurable lines/count), PressableButton (scale-95 on press with 100ms transition, disabled bypass), HoverCardPreview (200ms hide delay, 150-char abstract truncation, animated popover), ToastProvider (sonner at bottom-right, max 3 visible, rich colors), ProgressIndicator (determinate with width%, indeterminate with gradient animation, error/success variants, optional label/percentage), ScrollFadeIn (IntersectionObserver with 0.1 threshold, configurable stagger delay, animate-once), AnimatedCounter (requestAnimationFrame + ease-out cubic, abbreviation for K/M, prefix/suffix support).

**Tests:** 57 tests across 8 test files — page-transition.test.tsx (5: renders children, fade/slide props, exit props, transition duration, className), skeleton-shimmer.test.tsx (10: pulse class, rounded, className, multi-line, default 1 line, last line shorter, shimmer class, aspect-video, list count, default 3), pressable-button.test.tsx (8: renders, scale-95 mousedown, scale-1 mouseup, mouseleave reset, disabled bypass, onClick forward, transition style, className), hover-card-preview.test.tsx (6: trigger render, show on hover, authors display, abstract truncation, hide with delay, no abstract graceful), toast-provider.test.tsx (4: renders toaster, bottom-right position, max 3 visible, rich colors), progress-indicator.test.tsx (10: determinate, width%, indeterminate animate, 0% edge, 100% edge, error variant, success variant, label, percentage text, className), scroll-fade-in.test.tsx (7: renders children, initial opacity 0, visible on intersect, threshold config, unobserve after visible, stagger delay, className), animated-counter.test.tsx (7: initial 0, locale format, 0 no-animate, abbreviate K, abbreviate M, prefix/suffix, className)

**Integration:** ToastProvider added to AppShell layout, ScrollFadeIn wraps dashboard sections with 100ms staggered delays, landing page StatsCounter refactored to use AnimatedCounter (replaced custom useCountUp hook), PageTransition wraps landing page content, IntersectionObserver polyfill added to global test setup.

**Files:** `frontend/src/components/animations/page-transition.tsx`, `frontend/src/components/animations/skeleton-shimmer.tsx`, `frontend/src/components/animations/pressable-button.tsx`, `frontend/src/components/animations/hover-card-preview.tsx`, `frontend/src/components/animations/toast-provider.tsx`, `frontend/src/components/animations/progress-indicator.tsx`, `frontend/src/components/animations/scroll-fade-in.tsx`, `frontend/src/components/animations/animated-counter.tsx`, `frontend/src/components/animations/index.ts`, plus 8 test files. Modified: `frontend/src/app/globals.css` (progress-indeterminate keyframe), `frontend/src/components/layout/app-shell.tsx` (ToastProvider), `frontend/src/app/dashboard/page.tsx` (ScrollFadeIn), `frontend/src/components/landing/stats-counter.tsx` (AnimatedCounter), `frontend/src/app/page.tsx` (PageTransition), `frontend/src/test/setup.ts` (IntersectionObserver polyfill)

---

## S13.7 — Mobile-First Responsive Polish

**Status:** Done  
**Phase:** P13 (UI Enhancement)  
**Date:** 2026-03-13

### What
Complete mobile-first responsive overhaul: bottom navigation bar, swipe gestures, pull-to-refresh, touch-optimized targets (44px min), adaptive layouts, mobile filter sheet, and fluid typography with CSS clamp().

### How
- **BottomNav** (`frontend/src/components/layout/bottom-nav.tsx`): Fixed bottom bar with 6 nav items (icons + labels), active route highlighting via `usePathname`, hidden on md+ screens via `md:hidden`, 44px min touch targets.
- **useSwipe** hook (`frontend/src/lib/hooks/use-swipe.ts`): Touch event-based swipe detection with configurable threshold (default 50px), horizontal-only (ignores vertical scrolling), `onSwipeLeft`/`onSwipeRight`/`onSwiping` callbacks.
- **PullToRefresh** (`frontend/src/components/layout/pull-to-refresh.tsx`): Pull-down gesture with dampened resistance, visual spinner indicator, only activates at scrollTop=0, mobile-only display.
- **MobileFilterSheet** (`frontend/src/components/search/mobile-filter-sheet.tsx`): Bottom sheet (shadcn Sheet) with category + sort controls, active filter badge count, all options have 44px touch targets.
- **Fluid Typography** (`globals.css`): All headings use `clamp()` for smooth scaling (e.g., h1: `clamp(1.5rem, 1rem + 2vw, 2.25rem)`), body text also fluid.
- **Touch Target Utility** (`globals.css`): `.touch-target` class for 44px minimum hit targets.
- **AppShell Integration**: BottomNav added to app-shell, main content gets `pb-20` on mobile to clear the bottom nav.
- **Search Page**: Inline filters hidden on mobile (visible via MobileFilterSheet), results wrapped in PullToRefresh.

### Why
Mobile users need touch-friendly navigation and interactions. The existing sidebar-based mobile nav (drawer overlay) was functional but not optimized for one-handed use. Bottom navigation is the standard mobile pattern, and touch targets/gestures/fluid typography ensure a polished mobile experience.

### Core Features
- Bottom navigation bar (mobile only, replaces sidebar)
- Swipe gesture hook (reusable, configurable threshold)
- Pull-to-refresh on search results
- 44px minimum touch targets on all interactive elements
- Mobile filter bottom sheet with active filter badge
- Fluid typography scaling via CSS clamp()
- Adaptive layouts (single-column mobile → multi-column desktop)

### Test Coverage
- `bottom-nav.test.tsx` — 8 tests (rendering, active state, CSS classes, touch targets)
- `use-swipe.test.ts` — 7 tests (left/right swipe, threshold, vertical ignore, swiping callback)
- `pull-to-refresh.test.tsx` — 7 tests (pull indicator, refresh trigger, threshold, scroll position)
- `mobile-filter-sheet.test.tsx` — 10 tests (trigger, sheet content, filter selection, badge count)
- Updated `design-system.test.tsx` for clamp-based typography
