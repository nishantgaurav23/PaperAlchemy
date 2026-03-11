# Spec S1.1 — Dependency Declaration

## Overview
Clean dependency declaration for PaperAlchemy using UV package manager. The `pyproject.toml` must declare all production and development dependencies with version constraints, organized by purpose. This is the foundation spec — all other specs depend on packages declared here.

## Depends On
None (root spec)

## Location
- `pyproject.toml` (root)

## Requirements

### R1: Project Metadata
- Name: `paper-alchemy`
- Version: `0.1.0`
- Description: "AI Research Assistant — answers research questions with cited papers"
- Python: `>=3.12,<3.13`
- License: MIT
- `readme = "README.md"`

### R2: Production Dependencies (Grouped by Purpose)

#### Web Framework
- `fastapi[standard]>=0.115.0` — async API with auto-docs
- `uvicorn>=0.34.0` — ASGI server

#### Data Validation
- `pydantic>=2.11.0` — data validation
- `pydantic-settings>=2.8.0` — environment config

#### Database
- `sqlalchemy[asyncio]>=2.0.0` — async ORM
- `asyncpg>=0.30.0` — async PostgreSQL driver (replaces psycopg2-binary)
- `alembic>=1.13.0` — database migrations

#### Search
- `opensearch-py>=3.0.0` — OpenSearch client

#### HTTP
- `httpx>=0.28.0` — async HTTP client (arXiv, Jina, external APIs)

#### Async File I/O
- `aiofiles>=24.0.0` — async file read/write (PDF downloads, uploads, temp files)

#### SSE Streaming
- `sse-starlette>=2.0.0` — Server-Sent Events for token-by-token streaming

#### PDF Parsing
- `docling>=2.43.0` — section-aware PDF parser

#### LLM & Agents
- `langchain>=0.3.0` — LLM framework
- `langchain-core>=0.3.0` — core abstractions
- `langchain-community>=0.3.0` — community integrations
- `langchain-ollama>=0.3.0` — Ollama integration
- `langchain-google-genai>=2.0.0` — Google Gemini integration
- `langgraph>=0.2.0` — agent state machines

#### Embeddings & Re-ranking
- `sentence-transformers>=5.0.0` — cross-encoder re-ranking (ms-marco-MiniLM-L-12-v2)

#### Caching
- `redis>=6.0.0` — Redis client (async support)

#### Observability
- `langfuse>=3.0.0` — LLM tracing

#### Evaluation
- `ragas>=0.2.0` — RAG evaluation framework

#### Utilities
- `python-dateutil>=2.9.0` — date parsing (arXiv dates)
- `feedparser>=6.0.0` — arXiv Atom feed parsing

### R3: Development Dependencies (dependency-group: dev)
- `pytest>=8.0.0` — test runner
- `pytest-asyncio>=0.24.0` — async test support
- `pytest-cov>=6.0.0` — coverage
- `pytest-mock>=3.14.0` — mock fixtures
- `pytest-dotenv>=0.5.2` — .env loading in tests
- `pytest-env>=1.1.0` — env variable injection
- `ruff>=0.11.0` — linter + formatter
- `mypy>=1.15.0` — type checking
- `pre-commit>=4.0.0` — git hooks
- `jupyter>=1.0.0` — notebooks
- `notebook>=7.0.0` — notebook server
- `asgi-lifespan>=2.1.0` — testing FastAPI lifespan
- `polyfactory>=2.21.0` — test data factories
- `testcontainers>=4.10.0` — Docker-based integration tests
- `anyio[trio]>=4.9.0` — async testing utilities

### R4: Remove Unused Dependencies
Remove from current pyproject.toml:
- `psycopg2-binary` → replaced by `asyncpg` (async driver)
- `requests` → replaced by `httpx` (async)
- `gradio` → removed (will add back in specific spec if needed)
- `python-telegram-bot` → deferred to S12.1 (Telegram Bot phase)
- `types-sqlalchemy` → not needed with modern SQLAlchemy 2.0

### R5: Tool Configuration
- **Ruff**: line-length=130, exclude notebooks and .venv, src=["src", "tests"]
- **Ruff lint**: select = ["E", "F", "I", "N", "W", "UP", "B", "SIM", "ASYNC"]
- **Ruff format**: quote-style = "double"
- **pytest**: asyncio_mode = "auto", env_files = ".env.test"
- **mypy**: explicit_package_bases = true, plugins = ["pydantic.mypy"]

### R6: UV Lock File
- After updating pyproject.toml, run `uv sync` to generate/update `uv.lock`
- Verify all packages resolve without conflicts

## Tangible Outcomes
1. `pyproject.toml` has all deps declared with proper version bounds
2. `uv sync` succeeds without errors
3. `uv run python -c "import fastapi; import sqlalchemy; import langchain; import langgraph; import ragas"` works
4. `uv run ruff check src/` runs (even on empty src/)
5. `uv run pytest --co` collects 0 tests without errors
6. No unused or legacy deps remain

## TDD Notes
- **Test**: Run `uv sync` and verify exit code 0
- **Test**: Import key packages in Python and verify no ImportError
- **Test**: Run ruff, pytest, mypy with the new config and verify they execute
- This is primarily a configuration spec — tests verify the toolchain works
