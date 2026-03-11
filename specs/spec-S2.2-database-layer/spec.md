# Spec S2.2 -- Database Layer

## Overview
SQLAlchemy 2.0 async database layer for PaperAlchemy. Provides an async engine with connection pooling, async session factory, health check, and table creation/drop utilities. All database I/O must be async (`async def`, `await`).

The reference code (`_reference/src/db/`) uses sync SQLAlchemy — we rebuild with **async-first** design using `asyncpg` and `AsyncSession`.

## Dependencies
- **S1.2** (Environment Config) — provides `PostgresSettings` with `.url` (async URL) and `.sync_url`
- **S1.3** (Docker Infrastructure) — provides PostgreSQL 16 service in `compose.yml`

## Target Location
- `src/db/__init__.py` — public API exports
- `src/db/database.py` — `Database` class (async engine, sessions, health check)
- `src/db/base.py` — declarative base for ORM models

## Functional Requirements

### FR-1: Declarative Base
- **What**: Provide a shared `Base` class for all ORM models
- **Inputs**: None
- **Outputs**: `Base = declarative_base()` (or `DeclarativeBase` subclass)
- **Edge cases**: Must be importable without side effects (no engine creation)

### FR-2: Async Database Engine
- **What**: Create an async SQLAlchemy engine with connection pooling
- **Inputs**: `database_url: str`, `echo: bool = False`, optional pool settings
- **Outputs**: `AsyncEngine` instance
- **Configuration**:
  - `pool_size=5`, `max_overflow=10`
  - `pool_pre_ping=True` (verify connections before use)
  - `pool_recycle=3600` (recycle after 1 hour)
- **Edge cases**: Invalid URL should raise clear error at creation time

### FR-3: Async Session Factory
- **What**: Provide async sessions for database operations
- **Inputs**: None (uses the engine from FR-2)
- **Outputs**: `AsyncSession` via async context manager
- **Behavior**:
  - Auto-commit on successful exit
  - Auto-rollback on exception
  - Auto-close always
  - `expire_on_commit=False` for post-commit attribute access
- **Edge cases**: Nested session usage, concurrent sessions

### FR-4: Health Check
- **What**: Verify database connectivity with a simple query
- **Inputs**: None
- **Outputs**: `bool` — `True` if DB is reachable, `False` otherwise
- **Query**: `SELECT 1`
- **Edge cases**: Connection timeout, DB unreachable

### FR-5: Table Management
- **What**: Create and drop all ORM tables
- **Inputs**: None
- **Outputs**: Side effect — tables created/dropped
- **Methods**: `create_tables()`, `drop_tables()`
- **Edge cases**: Tables already exist (idempotent)

### FR-6: Engine Disposal
- **What**: Cleanly shut down the engine and all connections
- **Inputs**: None
- **Outputs**: Side effect — engine disposed
- **Method**: `close()`

### FR-7: Session Dependency for FastAPI
- **What**: Async generator suitable for `Depends()` injection
- **Inputs**: None
- **Outputs**: Yields `AsyncSession`
- **Pattern**: `async def get_db_session() -> AsyncGenerator[AsyncSession, None]`

## Tangible Outcomes
- [ ] `src/db/base.py` exists with `Base` class
- [ ] `src/db/database.py` exists with async `Database` class
- [ ] `src/db/__init__.py` exports `Database`, `Base`, `get_db_session`
- [ ] Health check returns `True` against running PostgreSQL
- [ ] Sessions auto-commit on success, auto-rollback on error
- [ ] Connection pooling configured (pool_size=5, max_overflow=10)
- [ ] All methods are async
- [ ] `Database` integrates into FastAPI lifespan (startup/shutdown)

## Test-Driven Requirements

### Tests to Write First
1. `test_database_creation`: Database object initializes with correct engine settings
2. `test_async_session_commit`: Session auto-commits on successful context exit
3. `test_async_session_rollback`: Session auto-rollbacks on exception
4. `test_async_session_closes`: Session is closed after context exit
5. `test_health_check_success`: Returns True when DB is reachable
6. `test_health_check_failure`: Returns False when DB is unreachable
7. `test_create_tables`: Tables are created from Base metadata
8. `test_drop_tables`: Tables are dropped
9. `test_close_disposes_engine`: Engine is disposed on close
10. `test_get_db_session_dependency`: FastAPI dependency yields session and cleans up
11. `test_database_url_from_settings`: Database URL correctly constructed from PostgresSettings

### Mocking Strategy
- Mock `create_async_engine` to avoid real DB connections in unit tests
- Mock `AsyncSession` for session behavior tests
- Use `AsyncMock` for async methods
- No real PostgreSQL needed for unit tests (integration tests later)

### Coverage
- All public methods tested
- Happy path + error paths
- Session lifecycle (commit/rollback/close)
