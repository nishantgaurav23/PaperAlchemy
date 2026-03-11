# Spec S2.4 -- Dependency Injection

## Overview
Implement FastAPI's `Depends()` pattern with typed `Annotated[]` aliases for clean, testable service injection. Routers declare what they need via type hints — they never import factories or create clients directly. This module starts with core dependencies (Settings, Database, AsyncSession) and is designed to grow as future specs add services (OpenSearch, Jina, LLM, Cache, etc.).

## Dependencies
- **S2.1** (FastAPI app factory) — provides the app with lifespan and `app.state`
- **S2.2** (Database layer) — provides `Database`, `AsyncSession`, `get_db_session()`

## Target Location
- `src/dependency.py` — dependency functions + `Annotated[]` type aliases

## Functional Requirements

### FR-1: Settings Dependency
- **What**: Provide `Settings` to any route handler via `Depends()`
- **Inputs**: `Request` object (via FastAPI injection)
- **Outputs**: `Settings` instance from `get_settings()` (lru_cached singleton)
- **Edge cases**: Settings not available (should never happen after app init)

### FR-2: Database Dependency
- **What**: Provide the singleton `Database` instance to route handlers
- **Inputs**: None (reads from module-level `_get_database()` in `src/db`)
- **Outputs**: `Database` instance
- **Edge cases**: Database not initialized — raises `RuntimeError`

### FR-3: Async Session Dependency
- **What**: Provide a per-request `AsyncSession` that auto-commits/rollbacks/closes
- **Inputs**: None (uses `get_db_session()` from `src/db`)
- **Outputs**: Yields `AsyncSession`, auto-managed lifecycle
- **Edge cases**: Database connection failure — `SQLAlchemyError` propagates

### FR-4: Annotated Type Aliases
- **What**: Pre-built `Annotated[]` aliases for concise router signatures
- **Aliases**:
  - `SettingsDep = Annotated[Settings, Depends(get_settings)]`
  - `DatabaseDep = Annotated[Database, Depends(get_database)]`
  - `SessionDep = Annotated[AsyncSession, Depends(get_db_session)]`
- **Usage**: `async def my_route(settings: SettingsDep, session: SessionDep)`

### FR-5: Extensibility Pattern
- **What**: Clear pattern for adding future service dependencies
- **Pattern**: Each new service gets: (1) a getter function, (2) an `Annotated[]` alias
- **Future services** (added by later specs): OpenSearch, Jina, LLM, Cache, Langfuse, ArXiv, PDF Parser, Agentic RAG
- **Convention**: getter function reads from `src/db` module or `request.app.state`

## Tangible Outcomes
- [ ] `src/dependency.py` exists with all dependency functions and type aliases
- [ ] `get_settings()` returns the `Settings` singleton
- [ ] `get_database()` returns the `Database` instance (or raises `RuntimeError`)
- [ ] `get_db_session()` yields an `AsyncSession` with proper lifecycle
- [ ] `SettingsDep`, `DatabaseDep`, `SessionDep` type aliases are importable
- [ ] Router can use `async def route(settings: SettingsDep)` pattern
- [ ] Dependencies are overridable via `app.dependency_overrides` for testing
- [ ] All tests pass with mocked dependencies (no real DB needed)

## Test-Driven Requirements

### Tests to Write First
1. `test_get_settings_returns_settings`: Verify `get_settings` returns a `Settings` instance
2. `test_get_database_returns_database`: Verify `get_database` returns `Database` after init
3. `test_get_database_raises_when_not_initialized`: Verify `RuntimeError` when DB not initialized
4. `test_get_db_session_yields_async_session`: Verify session lifecycle (yield + close)
5. `test_dependency_override_settings`: Override `get_settings` in test, verify router gets mock
6. `test_dependency_override_session`: Override `get_db_session` in test, verify router gets mock session
7. `test_annotated_aliases_importable`: Import all `*Dep` aliases successfully
8. `test_router_with_injected_dependencies`: Full integration — hit endpoint that uses `SettingsDep`

### Mocking Strategy
- Mock `Settings` with a test fixture (no real `.env` needed)
- Mock `Database` — no real PostgreSQL connection
- Mock `AsyncSession` — no real DB queries
- Use `app.dependency_overrides` to swap real deps for mocks in integration tests

### Coverage
- All public functions tested
- Error path tested (uninitialized database)
- Integration test with FastAPI TestClient
