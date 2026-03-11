# Spec S2.1 — FastAPI App Factory

## Overview
App factory pattern for the FastAPI application with async lifespan management, router registration, CORS middleware, and a health check endpoint. This is the entry point for the entire PaperAlchemy backend — all routers, middleware, and startup/shutdown logic wire through here.

## Dependencies
- S1.2 (Environment configuration) — `src/config.py` must exist with `get_settings()` and `AppSettings`

## Target Location
- `src/main.py` — app factory function + lifespan
- `src/routers/ping.py` — health check router

## Functional Requirements

### FR-1: App Factory Function
- **What**: `create_app()` function that returns a fully configured `FastAPI` instance
- **Behavior**:
  - Reads settings via `get_settings()`
  - Sets app `title`, `version`, `debug` from `AppSettings`
  - Registers lifespan context manager
  - Adds CORS middleware (configurable origins, default `["*"]` for dev)
  - Includes all routers with `/api/v1` prefix
  - Returns the app instance
- **Inputs**: Optional `Settings` override for testing
- **Outputs**: `FastAPI` instance
- **Edge cases**: No routers registered yet (only ping) — must not crash

### FR-2: Async Lifespan
- **What**: `@asynccontextmanager` lifespan for startup/shutdown hooks
- **Startup**:
  - Log app name, version, debug mode
  - Future: initialize DB pool, OpenSearch client, Redis, etc. (stubs for now)
- **Shutdown**:
  - Log shutdown message
  - Future: close DB pool, OpenSearch, Redis connections (stubs for now)
- **Edge cases**: Startup failure should log error and propagate (don't swallow)

### FR-3: CORS Middleware
- **What**: Add `CORSMiddleware` with configurable allowed origins
- **Defaults** (dev): `allow_origins=["*"]`, `allow_credentials=True`, `allow_methods=["*"]`, `allow_headers=["*"]`
- **Edge cases**: Production should restrict origins (future config)

### FR-4: Health Check Endpoint
- **What**: `GET /api/v1/ping` returns `{"status": "ok", "version": "x.y.z"}`
- **Location**: `src/routers/ping.py`
- **Response model**: Pydantic schema `PingResponse`
- **Status code**: 200
- **Edge cases**: Must work without any external services running

### FR-5: Router Registration
- **What**: All routers included via `app.include_router()` with `/api/v1` prefix
- **Initial routers**: Only `ping_router` for this spec
- **Convention**: Each router module exports a `router` variable
- **Edge cases**: Adding a router that doesn't exist should fail at import time (not silently)

## Tangible Outcomes
- [ ] `src/main.py` exists with `create_app()` function
- [ ] `src/routers/__init__.py` exists
- [ ] `src/routers/ping.py` exists with `GET /ping` endpoint
- [ ] `src/schemas/api/__init__.py` exists
- [ ] `src/schemas/api/health.py` exists with `PingResponse` schema
- [ ] App starts without errors: `uvicorn src.main:app`
- [ ] `GET /api/v1/ping` returns `{"status": "ok", "version": "0.1.0"}`
- [ ] CORS headers present in responses
- [ ] Lifespan logs startup and shutdown messages
- [ ] All tests pass with `pytest tests/unit/test_main.py`

## Test-Driven Requirements

### Tests to Write First
1. `test_create_app_returns_fastapi_instance`: Verify `create_app()` returns a FastAPI app
2. `test_create_app_sets_title_and_version`: Verify app metadata from settings
3. `test_ping_endpoint_returns_ok`: `GET /api/v1/ping` returns 200 with correct body
4. `test_ping_response_schema`: Response matches `PingResponse` model
5. `test_cors_headers_present`: OPTIONS request returns CORS headers
6. `test_lifespan_startup_shutdown`: Verify lifespan runs without errors (via TestClient)
7. `test_create_app_with_custom_settings`: Override settings for testing
8. `test_404_for_unknown_routes`: Unknown paths return 404

### Mocking Strategy
- No external services needed — this spec is self-contained
- Use `httpx.AsyncClient` with `ASGITransport` for async test client
- Override `get_settings()` via dependency override for custom settings tests

### Coverage
- All public functions tested
- Edge cases: unknown routes, CORS preflight, lifespan lifecycle
- Error paths: startup with invalid config (future)
