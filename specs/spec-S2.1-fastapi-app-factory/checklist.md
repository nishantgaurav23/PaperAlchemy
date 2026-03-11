# Checklist — Spec S2.1: FastAPI App Factory

## Phase 1: Setup & Dependencies
- [x] Verify S1.2 (environment config) is "done"
- [x] Create `src/routers/` directory with `__init__.py`
- [x] Create `src/schemas/api/` directory with `__init__.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_main.py`
- [x] Write `test_create_app_returns_fastapi_instance`
- [x] Write `test_create_app_sets_title_and_version`
- [x] Write `test_ping_endpoint_returns_ok`
- [x] Write `test_ping_response_schema`
- [x] Write `test_cors_headers_present`
- [x] Write `test_lifespan_startup_shutdown`
- [x] Write `test_create_app_with_custom_settings`
- [x] Write `test_404_for_unknown_routes`
- [x] Run tests — expect failures (Red)

## Phase 3: Implementation
- [x] Create `src/schemas/api/health.py` with `PingResponse`
- [x] Create `src/routers/ping.py` with health check endpoint
- [x] Create `src/main.py` with `create_app()` and lifespan
- [x] Run tests — expect pass (Green)
- [x] Refactor if needed (no refactoring needed)

## Phase 4: Integration
- [x] Verify app starts: `uvicorn src.main:app`
- [x] Verify `GET /api/v1/ping` returns correct response
- [x] Run lint (`ruff check src/ tests/`)
- [x] Run full test suite (70/70 passed)

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook: `notebooks/specs/S2.1_app_factory.ipynb`
- [x] Update roadmap.md status to "done"
- [x] Append spec summary to `docs/spec-summaries.md`
