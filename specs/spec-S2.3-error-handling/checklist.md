# Checklist -- Spec S2.3: Error Handling & Middleware

## Phase 1: Setup & Dependencies
- [x] Verify S2.1 (FastAPI app factory) is "done"
- [x] Create target files: `src/exceptions.py`, `src/middlewares.py`, `src/schemas/api/error.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_exceptions.py`
- [x] Create `tests/unit/test_middlewares.py`
- [x] Write failing tests for FR-1 (exception hierarchy)
- [x] Write failing tests for FR-2 (error response schema)
- [x] Write failing tests for FR-3 (global exception handlers)
- [x] Write failing tests for FR-4 (request logging middleware)
- [x] Write failing tests for FR-5 (wiring into app)
- [x] Run tests -- expect failures (Red) — 23 failed, 11 errors

## Phase 3: Implementation
- [x] Implement FR-1 -- exception hierarchy in `src/exceptions.py`
- [x] Implement FR-2 -- ErrorResponse schema in `src/schemas/api/error.py`
- [x] Implement FR-3 -- global exception handlers
- [x] Implement FR-4 -- RequestLoggingMiddleware in `src/middlewares.py`
- [x] Implement FR-5 -- wire into `create_app()` in `src/main.py`
- [x] Run tests -- expect pass (Green) — 34 passed
- [x] Refactor if needed

## Phase 4: Integration
- [x] Verify middleware is outermost in middleware stack
- [x] Run lint (`ruff check src/ tests/`) — All checks passed
- [x] Run full test suite (`pytest`) — 125 passed

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Notebook created: `notebooks/specs/S2.3_error_handling.ipynb`
- [x] Update roadmap.md status to "done"
- [x] Append spec summary to `docs/spec-summaries.md`
