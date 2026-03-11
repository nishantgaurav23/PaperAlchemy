# Checklist -- Spec S2.4: Dependency Injection

## Phase 1: Setup & Dependencies
- [x] Verify S2.1 (FastAPI app factory) is "done"
- [x] Verify S2.2 (Database layer) is "done"
- [x] Create `src/dependency.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_dependency.py`
- [x] Write failing tests for FR-1 (Settings dependency)
- [x] Write failing tests for FR-2 (Database dependency)
- [x] Write failing tests for FR-3 (Async session dependency)
- [x] Write failing tests for FR-4 (Annotated type aliases)
- [x] Write failing tests for FR-5 (Dependency overrides)
- [x] Write integration test (router with injected deps)
- [x] Run tests -- expect failures (Red) — 13 failed, 2 errors

## Phase 3: Implementation
- [x] Implement `get_settings()` dependency function (re-export from src.config)
- [x] Implement `get_database()` dependency function (wraps src.db._get_database)
- [x] Implement `get_db_session()` re-export from `src/db`
- [x] Create `SettingsDep`, `DatabaseDep`, `SessionDep` type aliases
- [x] Run tests -- expect pass (Green) — 15/15 passing
- [x] Refactor if needed — clean, no refactoring needed

## Phase 4: Integration
- [x] Wire into app (verify imports work from routers)
- [x] Run lint (`ruff check` + `ruff format --check`) — clean
- [x] Run full test suite — 140/140 passing

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook: `notebooks/specs/S2.4_di.ipynb`
- [x] Update roadmap.md status to "done"
- [x] Append spec summary to `docs/spec-summaries.md`
