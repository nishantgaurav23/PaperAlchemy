# Checklist -- Spec S2.2: Database Layer

## Phase 1: Setup & Dependencies
- [x] Verify S1.2 (Environment Config) is "done"
- [x] Verify S1.3 (Docker Infrastructure) is "done"
- [x] Create `src/db/` directory with `__init__.py`, `base.py`, `database.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_database.py`
- [x] Write failing tests for async engine creation
- [x] Write failing tests for async session lifecycle (commit/rollback/close)
- [x] Write failing tests for health check (success + failure)
- [x] Write failing tests for table management (create/drop)
- [x] Write failing tests for engine disposal
- [x] Write failing tests for `get_db_session` FastAPI dependency
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement `src/db/base.py` — declarative Base
- [x] Implement `src/db/database.py` — async Database class
- [x] Implement `src/db/__init__.py` — public exports
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Wire Database into FastAPI lifespan (startup/shutdown in `src/main.py`)
- [x] Run lint (`ruff check src/db/ tests/unit/test_database.py`)
- [x] Run full test suite (`pytest`) — 91/91 passing

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets (DB URL from settings only)
- [x] Create notebook: `notebooks/specs/S2.2_database.ipynb`
- [x] Update roadmap.md status to "done"
- [x] Append spec summary to `docs/spec-summaries.md`
