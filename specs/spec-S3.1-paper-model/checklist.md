# Checklist -- Spec S3.1: Paper ORM Model & Repository

## Phase 1: Setup & Dependencies
- [x] Verify S2.2 (Database layer) is "done"
- [x] Create `src/models/` directory
- [x] Create `src/repositories/` directory
- [x] Create `src/schemas/paper.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_paper_model.py`
- [x] Write failing tests for Paper ORM model (columns, table name, indexes)
- [x] Write failing tests for PaperRepository CRUD (create, get, update, delete)
- [x] Write failing tests for PaperRepository queries (date range, category, status, search)
- [x] Write failing tests for upsert + bulk upsert
- [x] Write failing tests for Pydantic schemas (PaperCreate, PaperUpdate, PaperResponse)
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement `src/models/paper.py` -- Paper ORM class
- [x] Implement `src/models/__init__.py` -- re-export Paper
- [x] Implement `src/schemas/paper.py` -- PaperCreate, PaperUpdate, PaperResponse
- [x] Implement `src/repositories/paper.py` -- async PaperRepository
- [x] Implement `src/repositories/__init__.py` -- re-export PaperRepository
- [x] Run tests -- expect pass (Green) — 36/36 passing
- [x] Refactor if needed

## Phase 4: Integration
- [x] Wire PaperRepository into DI (`src/dependency.py` — PaperRepoDep)
- [x] Import Paper model in `src/db/__init__.py` so create_tables sees it
- [x] Run lint (`ruff check src/ tests/`) — all passed
- [x] Run full test suite — 176/176 passing

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Notebook created: `notebooks/specs/S3.1_paper_model.ipynb`
- [x] Update roadmap.md status to "done"
