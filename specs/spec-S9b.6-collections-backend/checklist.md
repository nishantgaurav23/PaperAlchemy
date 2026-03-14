# Checklist — Spec S9b.6: Collections Backend API

## Phase 1: Setup & Dependencies
- [x] Verify S2.4 (DI) and S3.1 (Paper model) are "done"
- [x] Verify `src/models/`, `src/repositories/`, `src/routers/` directories exist

## Phase 2: Tests First (TDD Red)
- [x] Create `tests/unit/test_collections.py` with:
  - [x] Test collection model creation (6 tests)
  - [x] Test repository CRUD: create, get_by_id, list_all, update, delete (9 tests)
  - [x] Test repository paper operations: add_paper, remove_paper, get_collection_papers (10 tests)
  - [x] Test repository edge cases: duplicate add paper (idempotent), remove non-existent paper, delete non-existent collection
  - [x] Test repository count (2 tests)
  - [x] Test Pydantic schemas (6 tests)
  - [x] Test API endpoints: GET, POST, PUT, DELETE /collections (9 tests)
  - [x] Test API paper endpoints: POST /collections/{id}/papers, DELETE /collections/{id}/papers/{paper_id} (4 tests)
  - [x] Test API error cases: 404 collection, 404 paper
- [x] Run tests — all fail (Red) ✅

## Phase 3: Implementation (TDD Green)
- [x] Create `src/models/collection.py` — Collection model + collection_papers association table
- [x] Register model in `src/models/__init__.py`
- [x] Create `src/schemas/collection.py` — Pydantic schemas
- [x] Create `src/repositories/collection.py` — CollectionRepository
- [x] Create `src/routers/collections.py` — REST API router
- [x] Add CollectionRepoDep to `src/dependency.py`
- [x] Register collections router in `src/main.py`
- [x] Run tests — all 48 pass (Green) ✅

## Phase 4: Refactor & Polish
- [x] Lint check: all files pass ruff
- [x] Verify all existing tests still pass (1092 passed, 4 pre-existing arxiv failures)
- [x] Update roadmap.md status to "done"

## Phase 5: Verification
- [x] All tests pass (`pytest tests/unit/test_collections.py -v`) — 48 passed
- [x] Lint clean
- [x] API endpoints documented in OpenAPI (auto via FastAPI)
- [x] Spec summary appended to `docs/spec-summaries.md`
