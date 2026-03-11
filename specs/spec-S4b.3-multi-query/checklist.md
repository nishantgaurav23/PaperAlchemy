# Checklist -- Spec S4b.3: Multi-Query Retrieval

## Phase 1: Setup & Dependencies
- [x] Verify S4.4 (hybrid search) is "done"
- [x] Verify S5.1 (LLM client) is "done"
- [x] Verify `src/services/retrieval/` directory exists

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_multi_query.py`
- [x] Write failing tests for FR-1 (MultiQueryService interface)
- [x] Write failing tests for FR-2 (query variation generation)
- [x] Write failing tests for FR-3 (parallel hybrid search)
- [x] Write failing tests for FR-4 (result deduplication)
- [x] Write failing tests for FR-5 (RRF fusion)
- [x] Write failing tests for FR-6 (graceful fallback)
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Add `MultiQuerySettings` to `src/config.py` (FR-7)
- [x] Create `MultiQueryResult` dataclass
- [x] Implement `generate_query_variations()` -- pass tests (FR-2)
- [x] Implement parallel search with `asyncio.gather` (FR-3)
- [x] Implement deduplication by chunk_id (FR-4)
- [x] Implement RRF fusion scoring (FR-5)
- [x] Implement `retrieve_with_multi_query()` full pipeline (FR-1)
- [x] Implement graceful fallback (FR-6)
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Add factory function (FR-8)
- [x] Update `src/services/retrieval/__init__.py` exports
- [x] Run lint (`ruff check src/services/retrieval/multi_query.py`)
- [x] Run full test suite (482 passed, 9 skipped)

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook `notebooks/specs/S4b.3_multi_query.ipynb`
- [x] Update roadmap.md status to "done"
