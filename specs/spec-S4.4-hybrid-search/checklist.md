# Checklist -- Spec S4.4: Hybrid Search (BM25 + KNN + RRF)

## Phase 1: Setup & Dependencies
- [x] Verify S4.1 (OpenSearch client) is "done"
- [x] Verify S4.2 (Text chunker) is "done"
- [x] Verify S4.3 (Embedding service) is "done"
- [x] Create `src/schemas/api/search.py`
- [x] Create `src/routers/search.py`
- [x] Create `tests/unit/test_search_router.py`

## Phase 2: Tests First (TDD)
- [x] Write schema validation tests (FR-1, FR-2)
- [x] Write result mapping tests (FR-5)
- [x] Write hybrid search endpoint tests (FR-3, FR-4, FR-7)
- [x] Write BM25 fallback tests (FR-3)
- [x] Write error handling tests (FR-6)
- [x] Write health check tests (FR-8)
- [x] Write pagination and filter tests
- [x] Run tests — expect failures (Red) ✓ 11 failed, 14 passed

## Phase 3: Implementation
- [x] Implement `HybridSearchRequest` schema (FR-1)
- [x] Implement `SearchHit` and `SearchResponse` schemas (FR-2)
- [x] Implement result mapping helper (FR-5)
- [x] Implement search router with hybrid search endpoint (FR-3, FR-4, FR-6, FR-7)
- [x] Implement health check endpoint (FR-8)
- [x] Run tests — expect pass (Green) ✓ 25 passed
- [x] Refactor: fixed DI wrappers in dependency.py for no-arg injection

## Phase 4: Integration
- [x] Register search router in `src/main.py`
- [x] Run lint (`ruff check` + `ruff format`)
- [x] Run full test suite (`pytest`) ✓ 396 passed, 9 skipped

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook: `notebooks/specs/S4.4_hybrid_search.ipynb`
- [x] Update `roadmap.md` status to "done"
- [x] Append summary to `docs/spec-summaries.md`
