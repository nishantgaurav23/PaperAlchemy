# Checklist -- Spec S4b.1: Cross-Encoder Re-ranking

## Phase 1: Setup & Dependencies
- [x] Verify S4.4 (hybrid search) is "done"
- [x] Verify `sentence-transformers` in pyproject.toml dependencies
- [x] Create `src/services/reranking/` directory
- [x] Update `RerankerSettings` in `src/config.py` with provider + cohere_api_key fields

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_reranker.py`
- [x] Write failing tests for FR-1 (rerank interface)
- [x] Write failing tests for FR-2 (local cross-encoder scoring)
- [x] Write failing tests for FR-4 (SearchHit re-ranking)
- [x] Write failing tests for FR-6 (factory function)
- [x] Run tests — expect failures (Red) — 18 failed

## Phase 3: Implementation
- [x] Create `src/services/reranking/__init__.py` with exports
- [x] Implement `RerankResult` dataclass inline in service.py
- [x] Implement `RerankerService` in `src/services/reranking/service.py` — pass tests
- [x] Implement `create_reranker_service()` in `src/services/reranking/factory.py` — pass tests
- [x] Run tests — expect pass (Green) — 18 passed
- [x] No refactoring needed

## Phase 4: Integration
- [x] Add `RerankerDep` to `src/dependency.py`
- [x] Wire factory into app lifespan or DI
- [x] Run lint (`ruff check src/ tests/`) — all checks passed
- [x] Run full test suite — 414 passed, 9 skipped

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook: `notebooks/specs/S4b.1_reranker.ipynb`
- [x] Update roadmap.md status to "done"
