# Checklist -- Spec S4.1: OpenSearch Client + Index Configuration

## Phase 1: Setup & Dependencies
- [x] Verify S1.3 (Docker infrastructure) is "done"
- [x] Create `src/services/opensearch/` directory
- [x] Create `__init__.py`, `index_config.py`, `query_builder.py`, `client.py`, `factory.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_opensearch_client.py`
- [x] Write tests for index_config (mapping structure, RRF pipeline)
- [x] Write tests for QueryBuilder (BM25, categories, chunk mode, empty query, sort, source)
- [x] Write tests for OpenSearchClient (health, setup, search BM25/vector/hybrid/unified, bulk, delete, get chunks, stats)
- [x] Write tests for factory functions (cached singleton, fresh instance)
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement index_config.py (ARXIV_PAPERS_CHUNKS_MAPPING, HYBRID_RRF_PIPELINE)
- [x] Implement query_builder.py (QueryBuilder class with build, text query, filters, highlight, sort)
- [x] Implement client.py (OpenSearchClient with all FR methods)
- [x] Implement factory.py (make_opensearch_client, make_opensearch_client_fresh)
- [x] Implement __init__.py (exports)
- [x] Run tests -- expect pass (Green) -- 40/40 passed
- [x] Refactor if needed (ruff auto-fix applied)

## Phase 4: Integration
- [x] Wire into dependency.py (add OpenSearchDep)
- [x] Run lint: `ruff check src/services/opensearch/` -- All checks passed
- [x] Run full test suite: `pytest tests/ -v` -- 305 passed, 9 skipped

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Notebook created: `notebooks/specs/S4.1_opensearch.ipynb`
- [x] Update roadmap.md status to "done"
