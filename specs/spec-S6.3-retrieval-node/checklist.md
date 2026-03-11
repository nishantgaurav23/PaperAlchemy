# Checklist -- Spec S6.3: Retrieval Node

## Phase 1: Setup & Dependencies
- [x] Verify S6.1 (agent state & context) is "done"
- [x] Verify S4b.5 (retrieval pipeline) is "done"
- [x] Create `src/services/agents/nodes/retrieve_node.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_retrieve_node.py`
- [x] Write failing tests for FR-1 (retrieve documents via pipeline)
- [x] Write failing tests for FR-2 (SearchHit → SourceItem conversion)
- [x] Write failing tests for FR-3 (retrieval attempt tracking)
- [x] Write failing tests for FR-4 (metadata enrichment)
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement `convert_search_hits_to_sources()` helper — pass FR-2 tests
- [x] Implement `ainvoke_retrieve_step()` — pass FR-1, FR-3, FR-4 tests
- [x] Handle edge cases (None pipeline, exception, empty results)
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Export from `src/services/agents/nodes/__init__.py`
- [x] Run lint (`ruff check src/services/agents/nodes/retrieve_node.py`)
- [x] Run full test suite

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook `notebooks/specs/S6.3_retrieval.ipynb`
- [x] Update roadmap.md status to "done"
