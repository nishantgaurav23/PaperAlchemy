# Checklist -- Spec S4b.2: HyDE (Hypothetical Document Embeddings)

## Phase 1: Setup & Dependencies
- [x] Verify S4.4 (hybrid search) is "done"
- [x] Verify S5.1 (LLM client) is "done"
- [x] Create `src/services/retrieval/` package
- [x] Add `HyDESettings` to `src/config.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_hyde.py`
- [x] Write failing tests for hypothetical document generation (FR-2)
- [x] Write failing tests for HyDE embedding (FR-3)
- [x] Write failing tests for vector search with HyDE (FR-4)
- [x] Write failing tests for fallback behavior (FR-5)
- [x] Write failing tests for full flow (FR-1)
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement `HyDEResult` schema
- [x] Implement `HyDEService.__init__()` with dependencies
- [x] Implement `generate_hypothetical_document()` (FR-2)
- [x] Implement `_embed_hypothetical()` (FR-3)
- [x] Implement `retrieve_with_hyde()` full flow (FR-1, FR-4)
- [x] Implement fallback logic (FR-5)
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Create factory function (FR-7)
- [x] Add `HyDESettings` to root `Settings` class
- [x] Wire into `src/services/retrieval/__init__.py`
- [x] Add `HyDEDep` to `src/dependency.py`
- [x] Run lint (ruff check)
- [x] Run full test suite (460 passed)

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Notebook created at `notebooks/specs/S4b.2_hyde.ipynb`
- [x] Update roadmap.md status to "done"
