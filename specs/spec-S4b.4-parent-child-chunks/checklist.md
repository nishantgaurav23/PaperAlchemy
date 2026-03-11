# Checklist -- Spec S4b.4: Parent-Child Chunk Retrieval

## Phase 1: Setup & Dependencies
- [x] Verify S4.2 (Text chunker) is "done"
- [x] Create `src/services/indexing/parent_child.py`
- [x] Add `ChildChunk` and `ParentChildResult` to `src/schemas/indexing.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_parent_child.py`
- [x] Write failing tests for FR-1 (initialization + validation)
- [x] Write failing tests for FR-2 (create parent-child chunks)
- [x] Write failing tests for FR-3 (ChildChunk schema)
- [x] Write failing tests for FR-4 (split parent into children)
- [x] Write failing tests for FR-5 (parent expansion / dedup)
- [x] Write failing tests for FR-6 (prepare for indexing)
- [x] Run tests -- expect failures (Red) — 22 FAILED, 3 passed

## Phase 3: Implementation
- [x] Add `ChildChunk` and `ParentChildResult` schemas
- [x] Implement `ParentChildChunker.__init__()` with validation -- pass FR-1 tests
- [x] Implement `split_parent_into_children()` -- pass FR-4 tests
- [x] Implement `create_parent_child_chunks()` -- pass FR-2 tests
- [x] Implement `expand_to_parents()` -- pass FR-5 tests
- [x] Implement `prepare_for_indexing()` -- pass FR-6 tests
- [x] Run tests -- expect pass (Green) — 25 passed
- [x] Refactor if needed — clean, no refactoring needed

## Phase 4: Integration
- [x] Export from `src/services/indexing/__init__.py`
- [x] Run lint (`ruff check` + `ruff format`) — clean
- [x] Run full test suite — 66 passed (25 parent-child + 41 text chunker)

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook: `notebooks/specs/S4b.4_parent_child.ipynb`
- [x] Update roadmap.md status to "done"
