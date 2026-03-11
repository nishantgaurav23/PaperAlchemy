# Checklist -- Spec S4.2: Text Chunker (Section-Aware)

## Phase 1: Setup & Dependencies
- [x] Verify S3.3 (PDF Parser) is "done"
- [x] Create `src/services/indexing/` directory
- [x] Create `src/schemas/indexing.py` file

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_text_chunker.py`
- [x] Write failing tests for FR-1 (data models)
- [x] Write failing tests for FR-2 (word-based chunking)
- [x] Write failing tests for FR-3 (section parsing)
- [x] Write failing tests for FR-4 (section filtering)
- [x] Write failing tests for FR-5 (section-based chunking)
- [x] Write failing tests for FR-6 (chunk_paper entry point)
- [x] Write failing tests for FR-7 (configuration validation)
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement FR-1 (ChunkMetadata + TextChunk models) -- pass tests
- [x] Implement FR-2 (chunk_text word-based) -- pass tests
- [x] Implement FR-3 (_parse_sections) -- pass tests
- [x] Implement FR-4 (_filter_sections) -- pass tests
- [x] Implement FR-5 (_chunk_by_sections) -- pass tests
- [x] Implement FR-6 (chunk_paper) -- pass tests
- [x] Implement FR-7 (config validation) -- pass tests
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Create `src/services/indexing/__init__.py` with exports
- [x] Run lint (`ruff check src/services/indexing/ src/schemas/indexing.py`)
- [x] Run full test suite

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Notebook created: `notebooks/specs/S4.2_chunker.ipynb`
- [x] Update roadmap.md status to "done"
