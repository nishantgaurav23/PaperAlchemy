# Checklist -- Spec S8.3: Key Highlights Extraction

## Phase 1: Setup & Dependencies
- [x] Verify S8.1 (PDF Upload) is "done"
- [x] Verify S5.1 (LLM Client) is "done"
- [x] Create/verify target files: `src/services/analysis/highlights.py`

## Phase 2: Tests First (TDD)
- [x] Create test file: `tests/unit/test_analysis_highlights.py`
- [x] Write failing tests for FR-1 (highlights extraction — all 5 fields)
- [x] Write failing tests for FR-2 (content preparation)
- [x] Write failing tests for FR-3 (API endpoint)
- [x] Write failing tests for FR-4 (caching/force regeneration)
- [x] Run tests -- expect failures (Red) — 15/15 FAILED

## Phase 3: Implementation
- [x] Implement `PaperHighlights` + `HighlightsResponse` Pydantic schemas
- [x] Implement `HighlightsService.extract_highlights()` -- pass tests
- [x] Implement `HighlightsService._prepare_content()` -- pass tests
- [x] Implement LLM prompt with structured JSON output parsing
- [x] Implement `POST /api/v1/papers/{paper_id}/highlights` endpoint
- [x] Run tests -- expect pass (Green) — 15/15 PASSED
- [x] Refactor if needed (clean)

## Phase 4: Integration
- [x] Wire endpoint into analysis router
- [x] Register in FastAPI app (already registered)
- [x] Run lint (ruff check) — all clean
- [x] Run full test suite — 944 passed, 9 skipped

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook: `notebooks/specs/S8.3_highlights.ipynb`
- [x] Update roadmap.md status to "done"
- [x] Append spec summary to `docs/spec-summaries.md`
