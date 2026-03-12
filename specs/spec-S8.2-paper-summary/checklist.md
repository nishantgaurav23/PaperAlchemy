# Checklist -- Spec S8.2: AI-Generated Paper Summary

## Phase 1: Setup & Dependencies
- [x] Verify S8.1 (PDF Upload) is "done"
- [x] Verify S5.1 (LLM Client) is "done"
- [x] Create `src/services/analysis/__init__.py`
- [x] Create `src/services/analysis/summarizer.py`
- [x] Create `src/schemas/api/analysis.py`
- [x] Create `src/routers/analysis.py`
- [x] Create `tests/unit/test_analysis_summarizer.py`

## Phase 2: Tests First (TDD)
- [x] Write `test_summarize_full_paper`
- [x] Write `test_summarize_abstract_only`
- [x] Write `test_summarize_paper_not_found`
- [x] Write `test_summarize_insufficient_content`
- [x] Write `test_summarize_llm_failure`
- [x] Write `test_extract_content_full_paper`
- [x] Write `test_extract_content_truncation`
- [x] Write `test_summary_response_schema`
- [x] Write `test_summary_endpoint_success`
- [x] Write `test_summary_endpoint_not_found`
- [x] Write `test_force_regeneration`
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement `PaperSummary` and `SummaryResponse` Pydantic schemas
- [x] Implement content extraction (`_extract_content`)
- [x] Implement summary prompt template
- [x] Implement `SummarizerService.summarize()` -- pass tests
- [x] Implement `POST /api/v1/papers/{paper_id}/summary` endpoint
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Wire router into `src/main.py`
- [x] Add DI dependencies to `src/dependency.py` (if needed)
- [x] Run lint (`ruff check src/ tests/`)
- [x] Run full test suite

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook `notebooks/specs/S8.2_summary.ipynb`
- [x] Update `roadmap.md` status to "done"
- [x] Append spec summary to `docs/spec-summaries.md`
