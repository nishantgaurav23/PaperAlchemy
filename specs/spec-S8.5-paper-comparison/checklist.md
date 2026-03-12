# Checklist -- Spec S8.5: Side-by-Side Paper Comparison

## Phase 1: Setup & Dependencies
- [x] Verify S8.2 (Paper Summary) is "done"
- [x] Create target files/directories

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_analysis_comparator.py`
- [x] Write failing tests for FR-1 (comparison generation)
- [x] Write failing tests for FR-2 (multi-paper content extraction)
- [x] Write failing tests for FR-3 (API endpoint)
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Add `PaperComparison`, `ComparedPaper`, `ComparisonRequest`, `ComparisonResponse` schemas to `src/schemas/api/analysis.py`
- [x] Implement `ComparatorService` in `src/services/analysis/comparator.py`
- [x] Implement FR-1 -- `compare()` method with structured LLM comparison
- [x] Implement FR-2 -- `_extract_multi_content()` with per-paper labelling and truncation
- [x] Implement LLM output parsing with JSON fallback
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Add `POST /api/v1/papers/compare` endpoint to `src/routers/analysis.py`
- [x] Run lint (`ruff check src/ tests/`)
- [x] Run full test suite (`pytest tests/unit/`)

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook `notebooks/specs/S8.5_comparison.ipynb`
- [x] Update `roadmap.md` status to "done"
- [x] Append spec summary to `docs/spec-summaries.md`
