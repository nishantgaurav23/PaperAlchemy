# Checklist -- Spec S6.4: Document Grading Node

## Phase 1: Setup & Dependencies
- [x] Verify S6.1 (Agent state & context) is "done"
- [x] Verify `src/services/agents/nodes/` directory exists
- [x] Verify `GradeDocuments`, `GradingResult`, `SourceItem` models exist in `models.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_grade_documents_node.py`
- [x] Write `test_grade_documents_all_relevant`
- [x] Write `test_grade_documents_mixed_relevance`
- [x] Write `test_grade_documents_none_relevant`
- [x] Write `test_grade_documents_empty_sources`
- [x] Write `test_grade_documents_llm_failure`
- [x] Write `test_continue_after_grading_has_relevant`
- [x] Write `test_continue_after_grading_no_relevant`
- [x] Write `test_continue_after_grading_exhausted_retries`
- [x] Write `test_grading_prompt_includes_query_and_chunk`
- [x] Write `test_grading_result_structure`
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement `GRADING_PROMPT` template
- [x] Implement `ainvoke_grade_documents_step` — pass tests
- [x] Implement `continue_after_grading` conditional edge — pass tests
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Export from `src/services/agents/nodes/__init__.py`
- [x] Run lint (`ruff check src/services/agents/nodes/grade_documents_node.py`)
- [x] Run full test suite

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook `notebooks/specs/S6.4_grading.ipynb`
- [x] Update roadmap.md status to "done"
