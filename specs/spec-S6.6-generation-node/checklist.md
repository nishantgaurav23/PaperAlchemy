# Checklist -- Spec S6.6: Answer Generation Node

## Phase 1: Setup & Dependencies
- [x] Verify S6.1 (agent state & context) is "done"
- [x] Verify S5.5 (citation enforcement) is "done"
- [x] Create `src/services/agents/nodes/generate_answer_node.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_generate_answer_node.py`
- [x] Write test_generation_prompt_includes_sources
- [x] Write test_generate_answer_with_sources
- [x] Write test_generate_answer_no_sources_fallback
- [x] Write test_generate_answer_llm_failure
- [x] Write test_citation_post_processing
- [x] Write test_uses_rewritten_query_when_available
- [x] Write test_metadata_includes_citation_info
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement FR-1: Generation prompt construction
- [x] Implement FR-2: LLM answer generation
- [x] Implement FR-3: Citation post-processing (SourceItem -> SourceReference + enforce_citations)
- [x] Implement FR-4: State update with AIMessage
- [x] Implement FR-5: No-sources fallback
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Register in `src/services/agents/nodes/__init__.py`
- [x] Run lint (ruff check)
- [x] Run full test suite (751 passed, 9 skipped)

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Notebook created: `notebooks/specs/S6.6_generation.ipynb`
- [x] Update roadmap.md status to "done"
