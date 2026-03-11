# Checklist -- Spec S6.5: Query Rewrite Node

## Phase 1: Setup & Dependencies
- [x] Verify S6.1 (agent state & context) is "done"
- [x] Confirm target file path: `src/services/agents/nodes/rewrite_query_node.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_rewrite_query_node.py`
- [x] Write test_rewrite_query_basic
- [x] Write test_rewrite_uses_original_query
- [x] Write test_rewrite_fallback_on_missing_original
- [x] Write test_rewrite_llm_failure_fallback
- [x] Write test_rewrite_structured_output
- [x] Write test_rewrite_temperature_03
- [x] Write test_rewrite_metadata_enrichment
- [x] Write test_rewrite_empty_query_handling
- [x] Write test_rewrite_appends_human_message
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Add QueryRewriteOutput model to models.py or local to node
- [x] Implement ainvoke_rewrite_query_step() with LLM structured output
- [x] Add rewrite prompt template (REWRITE_PROMPT)
- [x] Add fallback logic for LLM failure
- [x] Add metadata enrichment
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Export from `src/services/agents/nodes/__init__.py`
- [x] Run lint (ruff check)
- [x] Run full test suite

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Notebook created: `notebooks/specs/S6.5_rewrite.ipynb`
- [x] Update roadmap.md status to "done"
