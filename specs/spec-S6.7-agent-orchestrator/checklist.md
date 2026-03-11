# Checklist -- Spec S6.7: Agent Orchestrator (LangGraph)

## Phase 1: Setup & Dependencies
- [x] Verify S6.2 (guardrail node) is done
- [x] Verify S6.3 (retrieval node) is done
- [x] Verify S6.4 (grading node) is done
- [x] Verify S6.5 (rewrite node) is done
- [x] Verify S6.6 (generation node) is done
- [x] Create `src/services/agents/agentic_rag.py`
- [x] Create `src/services/agents/factory.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_agent_orchestrator.py`
- [x] Write test_graph_compilation
- [x] Write test_happy_path_ask
- [x] Write test_out_of_scope_query
- [x] Write test_rewrite_retry_loop
- [x] Write test_max_retrieval_attempts
- [x] Write test_empty_query_raises
- [x] Write test_extract_answer
- [x] Write test_extract_sources
- [x] Write test_extract_reasoning_steps
- [x] Write test_factory_creates_service
- [x] Write test_factory_requires_llm_provider
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement `AgenticRAGResponse` model
- [x] Implement out_of_scope node function
- [x] Implement `AgenticRAGService.__init__()` with graph construction
- [x] Implement `AgenticRAGService.ask()` method
- [x] Implement `_extract_answer()` helper
- [x] Implement `_extract_sources()` helper
- [x] Implement `_extract_reasoning_steps()` helper
- [x] Implement `create_agentic_rag_service()` factory
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Update `src/services/agents/__init__.py` exports
- [x] Run lint (`ruff check src/services/agents/`)
- [x] Run full test suite (`pytest tests/`)

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook: `notebooks/specs/S6.7_orchestrator.ipynb`
- [x] Update `roadmap.md` status to "done"
