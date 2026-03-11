# Checklist -- Spec S5.2: RAG Chain

## Phase 1: Setup & Dependencies
- [x] Verify S5.1 (LLM Client) is "done"
- [x] Verify S4b.5 (Retrieval Pipeline) is "done"
- [x] Create `src/services/rag/` directory
- [x] Create target files: chain.py, prompts.py, models.py, factory.py, __init__.py

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_rag_chain.py`
- [x] Write test_rag_chain_returns_answer_with_sources
- [x] Write test_rag_chain_answer_contains_citations
- [x] Write test_rag_chain_sources_have_metadata
- [x] Write test_rag_chain_empty_query_raises_error
- [x] Write test_rag_chain_no_documents_found
- [x] Write test_rag_chain_llm_error_propagates
- [x] Write test_rag_chain_prompt_includes_context
- [x] Write test_rag_chain_prompt_enforces_citations
- [x] Write test_rag_chain_streaming_yields_tokens
- [x] Write test_rag_chain_streaming_ends_with_sources
- [x] Write test_rag_chain_categories_passed_to_retrieval
- [x] Write test_rag_chain_temperature_passed_to_llm
- [x] Write test_factory_creates_rag_chain
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement RAG response models (models.py)
- [x] Implement prompt templates (prompts.py)
- [x] Implement RAGChain.aquery() (chain.py)
- [x] Implement RAGChain.aquery_stream() (chain.py)
- [x] Implement factory function (factory.py)
- [x] Set up __init__.py exports
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Wire into dependency injection (src/dependency.py)
- [x] Run lint (ruff check)
- [x] Run full test suite (542 passed)

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook: notebooks/specs/S5.2_rag_chain.ipynb
- [x] Update roadmap.md status to "done"
- [x] Append spec summary to docs/spec-summaries.md
