# Checklist -- Spec S4b.5: Unified Advanced Retrieval Pipeline

## Phase 1: Setup & Dependencies
- [x] Verify S4b.1 (reranker) is "done"
- [x] Verify S4b.2 (HyDE) is "done"
- [x] Verify S4b.3 (multi-query) is "done"
- [x] Verify S4b.4 (parent-child) is "done"
- [x] Add `RetrievalPipelineSettings` to `src/config.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_retrieval_pipeline.py`
- [x] Write test_pipeline_init
- [x] Write test_full_pipeline_all_stages
- [x] Write test_pipeline_multi_query_disabled
- [x] Write test_pipeline_hyde_disabled
- [x] Write test_pipeline_reranker_disabled
- [x] Write test_pipeline_parent_expansion_disabled
- [x] Write test_pipeline_all_disabled
- [x] Write test_pipeline_multi_query_failure
- [x] Write test_pipeline_hyde_failure
- [x] Write test_pipeline_reranker_failure
- [x] Write test_pipeline_parent_expansion_failure
- [x] Write test_pipeline_deduplication
- [x] Write test_pipeline_timing_metadata
- [x] Write test_pipeline_stages_executed_tracking
- [x] Write test_pipeline_top_k_respected
- [x] Write test_factory_function
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Create `src/services/retrieval/pipeline.py` with `RetrievalPipeline` class
- [x] Implement `RetrievalResult` dataclass
- [x] Implement `retrieve()` method — full pipeline orchestration
- [x] Implement graceful degradation for each stage
- [x] Implement deduplication logic
- [x] Implement timing metadata collection
- [x] Add factory function to `src/services/retrieval/factory.py`
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Update `src/services/retrieval/__init__.py` exports
- [x] Run lint (`ruff check src/services/retrieval/pipeline.py`)
- [x] Run full test suite

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook: `notebooks/specs/S4b.5_retrieval_pipeline.ipynb`
- [x] Update roadmap.md status to "done"
- [x] Append spec summary to `docs/spec-summaries.md`
