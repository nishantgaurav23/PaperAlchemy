# Spec S4b.5 -- Unified Advanced Retrieval Pipeline

## Overview
Orchestrates the full advanced retrieval pipeline: multi-query expansion → hybrid search → re-ranking → parent chunk expansion → top-K results. This is the single entry point that downstream consumers (RAG chain, agents) call for document retrieval. It composes S4b.1 (reranker), S4b.2 (HyDE), S4b.3 (multi-query), and S4b.4 (parent-child) into a unified, configurable pipeline.

## Dependencies
- S4b.1: Cross-encoder re-ranking (`src/services/reranking/`)
- S4b.2: HyDE retrieval (`src/services/retrieval/hyde.py`)
- S4b.3: Multi-query retrieval (`src/services/retrieval/multi_query.py`)
- S4b.4: Parent-child chunk retrieval (`src/services/indexing/parent_child.py`)

## Target Location
- `src/services/retrieval/pipeline.py`

## Functional Requirements

### FR-1: Pipeline Initialization
- **What**: `RetrievalPipeline` class that accepts all sub-service dependencies
- **Inputs**: `RetrievalPipelineSettings`, `MultiQueryService`, `HyDEService`, `RerankerService`, `ParentChildChunker`, `OpenSearchClient`, `JinaEmbeddingsClient`
- **Outputs**: Configured pipeline instance
- **Edge cases**: Some services may be disabled via settings (e.g., HyDE off, multi-query off)

### FR-2: Full Pipeline Retrieval (`retrieve`)
- **What**: Single async method that orchestrates the full pipeline
- **Inputs**: `query: str`, `top_k: int = 5`, `categories: list[str] | None = None`
- **Outputs**: `RetrievalResult` dataclass with ranked results, metadata, and timing info
- **Pipeline stages**:
  1. **Query Expansion** (parallel): Multi-query generates variations + HyDE generates hypothetical doc
  2. **Hybrid Search** (parallel): Run hybrid search for original + expanded queries + HyDE embedding
  3. **Merge & Deduplicate**: Combine all results, deduplicate by chunk_id, keep highest score
  4. **Re-rank**: Cross-encoder re-scores merged results → top-K
  5. **Parent Expansion**: Expand child chunks to parent sections for richer context
- **Edge cases**: If all expansion fails, fall back to plain hybrid search with original query. If re-ranker fails, return un-reranked results. If parent expansion fails, return child chunks as-is.

### FR-3: Graceful Degradation
- **What**: Each stage can fail independently without killing the pipeline
- **Behavior**:
  - Multi-query disabled/fails → use only original query
  - HyDE disabled/fails → skip hypothetical embedding
  - Re-ranker disabled/fails → return merged results sorted by score
  - Parent expansion disabled/fails → return child chunks
- **Fallback chain**: Always returns *something* — at minimum, a basic hybrid search

### FR-4: Pipeline Configuration
- **What**: `RetrievalPipelineSettings` controls which stages are active
- **Fields**:
  - `multi_query_enabled: bool = True`
  - `hyde_enabled: bool = True`
  - `reranker_enabled: bool = True`
  - `parent_expansion_enabled: bool = True`
  - `retrieval_top_k: int = 20` (pre-rerank fetch count)
  - `final_top_k: int = 5` (post-rerank return count)
- **Edge cases**: All stages disabled → plain hybrid search only

### FR-5: Result Model
- **What**: `RetrievalResult` captures pipeline output + metadata
- **Fields**:
  - `results: list[SearchHit]` — final ranked results
  - `query: str` — original query
  - `expanded_queries: list[str]` — multi-query variations (empty if disabled)
  - `hypothetical_document: str` — HyDE output (empty if disabled)
  - `stages_executed: list[str]` — which stages ran (e.g., ["multi_query", "hyde", "hybrid_search", "rerank", "parent_expand"])
  - `total_candidates: int` — pre-rerank candidate count
  - `timings: dict[str, float]` — per-stage timing in seconds

### FR-6: Factory Function
- **What**: `create_retrieval_pipeline()` wires all dependencies
- **Inputs**: Settings + all sub-services
- **Outputs**: Configured `RetrievalPipeline`
- **Location**: Added to `src/services/retrieval/factory.py`

## Tangible Outcomes
- [ ] `src/services/retrieval/pipeline.py` exists with `RetrievalPipeline` class
- [ ] `RetrievalPipeline.retrieve()` orchestrates multi-query → hybrid → rerank → parent-expand
- [ ] Each stage can be independently enabled/disabled via settings
- [ ] Graceful fallback: pipeline never raises, always returns results (at minimum basic search)
- [ ] `RetrievalResult` includes results, metadata, expanded queries, timings
- [ ] Factory function in `src/services/retrieval/factory.py`
- [ ] `RetrievalPipelineSettings` in `src/config.py`
- [ ] All public methods tested with mocked dependencies
- [ ] Notebook: `notebooks/specs/S4b.5_retrieval_pipeline.ipynb`

## Test-Driven Requirements

### Tests to Write First
1. `test_pipeline_init`: Pipeline accepts all dependencies
2. `test_full_pipeline_all_stages`: All stages execute in order with correct data flow
3. `test_pipeline_multi_query_disabled`: Skips multi-query, uses original query only
4. `test_pipeline_hyde_disabled`: Skips HyDE stage
5. `test_pipeline_reranker_disabled`: Returns merged results without re-ranking
6. `test_pipeline_parent_expansion_disabled`: Returns child chunks as-is
7. `test_pipeline_all_disabled`: Falls back to plain hybrid search
8. `test_pipeline_multi_query_failure`: Graceful fallback on multi-query error
9. `test_pipeline_hyde_failure`: Graceful fallback on HyDE error
10. `test_pipeline_reranker_failure`: Graceful fallback on reranker error
11. `test_pipeline_parent_expansion_failure`: Graceful fallback on parent error
12. `test_pipeline_deduplication`: Merges results from multiple queries, deduplicates by chunk_id
13. `test_pipeline_timing_metadata`: Timings dict populated for each executed stage
14. `test_pipeline_stages_executed_tracking`: stages_executed list reflects actual execution
15. `test_pipeline_top_k_respected`: Final results count <= final_top_k
16. `test_factory_function`: Factory creates pipeline with all services

### Mocking Strategy
- Mock `MultiQueryService.retrieve_with_multi_query()` → returns `MultiQueryResult`
- Mock `HyDEService.retrieve_with_hyde()` → returns `HyDEResult`
- Mock `RerankerService.rerank_search_hits()` → returns re-scored `list[SearchHit]`
- Mock `ParentChildChunker.expand_to_parents()` → returns expanded results
- Mock `OpenSearchClient.search_chunks_hybrid()` → returns search results
- Mock `JinaEmbeddingsClient.embed_query()` → returns 1024-dim vector

### Coverage
- All public functions tested
- All degradation paths (each stage failing independently)
- Edge cases: empty results, duplicate chunks, all stages disabled
- Timing metadata accuracy
