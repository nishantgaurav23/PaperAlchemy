# Spec S6.3 -- Retrieval Node (Tool-Based Document Retrieval)

## Overview
The retrieval node is the second major node in the agentic RAG LangGraph workflow (after the guardrail node). It uses the advanced retrieval pipeline (S4b.5) to fetch relevant documents for the user's query and populates the agent state with `SourceItem` results for downstream grading and generation.

**Key principle**: Retrieval is MANDATORY. The agent must always invoke the retrieval pipeline as a tool call — never skip it or answer from LLM memory alone. If retrieval fails, the node must populate an error state rather than silently continuing.

## Dependencies
- **S6.1** (Agent state & context) — `AgentState`, `AgentContext`, `SourceItem`
- **S4b.5** (Unified retrieval pipeline) — `RetrievalPipeline.retrieve()`

## Target Location
- `src/services/agents/nodes/retrieve_node.py`

## Functional Requirements

### FR-1: Retrieve Documents via Pipeline
- **What**: Async function `ainvoke_retrieve_step(state, context)` that calls `context.retrieval_pipeline.retrieve()` with the current query (rewritten_query if set, otherwise original_query) and converts results to `SourceItem` objects.
- **Inputs**: `AgentState` (reads `rewritten_query`, `original_query`, `messages`, `retrieval_attempts`), `AgentContext` (reads `retrieval_pipeline`, `top_k`)
- **Outputs**: Partial state dict with `sources: list[SourceItem]`, `retrieval_attempts: int` (incremented), `metadata: dict` (pipeline timings/stages)
- **Edge cases**:
  - `retrieval_pipeline` is `None` → raise or return empty sources with error metadata
  - Pipeline returns zero results → return empty `sources`, log warning
  - Pipeline raises exception → catch, log, return empty `sources` with error in metadata
  - Query is empty/whitespace → fall back to extracting from messages via `get_latest_query()`

### FR-2: SearchHit → SourceItem Conversion
- **What**: Convert `RetrievalResult.results` (list of `SearchHit`) into `list[SourceItem]` with proper field mapping.
- **Mapping**:
  - `SearchHit.arxiv_id` → `SourceItem.arxiv_id`
  - `SearchHit.title` → `SourceItem.title`
  - `SearchHit.authors` → `SourceItem.authors`
  - `SearchHit.pdf_url` or constructed `https://arxiv.org/abs/{arxiv_id}` → `SourceItem.url`
  - `SearchHit.score` → `SourceItem.relevance_score`
  - `SearchHit.chunk_text` → `SourceItem.chunk_text`
- **Edge cases**: Missing `arxiv_id` → skip that hit; missing fields → use defaults

### FR-3: Retrieval Attempt Tracking
- **What**: Increment `retrieval_attempts` by 1 on each invocation. This supports the rewrite-retry loop (max 3 attempts enforced by orchestrator).
- **Inputs**: `state["retrieval_attempts"]` (defaults to 0)
- **Outputs**: `retrieval_attempts` incremented by 1

### FR-4: Metadata Enrichment
- **What**: Store pipeline execution metadata in `state["metadata"]` under the key `"retrieval"`.
- **Data**: `stages_executed`, `total_candidates`, `timings`, `query_used`, `num_results`
- **Purpose**: Observability and debugging — downstream nodes and tracing can inspect this.

## Tangible Outcomes
- [ ] `src/services/agents/nodes/retrieve_node.py` exists with `ainvoke_retrieve_step` async function
- [ ] Conversion from `SearchHit` → `SourceItem` works correctly for all field mappings
- [ ] Retrieval attempts are incremented on each call
- [ ] Pipeline metadata (stages, timings, candidates) stored in state metadata
- [ ] Graceful handling when pipeline is None, returns empty, or raises
- [ ] Node is exported from `src/services/agents/nodes/__init__.py`
- [ ] All tests pass in `tests/unit/test_retrieve_node.py`

## Test-Driven Requirements

### Tests to Write First
1. `test_retrieve_step_returns_sources`: Given a mock pipeline returning 3 SearchHits, verify the node returns 3 SourceItems with correct field mapping
2. `test_retrieve_step_uses_rewritten_query`: When `rewritten_query` is set, pipeline is called with it (not `original_query`)
3. `test_retrieve_step_falls_back_to_original_query`: When `rewritten_query` is None, uses `original_query`
4. `test_retrieve_step_increments_attempts`: Verify `retrieval_attempts` goes from 0 to 1
5. `test_retrieve_step_pipeline_none`: When `retrieval_pipeline` is None, returns empty sources with error metadata
6. `test_retrieve_step_pipeline_exception`: When pipeline raises, returns empty sources, logs error
7. `test_retrieve_step_empty_results`: When pipeline returns zero hits, sources is empty list
8. `test_search_hit_to_source_item_mapping`: Verify each field maps correctly including URL construction
9. `test_search_hit_missing_arxiv_id_skipped`: Hits with empty arxiv_id are filtered out
10. `test_metadata_enrichment`: Verify stages_executed, timings, total_candidates are stored in metadata

### Mocking Strategy
- Mock `RetrievalPipeline.retrieve()` as `AsyncMock` returning a `RetrievalResult`
- Mock `AgentContext` with mock pipeline and `top_k=5`
- Use fixture `AgentState` created via `create_initial_state()`
- No external services needed — everything is mocked

### Coverage
- All public functions tested
- Edge cases: None pipeline, exception, empty results, missing fields
- Error paths: pipeline failure, missing query
