# Spec S6.7 -- Agent Orchestrator (LangGraph)

## Overview
Compile a LangGraph `StateGraph` that wires together the guardrail, retrieval, grading, rewrite, and generation nodes (S6.2-S6.6) into a complete agentic RAG workflow. The graph is compiled **once at startup** and reused for every request. The orchestrator exposes a single `ask()` entry point that returns a citation-backed answer with sources and reasoning steps.

## Dependencies
- **S6.2** Guardrail node (`ainvoke_guardrail_step`, `continue_after_guardrail`)
- **S6.3** Retrieval node (`ainvoke_retrieve_step`)
- **S6.4** Grading node (`ainvoke_grade_documents_step`, `continue_after_grading`)
- **S6.5** Rewrite node (`ainvoke_rewrite_query_step`)
- **S6.6** Generation node (`ainvoke_generate_answer_step`)

## Target Location
- `src/services/agents/agentic_rag.py` — `AgenticRAGService` class
- `src/services/agents/factory.py` — Factory function for DI
- `src/services/agents/__init__.py` — Updated exports

## Functional Requirements

### FR-1: StateGraph Construction
- **What**: Build a `langgraph.graph.StateGraph[AgentState]` with `context_schema=AgentContext`
- **Nodes**: guardrail, out_of_scope, retrieve, grade_documents, rewrite_query, generate_answer
- **Edges**:
  - `START → guardrail`
  - `guardrail → conditional(continue_after_guardrail) → {continue: retrieve, out_of_scope: out_of_scope}`
  - `out_of_scope → END`
  - `retrieve → grade_documents`
  - `grade_documents → conditional(continue_after_grading) → {generate: generate_answer, rewrite: rewrite_query}`
  - `rewrite_query → retrieve` (loop back)
  - `generate_answer → END`
- **Compilation**: `graph.compile()` called once in `__init__`, stored as `self._compiled_graph`
- **Edge cases**: Graph compilation failure raises `AgenticRAGError`

### FR-2: Public `ask()` Method
- **What**: Entry point for all research questions
- **Inputs**: `query: str`, `user_id: str = "api_user"`, `model_name: str | None = None`, `top_k: int | None = None`
- **Outputs**: `AgenticRAGResponse` with fields:
  - `answer: str` — Citation-backed response text with inline [N] refs + source list
  - `sources: list[SourceReference]` — Structured source list (title, authors, arxiv_url)
  - `reasoning_steps: list[str]` — Chronological trace of node decisions
  - `metadata: dict[str, Any]` — Timing, retrieval stats, grading results
- **Behavior**:
  1. Create `AgentContext` with `llm_provider`, `retrieval_pipeline`, per-request overrides
  2. Create initial state via `create_initial_state(query)`
  3. Invoke compiled graph: `await self._compiled_graph.ainvoke(state, config={"configurable": {"context": context}})`
  4. Extract answer from last `AIMessage` in final state
  5. Extract sources from `relevant_sources` in final state
  6. Build reasoning steps from metadata + state transitions
  7. Return `AgenticRAGResponse`
- **Edge cases**:
  - Empty query → raise `ValueError`
  - Graph execution timeout → raise `AgenticRAGError` with timeout info
  - LLM provider unavailable → error propagated in metadata, fallback answer returned

### FR-3: Out-of-Scope Handler
- **What**: Simple node that returns a polite rejection for off-topic queries
- **Output**: `AIMessage` with content: "I'm a research assistant focused on academic papers. I can't help with that topic, but I'd be happy to answer questions about scientific research papers in my knowledge base."
- **State update**: Sets `messages` with rejection AIMessage

### FR-4: Result Extraction Helpers
- **What**: Private methods to extract structured results from final graph state
- `_extract_answer(state) → str` — Last AIMessage content, or fallback error message
- `_extract_sources(state) → list[SourceReference]` — Convert `relevant_sources` SourceItems to SourceReferences
- `_extract_reasoning_steps(state) → list[str]` — Human-readable list of what happened:
  - "Guardrail: passed (score=75, reason=...)"
  - "Retrieved 8 documents (attempt 1/3)"
  - "Grading: 5/8 relevant"
  - "Generated answer with 5 citations"

### FR-5: Factory Function
- **What**: `create_agentic_rag_service(llm_provider, retrieval_pipeline, cache_service=None) → AgenticRAGService`
- **Purpose**: DI-friendly construction for FastAPI `Depends()`
- **Validates**: `llm_provider` is not None (required)

## Tangible Outcomes
- [ ] `AgenticRAGService` class with compiled LangGraph in `src/services/agents/agentic_rag.py`
- [ ] Factory in `src/services/agents/factory.py`
- [ ] `ask()` returns `AgenticRAGResponse` with answer, sources, reasoning_steps, metadata
- [ ] Graph routes: guardrail → retrieve → grade → generate (happy path)
- [ ] Graph routes: guardrail → out_of_scope (off-topic queries)
- [ ] Graph routes: grade → rewrite → retrieve → grade → generate (retry loop)
- [ ] Max retrieval attempts enforced (no infinite loops)
- [ ] All external services mocked in tests
- [ ] Notebook: `notebooks/specs/S6.7_orchestrator.ipynb`

## Test-Driven Requirements

### Tests to Write First
1. `test_graph_compilation`: Graph compiles without error, has expected nodes
2. `test_happy_path_ask`: Query → guardrail pass → retrieve → grade relevant → generate with citations
3. `test_out_of_scope_query`: Off-topic query → guardrail reject → out_of_scope message → END
4. `test_rewrite_retry_loop`: All docs grade irrelevant → rewrite → re-retrieve → grade relevant → generate
5. `test_max_retrieval_attempts`: Rewrite loop stops after max attempts, forces generation
6. `test_empty_query_raises`: Empty string query raises ValueError
7. `test_extract_answer`: Extracts last AIMessage content from final state
8. `test_extract_sources`: Converts SourceItems to SourceReferences correctly
9. `test_extract_reasoning_steps`: Builds human-readable step list from metadata
10. `test_factory_creates_service`: Factory returns valid AgenticRAGService
11. `test_factory_requires_llm_provider`: Factory raises without llm_provider

### Mocking Strategy
- Mock `llm_provider.get_langchain_model()` → returns mock ChatModel with `.ainvoke()` and `.with_structured_output()`
- Mock `retrieval_pipeline.retrieve()` → returns `RetrievalResult` with fake SearchHits
- Mock all LLM structured outputs (GuardrailScoring, GradeDocuments, QueryRewriteOutput)
- Use `AsyncMock` for all async operations
- No real OpenSearch, Redis, Jina, Ollama, or Gemini calls

### Coverage
- All public methods tested
- All graph routing paths covered (happy, out-of-scope, rewrite loop, max attempts)
- Error paths tested (empty query, graph failure)
- Result extraction edge cases (empty state, missing fields)
