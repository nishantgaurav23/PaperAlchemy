# Spec S6.1 -- Agent State & Runtime Context

## Overview

Define the foundational data structures for the LangGraph-based agentic RAG system.
This spec creates:
1. **AgentState** — a TypedDict that flows through every LangGraph node, carrying message history, retrieval results, grading outcomes, routing decisions, and citation data.
2. **AgentContext** — a runtime dataclass injected into every node, holding live service clients (LLM, retrieval pipeline, cache) and per-request configuration (model, temperature, thresholds).
3. **Structured output models** — Pydantic models for structured LLM responses (guardrail scoring, document grading, routing decisions, source items).

These structures are the backbone of Phase 6 — every subsequent agent node (S6.2–S6.8) depends on them.

## Dependencies

- **S5.1** (LLM client) — `LLMProvider` protocol used in `AgentContext`

## Target Location

- `src/services/agents/state.py` — AgentState TypedDict
- `src/services/agents/context.py` — AgentContext dataclass
- `src/services/agents/models.py` — Structured output Pydantic models
- `src/services/agents/__init__.py` — Public exports

## Functional Requirements

### FR-1: AgentState TypedDict

- **What**: LangGraph state schema that flows through all agent nodes. Uses `TypedDict` with `total=False` so nodes return partial dicts with only modified fields.
- **Fields**:
  - `messages: Annotated[list[AnyMessage], add_messages]` — Append-only message history (LangGraph reducer)
  - `original_query: str | None` — User's original question
  - `rewritten_query: str | None` — Optimized query after rewrite node
  - `retrieval_attempts: int` — Counter for retrieve→grade→rewrite loops (max 3)
  - `guardrail_result: GuardrailScoring | None` — Domain relevance score (0-100)
  - `routing_decision: str | None` — Current routing: "retrieve", "generate_answer", "rewrite_query", "out_of_scope"
  - `sources: list[SourceItem]` — Retrieved source documents with metadata
  - `grading_results: list[GradingResult]` — Per-document relevance grades
  - `relevant_sources: list[SourceItem]` — Filtered sources that passed grading
  - `metadata: dict[str, Any]` — Extensible metadata bucket (timings, debug info)
- **Edge cases**: All fields except `messages` default to sensible zero-values when not yet set

### FR-2: AgentContext Dataclass

- **What**: Per-request runtime context injected into every LangGraph node via `context=` parameter. Holds live service instances and request-scoped configuration.
- **Fields**:
  - `llm_provider: LLMProvider` — Unified LLM client (Ollama or Gemini)
  - `retrieval_pipeline: RetrievalPipeline | None` — Advanced retrieval (from S4b.5), optional for unit testing
  - `cache_service: CacheService | None` — Redis cache, optional
  - `model_name: str` — LLM model override (default from settings)
  - `temperature: float` — Generation temperature (default 0.7)
  - `top_k: int` — Number of documents to retrieve (default 5)
  - `max_retrieval_attempts: int` — Max retrieve→rewrite loops (default 3)
  - `guardrail_threshold: int` — Minimum relevance score (0-100, default 40)
  - `trace_id: str | None` — Langfuse trace ID for observability
  - `user_id: str` — Requesting user identifier (default "api_user")
- **Edge cases**: Optional services (`retrieval_pipeline`, `cache_service`) gracefully degrade when None

### FR-3: Structured Output Models

- **What**: Pydantic models for structured LLM outputs, used with `llm.with_structured_output(Model)`.
- **Models**:
  - `GuardrailScoring(BaseModel)`: `score: int` (0-100), `reason: str`
  - `GradeDocuments(BaseModel)`: `binary_score: Literal["yes", "no"]`, `reasoning: str`
  - `GradingResult(BaseModel)`: `document_id: str`, `is_relevant: bool`, `score: float`, `reasoning: str`
  - `SourceItem(BaseModel)`: `arxiv_id: str`, `title: str`, `authors: list[str]`, `url: str`, `relevance_score: float`, `chunk_text: str`
  - `RoutingDecision(BaseModel)`: `route: Literal["retrieve", "out_of_scope", "generate_answer", "rewrite_query"]`, `reason: str`
- **Edge cases**: Field constraints enforced (score 0-100, Literal enums), default values for optional fields

### FR-4: State Factory Function

- **What**: `create_initial_state(query: str) -> AgentState` factory that initializes a fresh state with proper defaults for a new query.
- **Inputs**: `query: str` — the user's research question
- **Outputs**: `AgentState` dict with `messages=[HumanMessage(content=query)]` and all other fields at defaults
- **Edge cases**: Empty query raises `ValueError`

### FR-5: Context Factory Function

- **What**: `create_agent_context(llm_provider, **overrides) -> AgentContext` factory for creating runtime context with optional service injection and config overrides.
- **Inputs**: `llm_provider: LLMProvider` (required), optional keyword args for all other fields
- **Outputs**: Configured `AgentContext` instance
- **Edge cases**: Missing `llm_provider` raises `TypeError`

## Tangible Outcomes

- [ ] `src/services/agents/state.py` exists with `AgentState` TypedDict and `create_initial_state()` factory
- [ ] `src/services/agents/context.py` exists with `AgentContext` dataclass and `create_agent_context()` factory
- [ ] `src/services/agents/models.py` exists with all 5 structured output models
- [ ] `src/services/agents/__init__.py` re-exports all public types
- [ ] `AgentState["messages"]` uses `Annotated[..., add_messages]` reducer
- [ ] `AgentState` has `total=False` (all fields optional in partial returns)
- [ ] All models pass Pydantic validation with valid data
- [ ] All models reject invalid data (score out of range, invalid route literal)
- [ ] `create_initial_state("test")` returns state with `HumanMessage` in messages
- [ ] `create_initial_state("")` raises `ValueError`
- [ ] Unit tests cover all models, state creation, context creation, and edge cases
- [ ] Notebook `notebooks/specs/S6.1_agent_state.ipynb` demonstrates all types interactively

## Test-Driven Requirements

### Tests to Write First

1. `test_agent_state_creation`: Verify `create_initial_state()` returns correct TypedDict with HumanMessage
2. `test_agent_state_empty_query`: Verify `create_initial_state("")` raises ValueError
3. `test_agent_state_has_add_messages_reducer`: Verify messages field uses add_messages annotation
4. `test_agent_state_partial_return`: Verify a node can return partial dict (e.g., just `{"routing_decision": "retrieve"}`)
5. `test_guardrail_scoring_valid`: Verify GuardrailScoring(score=75, reason="relevant") works
6. `test_guardrail_scoring_invalid_score`: Verify score outside 0-100 raises ValidationError
7. `test_grade_documents_valid`: Verify GradeDocuments with "yes"/"no" works
8. `test_grade_documents_invalid_score`: Verify invalid literal raises ValidationError
9. `test_source_item_defaults`: Verify SourceItem with minimal fields, defaults populated
10. `test_source_item_to_dict`: Verify SourceItem serializes to dict correctly
11. `test_routing_decision_valid_routes`: Verify all 4 route literals accepted
12. `test_routing_decision_invalid_route`: Verify invalid route raises ValidationError
13. `test_grading_result_fields`: Verify GradingResult with all fields
14. `test_agent_context_creation`: Verify `create_agent_context()` with LLMProvider mock
15. `test_agent_context_defaults`: Verify default values (temperature=0.7, top_k=5, etc.)
16. `test_agent_context_overrides`: Verify keyword overrides work
17. `test_agent_context_optional_services`: Verify None retrieval_pipeline/cache_service is valid

### Mocking Strategy

- Mock `LLMProvider` using a simple class implementing the Protocol
- No external services needed — this spec is pure data structures
- No database, no Redis, no OpenSearch, no HTTP calls

### Coverage

- All public classes and functions tested
- All field constraints tested (valid + invalid)
- All default values verified
- Edge cases: empty strings, None values, boundary values (score=0, score=100)
