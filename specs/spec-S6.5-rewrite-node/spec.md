# Spec S6.5 -- Query Rewrite Node

## Overview
Agent node that optimizes queries when document grading finds insufficient relevant results. Uses LLM to rewrite the user's query with synonym expansion, context refinement, and academic terminology enrichment. Operates in a rewrite -> re-retrieve loop with a maximum of 3 retries to prevent infinite cycles.

The node reads `original_query` (not the latest message) to prevent semantic drift across multiple rewrites. After rewriting, it stores the rewritten query in state and appends a HumanMessage so the retrieval node picks it up on the next iteration.

## Dependencies
- S6.1 (Agent state & context) -- AgentState, AgentContext, models

## Target Location
- `src/services/agents/nodes/rewrite_query_node.py`

## Functional Requirements

### FR-1: Query Rewrite via LLM
- **What**: Use structured LLM output to rewrite a query for better retrieval
- **Inputs**: AgentState (with original_query, messages, retrieval_attempts), AgentContext (with llm_provider)
- **Outputs**: Partial state dict with `rewritten_query` (str), `messages` (appended HumanMessage)
- **Behavior**:
  - Extract the original query from `state["original_query"]` (prevents drift on multi-rewrite)
  - Fallback to `get_latest_query(state["messages"])` if original_query is missing
  - Use temperature=0.3 (controlled creativity for synonyms/expansion)
  - Return structured output: `QueryRewriteOutput(rewritten_query=str, reasoning=str)`
  - Append rewritten query as `HumanMessage` to messages (picked up by retrieval node next)
  - Store rewritten query in `state["rewritten_query"]`
- **Edge cases**:
  - LLM failure: fallback to keyword expansion (append "research paper arxiv" to original query)
  - Empty query: return original query unchanged
  - Query too short (<3 chars): return with generic academic expansion

### FR-2: Rewrite Prompt Engineering
- **What**: Prompt template that guides the LLM to expand and refine queries
- **Prompt strategy**:
  - System context: "You are a query optimization specialist for academic paper retrieval"
  - Instructions: expand abbreviations, add synonyms, include related technical terms
  - Constraint: keep query focused on the original intent (don't drift)
  - Output: improved query string + brief reasoning
- **Edge cases**:
  - Already well-formed queries: minimal changes expected
  - Domain-specific jargon: expand with full terms (e.g., "NLP" -> "natural language processing NLP")

### FR-3: Metadata Enrichment
- **What**: Track rewrite history in state metadata for observability
- **Outputs**: `metadata["rewrite"]` dict with:
  - `original_query`: the input query
  - `rewritten_query`: the output query
  - `reasoning`: why the rewrite was made
  - `attempt_number`: which rewrite attempt this is
- **Edge cases**: Missing metadata dict -- create fresh

## Tangible Outcomes
- [ ] `ainvoke_rewrite_query_step(state, context)` function exists and is async
- [ ] Returns partial state with `rewritten_query` and `messages` (HumanMessage appended)
- [ ] Uses structured LLM output (`QueryRewriteOutput` Pydantic model)
- [ ] Temperature set to 0.3 for controlled creativity
- [ ] Reads `original_query` to prevent semantic drift
- [ ] Graceful fallback on LLM failure (keyword expansion)
- [ ] Metadata enriched with rewrite details
- [ ] Exported from `src/services/agents/nodes/__init__.py`
- [ ] All tests pass with mocked LLM

## Test-Driven Requirements

### Tests to Write First
1. `test_rewrite_query_basic`: Verify rewrite returns partial state with rewritten_query and messages
2. `test_rewrite_uses_original_query`: Confirm it reads original_query, not latest message
3. `test_rewrite_fallback_on_missing_original`: Falls back to latest message when original_query absent
4. `test_rewrite_llm_failure_fallback`: On LLM exception, returns keyword-expanded query
5. `test_rewrite_structured_output`: Verify QueryRewriteOutput model is used
6. `test_rewrite_temperature_03`: Confirm LLM called with temperature=0.3
7. `test_rewrite_metadata_enrichment`: Check metadata["rewrite"] has all fields
8. `test_rewrite_empty_query_handling`: Empty query returns unchanged
9. `test_rewrite_appends_human_message`: Verify HumanMessage is appended to messages

### Mocking Strategy
- Mock `context.llm_provider.get_langchain_model()` to return a mock LLM
- Mock `structured_llm.ainvoke()` to return `QueryRewriteOutput`
- Use `AsyncMock` for async calls
- No external services needed (pure LLM call)

### Coverage
- All public functions tested
- LLM success and failure paths
- Edge cases: empty query, missing original_query, missing metadata
- Structured output validation
