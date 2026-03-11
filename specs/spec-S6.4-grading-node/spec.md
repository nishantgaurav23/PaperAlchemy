# Spec S6.4 -- Document Grading Node

## Overview
Binary relevance grading node for the agentic RAG LangGraph workflow. After retrieval, this node evaluates each retrieved document against the user's query using an LLM with structured output (`GradeDocuments`). Documents scoring "yes" are promoted to `relevant_sources`; those scoring "no" are filtered out. If no documents pass grading, the workflow routes to query rewriting.

## Dependencies
- **S6.1** (Agent state & context) — `AgentState`, `AgentContext`, `GradingResult`, `GradeDocuments`, `SourceItem` models

## Target Location
- `src/services/agents/nodes/grade_documents_node.py`
- Tests: `tests/unit/test_grade_documents_node.py`
- Notebook: `notebooks/specs/S6.4_grading.ipynb`

## Functional Requirements

### FR-1: Binary Document Grading via LLM
- **What**: For each `SourceItem` in `state["sources"]`, invoke the LLM with structured output (`GradeDocuments`) to get a binary "yes"/"no" relevance grade plus reasoning.
- **Inputs**: `state["sources"]` (list of SourceItem), `state["messages"]` (for the user query), `context.llm_provider`
- **Outputs**: Partial state dict with `grading_results` (list of `GradingResult`) and `relevant_sources` (list of `SourceItem` where grade == "yes")
- **Edge cases**:
  - Empty sources list → return empty grading_results and relevant_sources
  - LLM call fails for a single document → mark that document as not relevant (fail-safe), log warning
  - All documents graded "no" → relevant_sources is empty (routing handles this downstream)

### FR-2: Grading Prompt Construction
- **What**: Build a prompt that includes the user query and the document chunk text, asking the LLM to assess binary relevance.
- **Inputs**: User query string, document chunk_text
- **Outputs**: Formatted prompt string
- **Edge cases**: Empty chunk_text → still grade (LLM may return "no")

### FR-3: Conditional Edge — Continue or Rewrite
- **What**: A routing function that checks `state["relevant_sources"]`. If any relevant sources exist, route to "generate". If none, route to "rewrite" (so the query can be improved and re-retrieved).
- **Inputs**: `state["relevant_sources"]`, `state["retrieval_attempts"]`, `context.max_retrieval_attempts`
- **Outputs**: `"generate"` if relevant_sources is non-empty; `"rewrite"` if empty and attempts < max; `"generate"` if attempts exhausted (force generation with whatever is available)
- **Edge cases**: Missing retrieval_attempts defaults to 0

### FR-4: Parallel Grading (Sequential LLM Calls)
- **What**: Grade documents sequentially (not concurrently) to respect LLM rate limits and maintain deterministic ordering. Use `asyncio.gather` only if explicitly opted in via context.
- **Inputs**: List of sources
- **Outputs**: Ordered list of GradingResult matching source order

## Tangible Outcomes
- [ ] `grade_documents_node.py` exists with `ainvoke_grade_documents_step` async function
- [ ] `continue_after_grading` conditional edge function exists
- [ ] Grading prompt uses structured output (`GradeDocuments` model from `models.py`)
- [ ] Empty sources → returns empty results without LLM calls
- [ ] LLM failure per-document → graceful degradation (marked not relevant)
- [ ] All "no" grades → relevant_sources is empty, routing returns "rewrite"
- [ ] Exhausted retries → routing returns "generate" even with no relevant sources
- [ ] All tests pass with mocked LLM provider
- [ ] Notebook `S6.4_grading.ipynb` demonstrates the node

## Test-Driven Requirements

### Tests to Write First
1. `test_grade_documents_all_relevant`: All sources graded "yes" → all in relevant_sources
2. `test_grade_documents_mixed_relevance`: Some "yes", some "no" → only "yes" in relevant_sources
3. `test_grade_documents_none_relevant`: All "no" → relevant_sources empty
4. `test_grade_documents_empty_sources`: No sources → empty results, no LLM calls
5. `test_grade_documents_llm_failure`: LLM raises → document marked not relevant, others still graded
6. `test_continue_after_grading_has_relevant`: relevant_sources non-empty → "generate"
7. `test_continue_after_grading_no_relevant`: relevant_sources empty, attempts < max → "rewrite"
8. `test_continue_after_grading_exhausted_retries`: relevant_sources empty, attempts >= max → "generate"
9. `test_grading_prompt_includes_query_and_chunk`: Verify prompt is correctly formatted
10. `test_grading_result_structure`: Each GradingResult has document_id, is_relevant, score, reasoning

### Mocking Strategy
- Mock `context.llm_provider.get_langchain_model()` → returns mock LLM
- Mock `llm.with_structured_output(GradeDocuments)` → returns mock structured LLM
- Mock `structured_llm.ainvoke()` → returns `GradeDocuments(binary_score="yes"|"no", reasoning="...")`
- Use `SourceItem` fixtures with realistic arxiv_id, title, chunk_text
- No real LLM calls in unit tests

### Coverage
- All public functions tested
- Edge cases covered (empty, failures, exhausted retries)
- Error paths tested (LLM exceptions)
