# Spec S6.2 -- Guardrail Node

## Overview
The guardrail node is the first node in the agentic RAG LangGraph workflow. It receives the user's query, asks the LLM to score it on a 0-100 domain relevance scale using structured output (GuardrailScoring), and stores the result in agent state. A companion conditional edge function reads that score to decide whether the workflow continues to retrieval or stops at out_of_scope.

Without domain validation, every query ("What is a dog?", "Hello", "2+2?") would trigger a full retrieval + grading + generation cycle, wasting compute on guaranteed-wrong results. The guardrail is the cheapest possible check: a single fast LLM call with temperature=0.0 (deterministic) that classifies the query before anything expensive happens.

## Dependencies
- S6.1 (Agent state & context) — AgentState, AgentContext, GuardrailScoring model

## Target Location
- `src/services/agents/nodes/guardrail_node.py`

## Functional Requirements

### FR-1: Guardrail Scoring Node (`ainvoke_guardrail_step`)
- **What**: Async LangGraph node that scores a user query for domain relevance (academic/scientific research) using structured LLM output.
- **Inputs**: `state: AgentState` (reads `messages` for latest query), `context: AgentContext` (reads `llm_provider`, `model_name`, `guardrail_threshold`)
- **Outputs**: Partial state dict `{"guardrail_result": GuardrailScoring(score=int, reason=str)}`
- **Behavior**:
  1. Extract the latest user query from `state["messages"]` (last HumanMessage content)
  2. Format a guardrail prompt asking the LLM to score domain relevance 0-100
  3. Call `context.llm_provider.get_langchain_model(temperature=0.0)` for deterministic scoring
  4. Use `.with_structured_output(GuardrailScoring)` for validated JSON output
  5. Return `{"guardrail_result": response}`
- **Edge cases**:
  - LLM call fails → fallback to `GuardrailScoring(score=50, reason="LLM validation failed...")` (conservative default, above threshold=40, so queries proceed)
  - Empty/missing messages → raise or return low score
  - No HumanMessage found → handle gracefully

### FR-2: Conditional Edge Function (`continue_after_guardrail`)
- **What**: Sync function that reads guardrail score from state and returns a routing string for LangGraph.
- **Inputs**: `state: AgentState` (reads `guardrail_result`), `context: AgentContext` (reads `guardrail_threshold`)
- **Outputs**: `Literal["continue", "out_of_scope"]`
- **Behavior**:
  - If `guardrail_result.score >= context.guardrail_threshold` → return `"continue"`
  - If `guardrail_result.score < context.guardrail_threshold` → return `"out_of_scope"`
  - If no `guardrail_result` in state → default to `"continue"` (don't silently drop valid queries)
- **Edge cases**:
  - Missing guardrail_result → log warning, return "continue"

### FR-3: Query Extraction Helper (`get_latest_query`)
- **What**: Extract the content of the last HumanMessage from a message list.
- **Inputs**: `messages: list[AnyMessage]`
- **Outputs**: `str` — the query text
- **Edge cases**:
  - No HumanMessage found → raise `ValueError`
  - Empty message list → raise `ValueError`

### FR-4: Guardrail Prompt
- **What**: System prompt template for domain relevance scoring.
- **Content**: Instructs the LLM to rate a query 0-100 for relevance to academic/scientific research (machine learning, AI, NLP, computer science, arxiv papers). Queries about specific papers, algorithms, methods, or research findings should score high. Generic, off-topic, or harmful queries should score low.

## Tangible Outcomes
- [ ] `src/services/agents/nodes/guardrail_node.py` exists with `ainvoke_guardrail_step` and `continue_after_guardrail`
- [ ] `src/services/agents/nodes/__init__.py` exports guardrail functions
- [ ] Structured LLM output via `GuardrailScoring` (score 0-100 + reason)
- [ ] Temperature=0.0 for deterministic scoring
- [ ] Fallback score=50 on LLM failure (graceful degradation)
- [ ] Conditional edge returns "continue" or "out_of_scope"
- [ ] Default threshold=40 (configurable via AgentContext)
- [ ] All tests pass with mocked LLM provider

## Test-Driven Requirements

### Tests to Write First
1. `test_ainvoke_guardrail_high_score`: On-topic query → score >= 40 → routing "continue"
2. `test_ainvoke_guardrail_low_score`: Off-topic query → score < 40 → routing "out_of_scope"
3. `test_ainvoke_guardrail_llm_failure_fallback`: LLM raises → fallback score=50
4. `test_continue_after_guardrail_above_threshold`: score >= threshold → "continue"
5. `test_continue_after_guardrail_below_threshold`: score < threshold → "out_of_scope"
6. `test_continue_after_guardrail_exact_threshold`: score == threshold → "continue"
7. `test_continue_after_guardrail_no_result`: Missing guardrail_result → "continue"
8. `test_get_latest_query_extracts_human_message`: Correctly extracts last HumanMessage
9. `test_get_latest_query_empty_messages_raises`: Empty list → ValueError
10. `test_get_latest_query_no_human_message_raises`: No HumanMessage → ValueError
11. `test_guardrail_uses_structured_output`: Verifies `.with_structured_output(GuardrailScoring)` is called
12. `test_guardrail_uses_zero_temperature`: Verifies temperature=0.0

### Mocking Strategy
- Mock `context.llm_provider.get_langchain_model()` to return a mock LLM
- Mock the structured LLM's `.ainvoke()` to return `GuardrailScoring` instances
- No real LLM calls in tests — all mocked
- Use `unittest.mock.AsyncMock` for async methods

### Coverage
- All public functions tested
- Edge cases covered (LLM failure, missing state, boundary scores)
- Error paths tested (fallback behavior)
