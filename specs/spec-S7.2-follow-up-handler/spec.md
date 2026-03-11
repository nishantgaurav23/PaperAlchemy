# Spec S7.2 -- Follow-up Handler

## Overview
Context-aware follow-up Q&A handler that resolves coreference in conversational queries.
When a user asks a follow-up like "What about its limitations?" or "How does that compare to BERT?",
the handler uses conversation history + LLM to rewrite the query into a self-contained,
fully-resolved question, then re-retrieves from the knowledge base every time.

## Dependencies
- **S7.1** (Conversation Memory) — Redis-backed session history
- **S5.2** (RAG Chain) — Retrieval + citation-enforcing generation

## Target Location
- `src/services/chat/follow_up.py`

## Functional Requirements

### FR-1: Query Rewriting with Coreference Resolution
- **What**: Given a follow-up query and conversation history, use the LLM to rewrite the query
  into a self-contained question that resolves all pronouns and references.
- **Inputs**:
  - `query: str` — the raw follow-up question (e.g., "What about its limitations?")
  - `history: list[ChatMessage]` — recent conversation (from ConversationMemory)
- **Outputs**: `str` — rewritten, self-contained query (e.g., "What are the limitations of the Transformer architecture described in Attention Is All You Need?")
- **Edge cases**:
  - Empty history → return original query unchanged
  - Query is already self-contained (no coreference) → LLM returns it unchanged
  - LLM failure → fall back to original query (graceful degradation)
  - Very long history → only use last N messages (configurable, default 10)

### FR-2: Follow-up Query Detection
- **What**: Determine whether a query is a follow-up (needs context) or a standalone question.
- **Inputs**: `query: str`, `history: list[ChatMessage]`
- **Outputs**: `bool` — True if follow-up detected
- **Heuristics** (fast, no LLM call):
  - Contains pronouns referring to prior context: "it", "its", "they", "them", "this", "that", "those", "these"
  - Starts with continuation words: "What about", "How about", "And", "Also", "But", "Can you"
  - Very short query (< 5 words) with history present
  - Empty history → always False (no context to follow up on)

### FR-3: Follow-up Orchestration
- **What**: Full follow-up pipeline: detect → rewrite (if needed) → query RAG → return response.
- **Inputs**:
  - `session_id: str`
  - `query: str`
  - `top_k: int | None`
  - `categories: list[str] | None`
  - `temperature: float | None`
  - `stream: bool` (default False)
- **Outputs**: `FollowUpResult` (contains rewritten_query, is_follow_up flag, RAGResponse or stream)
- **Behavior**:
  1. Retrieve history from ConversationMemory
  2. Detect if follow-up
  3. If follow-up, rewrite query using LLM + history
  4. Pass (rewritten) query to RAGChain.aquery() or aquery_stream()
  5. Store user message + assistant response in ConversationMemory
  6. Return FollowUpResult with both original and rewritten queries
- **Edge cases**:
  - ConversationMemory unavailable (None) → treat as standalone query, skip history storage
  - RAGChain error → propagate (don't swallow RAG errors)
  - Streaming mode → return async iterator, store response after stream completes

## Tangible Outcomes
- [ ] `FollowUpHandler` class with `handle()` and `handle_stream()` methods
- [ ] `is_follow_up()` heuristic function detects coreference queries
- [ ] `rewrite_query()` uses LLM to resolve coreferences into self-contained query
- [ ] Follow-up queries re-retrieve from knowledge base (never reuse old results)
- [ ] Conversation history stored after each exchange (user + assistant messages)
- [ ] Graceful degradation: works without ConversationMemory (treats all as standalone)
- [ ] Graceful degradation: LLM rewrite failure falls back to original query
- [ ] `FollowUpResult` model contains: original_query, rewritten_query, is_follow_up, response/stream

## Test-Driven Requirements

### Tests to Write First
1. `test_is_follow_up_with_pronouns`: "What about its limitations?" → True
2. `test_is_follow_up_with_continuation`: "How about compared to BERT?" → True
3. `test_is_follow_up_short_query_with_history`: "Why?" → True
4. `test_not_follow_up_standalone`: "Explain transformer architecture" → False
5. `test_not_follow_up_no_history`: Any query with empty history → False
6. `test_rewrite_resolves_coreference`: "its limitations" → full question with paper name
7. `test_rewrite_standalone_unchanged`: Self-contained query → returned unchanged
8. `test_rewrite_empty_history_returns_original`: No history → original query
9. `test_rewrite_llm_failure_returns_original`: LLM error → original query
10. `test_rewrite_trims_history`: Only last N messages used
11. `test_handle_standalone_query`: Full pipeline with standalone query
12. `test_handle_follow_up_query`: Full pipeline with follow-up → rewrite → RAG
13. `test_handle_stores_messages`: User + assistant messages stored in memory
14. `test_handle_no_memory`: Works without ConversationMemory
15. `test_handle_stream`: Streaming mode returns async iterator
16. `test_follow_up_result_model`: FollowUpResult fields correct

### Mocking Strategy
- Mock `LLMProvider.generate()` for query rewriting
- Mock `RAGChain.aquery()` and `aquery_stream()` for RAG responses
- Mock `ConversationMemory.get_history()` and `add_message()` for session ops
- Never call real external services

### Coverage
- All public functions tested
- Edge cases: empty history, LLM failure, no memory, streaming
- Error paths: LLM error on rewrite, RAG error propagation
