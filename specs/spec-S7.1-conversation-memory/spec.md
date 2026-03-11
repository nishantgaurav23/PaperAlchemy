# Spec S7.1 -- Conversation Memory

## Overview
Session-based conversation history backed by Redis. Each chat session maintains a sliding window of the last 20 messages (user + assistant pairs) with a 24-hour TTL. This enables follow-up questions and contextual responses in the chatbot by providing conversation history to the RAG pipeline and agent system.

Messages are stored as JSON-serialized lists per session. The memory service provides add, retrieve, clear, and list operations — all with graceful Redis failure handling (never propagate exceptions).

## Dependencies
- **S5.4** (Response Caching) — Redis client factory, connection patterns, graceful degradation patterns

## Target Location
- `src/services/chat/memory.py` — ConversationMemory class
- `src/services/chat/__init__.py` — Public exports

## Functional Requirements

### FR-1: Session Message Storage (add_message)
- **What**: Append a message (user or assistant) to a session's conversation history
- **Inputs**: `session_id: str`, `role: str` ("user" | "assistant"), `content: str`, optional `metadata: dict | None` (e.g., sources, citations)
- **Outputs**: None (fire-and-forget, log on failure)
- **Edge cases**:
  - Redis connection failure → log warning, do not raise
  - Empty content → raise ValueError
  - Invalid role → raise ValueError (only "user" or "assistant" allowed)
  - First message for session → create new list

### FR-2: Sliding Window Enforcement
- **What**: After adding a message, trim the conversation to the last N messages (default 20)
- **Inputs**: Automatic after each `add_message`, configurable `max_messages: int`
- **Outputs**: None — older messages silently discarded
- **Edge cases**: Fewer than max_messages → no trimming needed

### FR-3: Retrieve Conversation History (get_history)
- **What**: Return all messages in a session's conversation, ordered chronologically
- **Inputs**: `session_id: str`, optional `limit: int | None` (return last N messages)
- **Outputs**: `list[ChatMessage]` — list of message objects with role, content, timestamp, metadata
- **Edge cases**:
  - Session not found → return empty list
  - Redis failure → return empty list + log warning
  - Limit > stored messages → return all available

### FR-4: Clear Session (clear_session)
- **What**: Delete all messages for a given session
- **Inputs**: `session_id: str`
- **Outputs**: `bool` — True if session existed and was cleared, False otherwise
- **Edge cases**: Redis failure → return False + log warning

### FR-5: Session TTL Management
- **What**: Each session key has a 24h TTL, refreshed on every interaction (add_message or get_history)
- **Inputs**: Automatic — TTL from settings (default 24 hours)
- **Outputs**: None
- **Edge cases**: Redis failure during TTL refresh → log warning, do not raise

### FR-6: List Active Sessions (list_sessions)
- **What**: Return all active session IDs (for admin/debug purposes)
- **Inputs**: None
- **Outputs**: `list[str]` — session IDs
- **Edge cases**: Redis failure → return empty list + log warning

### FR-7: ChatMessage Model
- **What**: Pydantic model for individual chat messages
- **Fields**: `role: str`, `content: str`, `timestamp: datetime`, `metadata: dict | None`
- **Serialization**: JSON-serializable for Redis storage

### FR-8: ConversationMemory Factory
- **What**: Create ConversationMemory from settings, reusing Redis client from cache factory
- **Inputs**: Settings, optional existing Redis client
- **Outputs**: `ConversationMemory | None` — None if Redis unavailable
- **Edge cases**: Propagates None from Redis factory

## Tangible Outcomes
- [ ] `ChatMessage` Pydantic model with role, content, timestamp, metadata fields
- [ ] `ConversationMemory` class with `add_message`, `get_history`, `clear_session`, `list_sessions` methods
- [ ] Sliding window trims to last 20 messages (configurable via `max_messages`)
- [ ] 24h TTL per session, refreshed on every interaction
- [ ] Redis key format: `chat:session:{session_id}` — messages stored as JSON list
- [ ] All Redis failures caught and logged — never propagate exceptions
- [ ] `make_conversation_memory()` factory returns `ConversationMemory | None`
- [ ] `ConversationMemoryDep` Annotated type available in `src/dependency.py`
- [ ] All tests pass with mocked Redis (no real Redis needed)

## Test-Driven Requirements

### Tests to Write First
1. `test_chat_message_model`: ChatMessage serialization/deserialization round-trip
2. `test_chat_message_invalid_role`: Raises ValueError for invalid roles
3. `test_add_message_success`: Appends message to Redis list
4. `test_add_message_empty_content`: Raises ValueError
5. `test_add_message_redis_error`: Logs warning, does not raise
6. `test_add_message_with_metadata`: Stores metadata alongside message
7. `test_sliding_window_trims`: After 25 messages, only last 20 remain
8. `test_sliding_window_under_limit`: No trimming when under max_messages
9. `test_get_history_success`: Returns messages in chronological order
10. `test_get_history_empty_session`: Returns empty list for unknown session
11. `test_get_history_with_limit`: Returns only last N messages
12. `test_get_history_redis_error`: Returns empty list on Redis failure
13. `test_get_history_refreshes_ttl`: TTL is refreshed on get_history call
14. `test_clear_session_success`: Deletes session key, returns True
15. `test_clear_session_not_found`: Returns False for non-existent session
16. `test_clear_session_redis_error`: Returns False on Redis failure
17. `test_list_sessions`: Returns all active session IDs
18. `test_list_sessions_redis_error`: Returns empty list on Redis failure
19. `test_ttl_refreshed_on_add_message`: TTL reset after adding message
20. `test_make_conversation_memory_success`: Factory returns ConversationMemory
21. `test_make_conversation_memory_no_redis`: Factory returns None when Redis unavailable

### Mocking Strategy
- Mock `redis.asyncio.Redis` for all unit tests — no real Redis connection
- Use `AsyncMock` for async Redis methods (rpush, lrange, ltrim, delete, expire, scan_iter, exists)
- Reuse Redis client factory from S5.4 (`make_redis_client`)
- Mock `Settings` with test values for factory tests

### Coverage
- All public methods tested
- All Redis failure paths tested (graceful degradation)
- Sliding window edge cases tested
- TTL refresh behavior tested
- Serialization/deserialization round-trip tested
