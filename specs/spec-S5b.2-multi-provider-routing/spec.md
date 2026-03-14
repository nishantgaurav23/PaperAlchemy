# Spec S5b.2 -- Multi-Provider LLM Routing

## Overview
Route different task types to their optimal LLM provider: research Q&A to Gemini (optimized for knowledge), code generation to Claude (best at code), local dev/lightweight tasks to Ollama (no API costs). Provides a configurable per-task routing table, a fallback chain (primary fails -> try secondary), and per-provider cost tracking.

The router wraps multiple `LLMProvider` instances and exposes the same `LLMProvider` protocol, so callers don't need to know which provider handles their request — they just specify the task type.

## Dependencies
- **S5b.1** (Anthropic Claude LLM provider) — `done`
- Implicitly: S5.1 (Ollama), S5.2 (Gemini) — both `done`

## Target Location
`src/services/llm/router.py`

## Functional Requirements

### FR-1: Task Type Enum
- **What**: Define a `TaskType` enum for routing decisions
- **Values**: `RESEARCH_QA`, `CODE_GENERATION`, `SUMMARIZATION`, `GRADING`, `QUERY_REWRITE`, `GENERAL`
- **Each task maps to**: a primary provider name and optional fallback chain

### FR-2: Routing Configuration (Settings)
- **What**: Add `LLMRoutingSettings` to `src/config.py` with env prefix `LLM_ROUTING__`
- **Fields**:
  - `research_qa_provider: str = "gemini"` — provider name for research Q&A
  - `code_generation_provider: str = "anthropic"` — provider name for code gen
  - `summarization_provider: str = "gemini"` — provider name for summarization
  - `grading_provider: str = "gemini"` — provider name for grading
  - `query_rewrite_provider: str = "gemini"` — provider name for query rewrite
  - `general_provider: str = "gemini"` — default fallback provider
  - `fallback_enabled: bool = True` — whether to try fallback on failure
  - `fallback_order: str = "gemini,anthropic,ollama"` — comma-separated fallback chain
  - `cost_tracking_enabled: bool = True` — enable per-provider token/cost tracking
- **Edge cases**: Unknown provider name in config -> log warning and use "ollama" as fallback

### FR-3: LLMRouter Class
- **What**: `LLMRouter` class that implements the `LLMProvider` protocol
- **Constructor**: Takes `providers: dict[str, LLMProvider]`, `routing_settings: LLMRoutingSettings`
- **Methods**:
  - `generate(prompt, *, task_type=TaskType.GENERAL, model=None, temperature=None, max_tokens=None) -> LLMResponse`
  - `generate_stream(prompt, *, task_type=TaskType.GENERAL, ...) -> AsyncIterator[str]`
  - `route(task_type: TaskType) -> LLMProvider` — resolve the primary provider for a task
  - `generate_with_fallback(prompt, *, task_type, ...) -> LLMResponse` — try primary, then fallback chain
  - `health_check() -> HealthStatus` — aggregate health across all providers
  - `get_langchain_model(*, task_type=TaskType.GENERAL, model=None, temperature=None)` — return the langchain model for the routed provider
  - `close()` — close all providers
- **Inputs**: Standard LLMProvider params + `task_type: TaskType`
- **Outputs**: `LLMResponse` with `provider` field indicating which provider was used
- **Edge cases**:
  - Primary provider not available -> try fallback chain
  - All providers fail -> raise `LLMServiceError` with details of all failures
  - Unknown task type -> use `GENERAL` mapping

### FR-4: Fallback Chain
- **What**: When the primary provider for a task fails, automatically try the next provider in the fallback chain
- **Behavior**:
  - Parse `fallback_order` from settings into ordered list of provider names
  - Skip the failed primary provider
  - Try each fallback in order until one succeeds
  - Log each fallback attempt (warning level)
  - If all fail, raise `LLMServiceError` with concatenated error messages
- **Edge cases**: Fallback disabled (`fallback_enabled=False`) -> raise immediately on primary failure

### FR-5: Cost Tracking
- **What**: Track token usage per provider for cost estimation
- **Data model**: `ProviderUsageStats` dataclass with fields:
  - `provider_name: str`
  - `total_requests: int`
  - `total_prompt_tokens: int`
  - `total_completion_tokens: int`
  - `total_tokens: int`
  - `failed_requests: int`
- **Methods on LLMRouter**:
  - `get_usage_stats() -> dict[str, ProviderUsageStats]` — return stats per provider
  - `reset_usage_stats()` — clear all counters
- **Thread safety**: Use `asyncio.Lock` or simple atomic counters
- **Edge cases**: Usage metadata not returned by provider -> skip counting for that call

### FR-6: Factory Integration
- **What**: Add `create_llm_router()` factory function
- **Location**: `src/services/llm/factory.py`
- **Behavior**: Use `create_llm_providers()` to get all available providers, then create `LLMRouter` with routing settings
- **Wire into DI**: Add `LLMRouterDep` in `dependency.py`

## Tangible Outcomes
- [ ] `src/services/llm/router.py` exists with `LLMRouter` class and `TaskType` enum
- [ ] `LLMRouter` implements the `LLMProvider` protocol (generate, generate_stream, health_check, get_langchain_model, close)
- [ ] Task-based routing works: different `TaskType` values resolve to different providers
- [ ] Fallback chain: if primary provider fails, secondary is tried automatically
- [ ] Cost tracking: `get_usage_stats()` returns per-provider token counts
- [ ] `LLMRoutingSettings` added to config with env prefix `LLM_ROUTING__`
- [ ] Factory function `create_llm_router()` in `factory.py`
- [ ] `LLMRouterDep` in `dependency.py` for DI injection
- [ ] All tests pass with mocked providers (no real API calls)
- [ ] Lint clean (ruff check)

## Test-Driven Requirements

### Tests to Write First
1. `test_task_type_enum`: All task types defined and accessible
2. `test_route_research_qa_to_gemini`: RESEARCH_QA routes to gemini provider
3. `test_route_code_gen_to_anthropic`: CODE_GENERATION routes to anthropic provider
4. `test_route_general_to_default`: GENERAL routes to configured default provider
5. `test_generate_delegates_to_correct_provider`: generate() calls the right provider based on task_type
6. `test_generate_stream_delegates_to_correct_provider`: generate_stream() routes correctly
7. `test_fallback_on_primary_failure`: When primary fails, fallback provider is used
8. `test_fallback_chain_exhausted`: All providers fail -> LLMServiceError raised
9. `test_fallback_disabled`: fallback_enabled=False -> immediate failure on primary error
10. `test_cost_tracking_increments`: After generate(), usage stats for that provider increment
11. `test_cost_tracking_reset`: reset_usage_stats() clears all counters
12. `test_cost_tracking_failed_requests`: Failed requests increment failed_requests counter
13. `test_unknown_provider_in_config`: Unknown provider name falls back gracefully
14. `test_health_check_aggregates`: health_check() reports all providers' status
15. `test_get_langchain_model_routes`: get_langchain_model() returns model from correct provider
16. `test_close_closes_all_providers`: close() calls close() on every provider
17. `test_routing_settings_from_env`: LLMRoutingSettings reads from LLM_ROUTING__ env vars

### Mocking Strategy
- Create mock `LLMProvider` instances using `AsyncMock` for each provider (gemini, anthropic, ollama)
- Mock `generate()` to return `LLMResponse` with correct `provider` field
- Mock `generate_stream()` to yield test chunks
- Mock `health_check()` to return `HealthStatus`
- Use `monkeypatch.setenv` for routing settings tests

### Coverage
- All public methods tested
- All task type routing paths tested
- Fallback chain: success on first fallback, success on second fallback, all fail
- Cost tracking: increment, reset, missing usage metadata
- Edge cases: empty providers dict, unknown task type, unknown provider name
