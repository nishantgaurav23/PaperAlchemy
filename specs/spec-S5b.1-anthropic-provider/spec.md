# S5b.1 — Anthropic Claude LLM Provider

## Goal
Add `AnthropicProvider` implementing the `LLMProvider` protocol, supporting Claude Opus/Sonnet/Haiku models with streaming + non-streaming generation, tool use support for agentic code gen, and LangChain integration.

## Dependencies
- **S5.1** (Unified LLM client) — `LLMProvider` protocol, response models, factory — **done**
- **S9b.2** (Platform dependencies) — `anthropic` package installed — **done**

## Outcomes

### 1. AnthropicSettings (config)
- Pydantic `BaseSettings` with `ANTHROPIC__` env prefix
- Fields: `api_key` (str), `model` (str, default `claude-sonnet-4-20250514`), `temperature` (float, default 0.7), `max_tokens` (int, default 4096), `timeout` (int, default 60)
- Registered in `Settings` root model

### 2. AnthropicProvider (src/services/llm/anthropic_provider.py)
Implements the full `LLMProvider` protocol:
- **`generate()`** — Non-streaming generation via `anthropic.AsyncAnthropic.messages.create()`
- **`generate_stream()`** — Streaming via `messages.stream()` context manager, yields text delta chunks
- **`health_check()`** — Validates API key by listing models; returns `HealthStatus`
- **`get_langchain_model()`** — Returns `ChatAnthropic` from `langchain-anthropic`
- **`close()`** — Calls `await self._client.close()`
- **Error mapping**: `anthropic.AuthenticationError` → `ConfigurationError`, `anthropic.APITimeoutError` → `LLMTimeoutError`, `anthropic.APIConnectionError` → `LLMConnectionError`, other `anthropic.APIError` → `LLMServiceError`
- **Usage extraction**: Maps `usage.input_tokens`, `usage.output_tokens` → `UsageMetadata`

### 3. Factory Integration
- `create_llm_provider()` updated: if `ANTHROPIC__API_KEY` is set, try Anthropic first (before Gemini)
- `create_llm_providers()` updated: include `"anthropic"` key when API key configured
- Priority order: Anthropic → Gemini → Ollama

### 4. __init__.py exports
- Add `AnthropicProvider` to `src/services/llm/__init__.py` exports

## TDD Plan

### Red Phase — Tests First
1. `TestAnthropicProvider.test_generate` — mock `messages.create`, verify `LLMResponse` fields
2. `TestAnthropicProvider.test_generate_with_overrides` — custom model/temp/max_tokens
3. `TestAnthropicProvider.test_generate_stream` — mock streaming, collect text chunks
4. `TestAnthropicProvider.test_health_check_healthy` — mock models list
5. `TestAnthropicProvider.test_health_check_auth_failure` — mock auth error
6. `TestAnthropicProvider.test_missing_api_key_raises` — empty key → `ConfigurationError`
7. `TestAnthropicProvider.test_get_langchain_model` — verify `ChatAnthropic` instantiation
8. `TestAnthropicProvider.test_generate_timeout` — `APITimeoutError` → `LLMTimeoutError`
9. `TestAnthropicProvider.test_generate_auth_error` — `AuthenticationError` → `ConfigurationError`
10. `TestAnthropicProvider.test_generate_connection_error` — `APIConnectionError` → `LLMConnectionError`
11. `TestAnthropicProvider.test_close` — verify client close called
12. `TestFactory.test_selects_anthropic_when_api_key_set` — Anthropic takes priority
13. `TestFactory.test_multi_provider_includes_anthropic` — present in providers dict

### Green Phase
Implement `AnthropicProvider`, update config, update factory.

### Refactor Phase
Ensure consistent patterns with `GeminiProvider` and `OllamaProvider`.

## Files
| File | Action |
|------|--------|
| `src/config.py` | Add `AnthropicSettings`, register in `Settings` |
| `src/services/llm/anthropic_provider.py` | **New** — full provider implementation |
| `src/services/llm/factory.py` | Update priority + multi-provider |
| `src/services/llm/__init__.py` | Export `AnthropicProvider` |
| `tests/unit/test_llm_provider.py` | Add `TestAnthropicProvider` + updated factory tests |
| `.env.example` | Add `ANTHROPIC__API_KEY`, `ANTHROPIC__MODEL` |

## Non-Goals
- Multi-provider routing (that's S5b.2)
- Tool use / function calling (future enhancement)
- Cost tracking (that's S5b.2)
