# S5b.1 — Anthropic Claude LLM Provider — Checklist

## Phase 1: Config
- [x] Add `AnthropicSettings` to `src/config.py`
- [x] Register `anthropic` field in `Settings`
- [x] Add env vars to `.env.example` (already present)

## Phase 2: Tests (Red)
- [x] `test_generate` — non-streaming generation
- [x] `test_generate_with_overrides` — custom model/temp/tokens
- [x] `test_generate_stream` — streaming text chunks
- [x] `test_health_check_healthy` — successful health check
- [x] `test_health_check_auth_failure` — auth error returns unhealthy
- [x] `test_missing_api_key_raises` — empty key → ConfigurationError
- [x] `test_get_langchain_model` — ChatAnthropic instantiation
- [x] `test_generate_timeout` — APITimeoutError → LLMTimeoutError
- [x] `test_generate_auth_error` — AuthenticationError → ConfigurationError
- [x] `test_generate_connection_error` — APIConnectionError → LLMConnectionError
- [x] `test_close` — client close called
- [x] Factory: `test_selects_anthropic_when_api_key_set`
- [x] Factory: `test_anthropic_takes_priority_over_gemini`
- [x] Factory: `test_multi_provider_includes_anthropic`

## Phase 3: Implementation (Green)
- [x] Create `src/services/llm/anthropic_provider.py`
- [x] Update `src/services/llm/factory.py`
- [x] Update `src/services/llm/__init__.py`
- [x] Install `langchain-anthropic` dependency

## Phase 4: Refactor & Verify
- [x] All tests pass (41/41)
- [x] Lint clean (`ruff check`)
- [x] Consistent patterns with Gemini/Ollama providers
- [x] Roadmap status → done
