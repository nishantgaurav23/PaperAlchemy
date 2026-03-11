# Spec S5.1 -- Unified LLM Client (Ollama + Gemini 3 Flash)

## Overview

A provider-abstracted LLM client that supports both local (Ollama) and cloud (Google Gemini 3 Flash) inference. The service defines a common `LLMProvider` protocol so callers never couple to a specific backend. Provider switching is config-driven (`GEMINI__API_KEY` present = cloud, otherwise local).

Both providers must support:
- Single-shot text generation
- Streaming generation (async generator)
- LangChain-compatible model instance (for LangGraph agent nodes)

## Dependencies

- **S1.2** (Environment config) -- `OllamaSettings`, `GeminiSettings` in `src/config.py`

## Target Location

`src/services/llm/`

## Functional Requirements

### FR-1: LLMProvider Protocol
- **What**: Define an abstract interface that both Ollama and Gemini providers implement
- **Methods**:
  - `async generate(prompt: str, *, model: str | None, temperature: float | None, max_tokens: int | None) -> LLMResponse`
  - `async generate_stream(prompt: str, *, model: str | None, temperature: float | None, max_tokens: int | None) -> AsyncIterator[str]`
  - `async health_check() -> HealthStatus`
  - `get_langchain_model(*, model: str | None, temperature: float | None) -> BaseChatModel`
  - `async close() -> None`
- **Output schema** (`LLMResponse`): `text: str`, `model: str`, `provider: str`, `usage: UsageMetadata | None`
- **Output schema** (`UsageMetadata`): `prompt_tokens: int | None`, `completion_tokens: int | None`, `total_tokens: int | None`, `latency_ms: float | None`
- **Output schema** (`HealthStatus`): `healthy: bool`, `provider: str`, `message: str`, `version: str | None`

### FR-2: OllamaProvider
- **What**: LLMProvider implementation wrapping the Ollama HTTP API
- **Inputs**: `OllamaSettings` (host, port, default_model, timeout, temperature, top_p)
- **Behavior**:
  - `generate()`: POST to `/api/generate` with `stream=False`, extract usage metadata (token counts, timing)
  - `generate_stream()`: POST to `/api/generate` with `stream=True`, yield text chunks from NDJSON lines
  - `health_check()`: GET `/api/version`
  - `get_langchain_model()`: Return `ChatOllama` instance with configured base_url, model, temperature, timeout
  - `close()`: Close the `httpx.AsyncClient`
- **Error mapping**: `httpx.ConnectError` -> `LLMConnectionError`, `httpx.TimeoutException` -> `LLMTimeoutError`, other -> `LLMServiceError`
- **Edge cases**: Ollama not running, model not pulled, empty prompt

### FR-3: GeminiProvider
- **What**: LLMProvider implementation wrapping the Google Generative AI SDK
- **Inputs**: `GeminiSettings` (api_key, model, temperature, max_output_tokens, timeout)
- **Behavior**:
  - `generate()`: Use `google.genai` client to generate content, extract usage metadata
  - `generate_stream()`: Use `google.genai` client streaming, yield text chunks
  - `health_check()`: List models to verify API key validity
  - `get_langchain_model()`: Return `ChatGoogleGenerativeAI` instance
  - `close()`: No-op (SDK manages connections)
- **Error mapping**: Auth errors (invalid API key) -> `LLMServiceError`, timeout -> `LLMTimeoutError`, other -> `LLMServiceError`
- **Edge cases**: Missing API key (raise `ConfigurationError`), rate limiting, empty response

### FR-4: Factory Function
- **What**: `create_llm_provider(settings: Settings) -> LLMProvider`
- **Logic**:
  - If `settings.gemini.api_key` is non-empty -> `GeminiProvider`
  - Otherwise -> `OllamaProvider`
- **Edge cases**: Neither configured -> fallback to Ollama (always available locally)

### FR-5: Multi-provider Access
- **What**: `create_llm_providers(settings: Settings) -> dict[str, LLMProvider]`
- **Logic**: Return all available providers keyed by name ("ollama", "gemini")
- **Purpose**: Allow callers (e.g., agents) to explicitly choose a provider

## Tangible Outcomes
- [ ] `src/services/llm/__init__.py` exports `LLMProvider`, `LLMResponse`, `OllamaProvider`, `GeminiProvider`, `create_llm_provider`
- [ ] `src/services/llm/provider.py` defines `LLMProvider` protocol and response models
- [ ] `src/services/llm/ollama_provider.py` implements `OllamaProvider`
- [ ] `src/services/llm/gemini_provider.py` implements `GeminiProvider`
- [ ] `src/services/llm/factory.py` implements `create_llm_provider` and `create_llm_providers`
- [ ] `tests/unit/test_llm_provider.py` -- all provider tests pass
- [ ] `notebooks/specs/S5.1_llm_client.ipynb` -- interactive demo notebook
- [ ] Both providers produce identical `LLMResponse` schema
- [ ] Streaming yields `str` chunks from both providers
- [ ] `health_check()` returns `HealthStatus` for both providers
- [ ] `get_langchain_model()` returns a LangChain-compatible chat model from both providers
- [ ] Factory selects provider based on config (Gemini if API key set, else Ollama)

## Test-Driven Requirements

### Tests to Write First
1. `test_llm_response_model`: Validate LLMResponse and UsageMetadata schemas
2. `test_ollama_generate`: Mock httpx, verify POST body and response parsing
3. `test_ollama_generate_stream`: Mock streaming httpx, verify chunk yielding
4. `test_ollama_health_check_healthy`: Mock 200 response
5. `test_ollama_health_check_unreachable`: Mock ConnectError -> LLMConnectionError
6. `test_ollama_timeout`: Mock TimeoutException -> LLMTimeoutError
7. `test_ollama_get_langchain_model`: Verify ChatOllama instance returned
8. `test_gemini_generate`: Mock google.genai, verify response parsing
9. `test_gemini_generate_stream`: Mock streaming, verify chunk yielding
10. `test_gemini_health_check`: Mock model listing
11. `test_gemini_missing_api_key`: Verify ConfigurationError raised
12. `test_gemini_get_langchain_model`: Verify ChatGoogleGenerativeAI instance returned
13. `test_factory_selects_gemini`: Set API key -> GeminiProvider
14. `test_factory_selects_ollama`: No API key -> OllamaProvider
15. `test_factory_multi_provider`: Verify dict with available providers

### Mocking Strategy
- Mock `httpx.AsyncClient` for Ollama (both generate and stream)
- Mock `google.genai.Client` for Gemini
- Mock `ChatOllama` and `ChatGoogleGenerativeAI` constructors for LangChain tests
- Use `unittest.mock.AsyncMock` and `unittest.mock.patch`

### Coverage
- All public methods tested
- Error paths (connection, timeout, auth, missing config) tested
- Both providers tested symmetrically
- Factory logic tested for all branches
