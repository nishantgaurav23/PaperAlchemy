# Checklist -- Spec S5.1: Unified LLM Client (Ollama + Gemini 3 Flash)

## Phase 1: Setup & Dependencies
- [x] Verify S1.2 (Environment config) is "done"
- [x] Create `src/services/llm/` directory
- [x] Verify `OllamaSettings` and `GeminiSettings` exist in `src/config.py`
- [x] Verify LLM exception classes exist in `src/exceptions.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_llm_provider.py`
- [x] Write failing tests for LLMResponse / UsageMetadata / HealthStatus models
- [x] Write failing tests for OllamaProvider (generate, stream, health_check, langchain)
- [x] Write failing tests for GeminiProvider (generate, stream, health_check, langchain)
- [x] Write failing tests for factory (create_llm_provider, create_llm_providers)
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement `src/services/llm/provider.py` -- LLMProvider protocol + response models
- [x] Implement `src/services/llm/ollama_provider.py` -- OllamaProvider
- [x] Implement `src/services/llm/gemini_provider.py` -- GeminiProvider
- [x] Implement `src/services/llm/factory.py` -- create_llm_provider, create_llm_providers
- [x] Implement `src/services/llm/__init__.py` -- public exports
- [x] Run tests -- expect pass (Green) -- 27/27 passing
- [x] Refactor if needed

## Phase 4: Integration
- [x] Wire into DI container (`src/dependency.py`) -- LLMProviderDep added
- [x] Run lint (`ruff check src/services/llm/`) -- 0 errors
- [x] Run full test suite -- 441 passed, 9 skipped

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook `notebooks/specs/S5.1_llm_client.ipynb`
- [x] Update roadmap.md status to "done"
