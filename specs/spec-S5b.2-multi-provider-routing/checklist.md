# Checklist -- Spec S5b.2: Multi-Provider LLM Routing

## Phase 1: Setup & Dependencies
- [x] Verify S5b.1 (Anthropic provider) is "done"
- [x] Verify S5.1 (Ollama) and S5.2 (Gemini) are "done"
- [x] Create `src/services/llm/router.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_llm_router.py`
- [x] Write tests for TaskType enum
- [x] Write tests for routing logic (task -> provider mapping)
- [x] Write tests for generate() delegation
- [x] Write tests for generate_stream() delegation
- [x] Write tests for fallback chain (success, exhausted, disabled)
- [x] Write tests for cost tracking (increment, reset, failed)
- [x] Write tests for health_check aggregation
- [x] Write tests for get_langchain_model routing
- [x] Write tests for close() all providers
- [x] Write tests for LLMRoutingSettings
- [x] Write tests for unknown provider / edge cases
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement TaskType enum
- [x] Implement LLMRoutingSettings in config.py
- [x] Implement ProviderUsageStats dataclass
- [x] Implement LLMRouter class with route()
- [x] Implement generate() with task-based routing
- [x] Implement generate_stream() with task-based routing
- [x] Implement generate_with_fallback()
- [x] Implement cost tracking (get_usage_stats, reset_usage_stats)
- [x] Implement health_check() aggregation
- [x] Implement get_langchain_model() routing
- [x] Implement close() for all providers
- [x] Run tests -- expect pass (Green) -- 34/34 pass
- [x] Refactor if needed

## Phase 4: Integration
- [x] Add create_llm_router() factory function to factory.py
- [x] Add LLMRouterDep to dependency.py
- [x] Add LLMRoutingSettings to Settings class in config.py
- [x] Run lint (ruff check) -- all checks passed
- [x] Run full test suite -- 34/34 router tests pass, 1140 total pass

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Update roadmap.md status to "done"
