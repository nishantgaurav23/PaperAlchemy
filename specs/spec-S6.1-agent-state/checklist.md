# Checklist -- Spec S6.1: Agent State & Runtime Context

## Phase 1: Setup & Dependencies
- [x] Verify S5.1 (LLM client) is "done"
- [x] Create `src/services/agents/` directory with `__init__.py`
- [x] Create target files: `state.py`, `context.py`, `models.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_agent_state.py`
- [x] Create `tests/unit/test_agent_context.py`
- [x] Create `tests/unit/test_agent_models.py`
- [x] Write failing tests for AgentState (FR-1, FR-4)
- [x] Write failing tests for AgentContext (FR-2, FR-5)
- [x] Write failing tests for structured output models (FR-3)
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement `models.py` -- GuardrailScoring, GradeDocuments, GradingResult, SourceItem, RoutingDecision
- [x] Implement `state.py` -- AgentState TypedDict + create_initial_state()
- [x] Implement `context.py` -- AgentContext dataclass + create_agent_context()
- [x] Implement `__init__.py` -- re-export all public types
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Verify imports work from `src.services.agents`
- [x] Run lint (`ruff check src/services/agents/`)
- [x] Run full test suite (`pytest tests/`)

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook `notebooks/specs/S6.1_agent_state.ipynb`
- [x] Update roadmap.md status to "done"
- [x] Append spec summary to `docs/spec-summaries.md`
