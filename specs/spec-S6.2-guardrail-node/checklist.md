# Checklist -- Spec S6.2: Guardrail Node

## Phase 1: Setup & Dependencies
- [x] Verify S6.1 (Agent state & context) is "done"
- [x] Create `src/services/agents/nodes/` directory with `__init__.py`
- [x] Create target file: `guardrail_node.py`

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_guardrail_node.py`
- [x] Write failing tests for `ainvoke_guardrail_step` (FR-1)
- [x] Write failing tests for `continue_after_guardrail` (FR-2)
- [x] Write failing tests for `get_latest_query` (FR-3)
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement `get_latest_query` helper -- pass tests
- [x] Implement `GUARDRAIL_PROMPT` template (FR-4)
- [x] Implement `ainvoke_guardrail_step` node -- pass tests
- [x] Implement `continue_after_guardrail` edge -- pass tests
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Export from `src/services/agents/nodes/__init__.py`
- [x] Run lint (`ruff check src/services/agents/nodes/`)
- [x] Run full test suite (`pytest tests/`)

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook `notebooks/specs/S6.2_guardrail.ipynb`
- [x] Update roadmap.md status to "done"
- [x] Append spec summary to `docs/spec-summaries.md`
