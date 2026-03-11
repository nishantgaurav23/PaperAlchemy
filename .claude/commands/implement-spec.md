# Implement Spec

Implement a spec using strict TDD methodology: Red → Green → Refactor.

## Input
- Spec ID (e.g., "S1.1")

## Instructions

1. **Load Context**:
   - Read `specs/spec-{ID}-*/spec.md` for requirements
   - Read `specs/spec-{ID}-*/checklist.md` for progress tracking
   - Read `roadmap.md` to confirm status is "spec-written"

2. **Check Dependencies**:
   - Verify all specs listed in "Dependencies" have status "done"
   - If any dependency is not done, STOP and report which ones are blocking

3. **Phase 1 - Setup**:
   - Create target directories/files from spec's "Target Location"
   - Update checklist: check off Phase 1 items

4. **Phase 2 - Write Tests FIRST** (Red):
   - Create test file(s) based on "Tests to Write First" section
   - Write comprehensive failing tests for EACH functional requirement
   - Include edge cases, error paths, and happy paths
   - Apply mocking strategy from spec
   - Run tests — they MUST fail (this confirms tests are meaningful)
   - Update checklist: check off Phase 2 items

5. **Phase 3 - Implement** (Green):
   - Write MINIMAL code to make each test pass, one FR at a time
   - After each FR: run tests, confirm newly passing
   - Once all tests pass: review for refactoring opportunities
   - Refactor only if it improves clarity without changing behavior
   - Update checklist: check off Phase 3 items

6. **Phase 4 - Integration**:
   - Wire the implementation into the app (register routers, add to DI, etc.)
   - Run linting: `ruff check src/ tests/` and `ruff format --check src/ tests/`
   - Run full test suite: `pytest tests/ -v`
   - Fix any issues
   - Update checklist: check off Phase 4 items

7. **Phase 5 - Verification**:
   - Verify each "Tangible Outcome" from spec.md
   - Ensure no hardcoded secrets
   - Update checklist: check off Phase 5 items
   - Update `roadmap.md`: Change status to "done"

## Rules
- **Tests BEFORE code** — never write implementation first
- **One FR at a time** — implement incrementally
- **Minimal code** — only what's needed to pass tests
- **Update checklist progressively** — not at the end
- **Mock all externals** — tests never hit real APIs
- **Async by default** — use async def, await

## Output
- Implementation code at target location(s)
- Passing test suite
- Updated checklist.md (all items checked)
- Updated roadmap.md (status: "done")
