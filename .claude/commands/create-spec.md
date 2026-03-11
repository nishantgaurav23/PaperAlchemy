# Create Spec

Create a new spec folder with spec.md and checklist.md for a given spec ID.

## Input
- Spec ID + slug (e.g., "S1.1 dependency-declaration")

## Instructions

1. Read `roadmap.md` to find the spec entry matching the given ID.
2. Extract: Feature name, Dependencies, Location, Notes from the roadmap table.
3. Create the spec folder: `specs/spec-{ID}-{slug}/`
4. Create `spec.md` using this template:

```markdown
# Spec {ID} -- {Feature Name}

## Overview
{From roadmap feature description and notes}

## Dependencies
{List specs this depends on, from roadmap "Depends On" column}

## Target Location
{From roadmap "Location" column}

## Functional Requirements

### FR-1: {Name}
- **What**: Behavior description
- **Inputs**: Parameters, types
- **Outputs**: Return type, side effects
- **Edge cases**: Invalid input, timeouts, empty data

### FR-2: {Name}
...

## Tangible Outcomes
- [ ] Outcome 1 (testable, specific)
- [ ] Outcome 2 (testable, specific)

## Test-Driven Requirements

### Tests to Write First
1. test_{name}: Description of what it validates
2. test_{name}: Description of what it validates

### Mocking Strategy
- Mock all external services (arXiv API, Jina, Ollama, Gemini, OpenSearch, Redis)
- Use fixtures for database sessions
- Use testcontainers for integration tests if needed

### Coverage
- All public functions tested
- Edge cases covered
- Error paths tested
```

5. Create `checklist.md` using this template:

```markdown
# Checklist -- Spec {ID}: {Feature}

## Phase 1: Setup & Dependencies
- [ ] Verify all dependency specs are "done"
- [ ] Create target files/directories

## Phase 2: Tests First (TDD)
- [ ] Create test file(s)
- [ ] Write failing tests for each FR
- [ ] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [ ] Implement FR-1 -- pass tests
- [ ] Implement FR-2 -- pass tests (if applicable)
- [ ] Run tests -- expect pass (Green)
- [ ] Refactor if needed

## Phase 4: Integration
- [ ] Wire into app (register router, add to DI, etc.)
- [ ] Run lint (ruff check + mypy)
- [ ] Run full test suite

## Phase 5: Verification
- [ ] All tangible outcomes checked
- [ ] No hardcoded secrets
- [ ] Update roadmap.md status to "done"
```

6. Update `roadmap.md`: Change the spec's Status from `pending` to `spec-written`.

## Output
- Created: `specs/spec-{ID}-{slug}/spec.md`
- Created: `specs/spec-{ID}-{slug}/checklist.md`
- Updated: `roadmap.md` status column
