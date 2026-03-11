# Check Spec Dependencies

Verify all prerequisite specs are completed before implementing a spec.

## Input
- Spec ID (e.g., "S2.1")

## Instructions

1. **Load Spec**:
   - Read `specs/spec-{ID}-*/spec.md` (if it exists)
   - Read `roadmap.md` to find the spec entry

2. **Identify Dependencies**:
   - Extract the "Depends On" column from the roadmap table
   - If spec.md exists, also check its "Dependencies" section

3. **Check Each Dependency**:
   - For each dependency spec, check its status in `roadmap.md`
   - A dependency is READY if its status is "done"
   - A dependency is BLOCKING if its status is "pending" or "spec-written"

4. **Verify Tests Pass** (for "done" dependencies):
   - Run tests for each "done" dependency to confirm they still pass
   - If tests fail, mark as BLOCKING with note "tests failing"

5. **Generate Dependency Report**:

```
## Dependency Report — Spec {ID}: {Feature}

| Dependency | Feature | Status | Result |
|-----------|---------|--------|--------|
| S{x}.{y} | {name} | {status} | READY / BLOCKING |
| S{x}.{y} | {name} | {status} | READY / BLOCKING |

### Verdict: {READY TO IMPLEMENT / BLOCKED}

{If BLOCKED: list which specs need to be completed first, in recommended order}
```

## Output
- Dependency table with READY/BLOCKING status for each prerequisite
- Clear verdict: can this spec be implemented now, or what's blocking it
