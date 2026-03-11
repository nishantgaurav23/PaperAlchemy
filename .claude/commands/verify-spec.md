# Verify Spec

Post-implementation audit to confirm a spec is truly complete.

## Input
- Spec ID (e.g., "S1.1")

## Instructions

1. **Load Spec**:
   - Read `specs/spec-{ID}-*/spec.md`
   - Read `specs/spec-{ID}-*/checklist.md`

2. **Code Exists Check**:
   - Verify all files listed in "Target Location" exist
   - Verify they contain meaningful implementation (not stubs)

3. **Test Check**:
   - Run: `pytest tests/ -v -k "{relevant_test_pattern}"`
   - Confirm all tests pass
   - Check coverage is adequate (all public functions tested)

4. **Lint Check**:
   - Run: `ruff check src/ tests/`
   - Run: `ruff format --check src/ tests/`
   - Report any violations

5. **Tangible Outcomes Audit**:
   - For EACH outcome in spec.md, verify it is met:
     - Can it be demonstrated? (e.g., API endpoint responds, module imports, etc.)
     - Is there a test that proves it?

6. **Security Check**:
   - No hardcoded secrets (grep for API keys, passwords, tokens)
   - No .env files committed
   - Config via environment variables only

7. **Generate Verification Report**:

```
## Verification Report — Spec {ID}: {Feature}

### Code: {PASS/FAIL}
- Files: {list of files created/modified}

### Tests: {PASS/FAIL}
- Total: {N} tests
- Passed: {N}
- Failed: {N}
- Coverage: {estimate}

### Lint: {PASS/FAIL}
- Ruff check: {PASS/FAIL}
- Ruff format: {PASS/FAIL}

### Tangible Outcomes: {PASS/FAIL}
- [ ] Outcome 1: {PASS/FAIL} — {evidence}
- [ ] Outcome 2: {PASS/FAIL} — {evidence}

### Security: {PASS/FAIL}
- No hardcoded secrets: {PASS/FAIL}

### Overall: {PASS/FAIL}
```

8. If FAIL: List specific items to fix before spec can be marked "done".

9. **Update Spec Summaries** (only if Overall = PASS):
   - Append a new section to `docs/spec-summaries.md` for this spec
   - Follow the template format already in that file
   - Include these sections:
     - **What Was Done**: High-level deliverable description
     - **How It Was Done**: Implementation approach, patterns, libraries used
     - **Why It Matters**: How this enables downstream specs and improves the system
     - **Core Features**: Bulleted list of concrete capabilities added
     - **Key Files**: Files created or modified
     - **Dependencies Unlocked**: Which specs can now proceed (check roadmap.md "Depends On" column)

## Output
- Verification report (printed to console)
- If all PASS: confirm spec is complete, spec summary appended to `docs/spec-summaries.md`
- If any FAIL: list remediation steps (do NOT update spec summaries)
