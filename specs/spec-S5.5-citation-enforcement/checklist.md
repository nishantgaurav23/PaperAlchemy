# Checklist -- Spec S5.5: Citation Enforcement

## Phase 1: Setup & Dependencies
- [x] Verify S5.2 (RAG chain) is "done"
- [x] Create `src/services/rag/citation.py`
- [x] Create `tests/unit/test_citation.py`

## Phase 2: Tests First (TDD)
- [x] Write CitationParser tests (test_parse_citations_*)
- [x] Write CitationValidator tests (test_validate_citations_*)
- [x] Write SourceListFormatter tests (test_format_source_list_*)
- [x] Write CitationEnforcer tests (test_enforce_citations_*)
- [x] Write streaming citation tests (test_stream_with_citations_*)
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Define data models: `CitationValidation`, `CitationResult`
- [x] Implement `parse_citations(text)` -- pass parser tests
- [x] Implement `validate_citations(cited, sources)` -- pass validator tests
- [x] Implement `format_source_list(sources, cited_only)` -- pass formatter tests
- [x] Implement `enforce_citations(response)` -- pass enforcer tests
- [x] Implement `stream_with_citations(tokens, sources)` -- pass streaming tests
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Wire citation enforcement into RAGChain.aquery() (post-process response)
- [x] Wire streaming citations into RAGChain.aquery_stream()
- [x] Update ask router to use citation-enforced responses (no changes needed — router uses RAGChain which now enforces)
- [x] Run lint (ruff check)
- [x] Run full test suite (625 passed)

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook: `notebooks/specs/S5.5_citations.ipynb`
- [x] Update roadmap.md status to "done"
- [x] Append summary to `docs/spec-summaries.md`
