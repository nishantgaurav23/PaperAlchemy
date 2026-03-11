# Checklist -- Spec S9.6: Paper Detail View

## Phase 1: Setup & Dependencies
- [x] Verify S9.2 (Layout & navigation) is "done"
- [x] Create target directories: `frontend/src/app/papers/[id]/`, `frontend/src/components/paper/`

## Phase 2: Tests First (TDD)
- [x] Create test fixtures (mock paper data)
- [x] Write tests for PaperHeader component (11 tests)
- [x] Write tests for PaperSections component (7 tests)
- [x] Write tests for PaperAnalysis component (6 tests)
- [x] Write tests for RelatedPapers component (7 tests)
- [x] Write tests for papers API client (7 tests)
- [x] Write tests for paper detail page (9 tests: loading, error, not-found, success, retry, related, sections, analysis)
- [x] Run tests -- expect failures (Red) -- confirmed 6 test files failing

## Phase 3: Implementation
- [x] Extend `types/paper.ts` with detail types (PaperDetail, PaperSection, RelatedPapersResponse)
- [x] Create `lib/api/papers.ts` API client (getPaper, getRelatedPapers) with mock mode
- [x] Implement PaperHeader component (metadata, links, copy citation)
- [x] Implement PaperSections component (collapsible sections)
- [x] Implement PaperAnalysis component (summary, highlights, methodology tabs)
- [x] Implement RelatedPapers component (horizontal card list)
- [x] Implement PaperDetailSkeleton (loading state)
- [x] Implement paper detail page (`app/papers/[id]/page.tsx`)
- [x] Run tests -- expect pass (Green) -- all 230 tests passing
- [x] Refactor: useReducer pattern to satisfy ESLint react-hooks/set-state-in-effect rule

## Phase 4: Integration
- [x] PaperCard already links to `/papers/{id}` (verified in existing code)
- [x] Run lint (`pnpm lint`) -- clean
- [x] Run full test suite (`pnpm test`) -- 36 files, 230 tests passing

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Update roadmap.md status to "done"
