# Checklist -- Spec S9.5: Paper Upload UI

## Phase 1: Setup & Dependencies
- [x] Verify S9.2 (Layout & Navigation) is done
- [x] Create target directories: `frontend/src/app/upload/`, `frontend/src/components/upload/`, `frontend/src/lib/api/upload.ts`, `frontend/src/types/upload.ts`

## Phase 2: Tests First (TDD)
- [x] Create test files for upload components
- [x] Write failing tests for DropZone component (11 tests)
- [x] Write failing tests for UploadProgress component (5 tests)
- [x] Write failing tests for AnalysisResults component (8 tests)
- [x] Write failing tests for upload API client (2 tests)
- [x] Run tests — expect failures (Red) ✓ 3 test files failed as expected

## Phase 3: Implementation
- [x] Implement TypeScript types (`types/upload.ts`)
- [x] Implement upload API client with mock fallback (`lib/api/upload.ts`)
- [x] Implement DropZone component (`components/upload/drop-zone.tsx`)
- [x] Implement UploadProgress component (`components/upload/upload-progress.tsx`)
- [x] Implement AnalysisResults component (`components/upload/analysis-results.tsx`)
- [x] Implement Upload page (`app/upload/page.tsx`)
- [x] Run tests — expect pass (Green) ✓ 26/26 tests pass
- [x] Refactor if needed

## Phase 4: Integration
- [x] Verify upload page is accessible via sidebar navigation (already wired in S9.2)
- [x] Run lint (ESLint) ✓ passes
- [x] Run full test suite (Vitest) ✓ 183/183 tests pass across 30 files
- [x] Verify responsive design (mobile + desktop) — Tailwind responsive classes used
- [x] Verify dark mode — uses theme-aware CSS variables

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Update roadmap.md status to "done"
