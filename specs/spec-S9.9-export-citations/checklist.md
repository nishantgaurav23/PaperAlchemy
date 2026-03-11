# Checklist -- Spec S9.9: Export & Citations

## Phase 1: Setup & Dependencies
- [x] Verify S9.6 (Paper detail view) is "done"
- [x] Create `frontend/src/components/export/` directory
- [x] Create `frontend/src/lib/export/` directory for formatting utilities

## Phase 2: Tests First (TDD)
- [x] Create `frontend/src/lib/export/formatters.test.ts` — format function tests
- [x] Create `frontend/src/lib/export/clipboard.test.ts` — clipboard + download tests
- [x] Create `frontend/src/components/export/export-button.test.tsx` — component tests
- [x] Write failing tests for BibTeX formatting (FR-1)
- [x] Write failing tests for Markdown formatting (FR-2)
- [x] Write failing tests for Slide snippet formatting (FR-3)
- [x] Write failing tests for clipboard copy (FR-4)
- [x] Write failing tests for file download (FR-5)
- [x] Write failing tests for ExportButton component (FR-7)
- [x] Write failing tests for bulk export (FR-6)
- [x] Run tests — expect failures (Red)

## Phase 3: Implementation
- [x] Implement `formatBibtex(paper)` — pass tests
- [x] Implement `formatMarkdown(paper)` — pass tests
- [x] Implement `formatSlideSnippet(paper)` — pass tests
- [x] Implement `copyToClipboard(text)` — pass tests
- [x] Implement `downloadFile(content, filename)` — pass tests
- [x] Implement `formatBulkBibtex(papers)` — pass tests
- [x] Implement `formatBulkMarkdown(papers)` — pass tests
- [x] Implement `ExportButton` component — pass tests
- [x] Run tests — expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Add ExportButton to paper detail page header (PaperHeader)
- [x] Add bulk export to collection detail page
- [x] Run lint (`pnpm lint`)
- [x] Run full test suite (`pnpm test`)

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Update roadmap.md status to "done"
