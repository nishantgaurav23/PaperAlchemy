# Checklist -- Spec S3.3: PDF Parser (Docling)

## Phase 1: Setup & Dependencies
- [x] Verify S1.2 (environment config) is "done"
- [x] Add `PDFParserSettings` to `src/config.py`
- [x] Create `src/schemas/pdf.py` with `Section` and `PDFContent` models
- [x] Create `src/services/pdf_parser/` directory

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_pdf_parser.py`
- [x] Write failing tests for FR-1 (file validation)
- [x] Write failing tests for FR-2 (section-aware parsing)
- [x] Write failing tests for FR-3 (async + timeout)
- [x] Write failing tests for FR-4 (batch parsing)
- [x] Write failing tests for FR-5 (factory + config)
- [x] Write failing tests for FR-6 (cleanup)
- [x] Write tests for Pydantic models (Section, PDFContent)
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement `src/schemas/pdf.py` -- Section + PDFContent models
- [x] Implement `src/services/pdf_parser/service.py` -- PDFParserService
- [x] Implement FR-1 -- `_validate_file()` -- pass validation tests
- [x] Implement FR-2 -- `_parse_sync()` -- pass parsing tests
- [x] Implement FR-3 -- `parse_pdf()` -- pass async/timeout tests
- [x] Implement FR-4 -- `parse_multiple()` -- pass batch tests
- [x] Implement FR-6 -- `close()` -- pass cleanup tests
- [x] Implement `src/services/pdf_parser/factory.py` -- pass factory tests
- [x] Implement `src/services/pdf_parser/__init__.py` -- exports
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Wire `PDFParserSettings` into `Settings` class in config
- [x] Register in dependency injection if needed
- [x] Run lint (`ruff check src/ tests/`)
- [x] Run full test suite

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook: `notebooks/specs/S3.3_pdf_parser.ipynb`
- [x] Update roadmap.md status to "done"
- [x] Append spec summary to `docs/spec-summaries.md`
