# Checklist -- Spec S8.1: PDF Upload Endpoint

## Phase 1: Setup & Dependencies
- [x] Verify all dependency specs are "done" (S3.3, S2.4, S3.1, S4.2, S4.3, S4.1)
- [x] Create target files/directories:
  - [x] `src/routers/upload.py`
  - [x] `src/schemas/api/upload.py`
  - [x] `src/services/upload/__init__.py`
  - [x] `src/services/upload/service.py`
  - [x] `tests/unit/test_upload_router.py`
  - [x] `tests/unit/test_upload_service.py`

## Phase 2: Tests First (TDD)
- [x] Create test files
- [x] Write failing router tests:
  - [x] `test_upload_valid_pdf`
  - [x] `test_upload_non_pdf_rejected`
  - [x] `test_upload_oversized_rejected`
  - [x] `test_upload_empty_file_rejected`
  - [x] `test_upload_invalid_magic_bytes`
  - [x] `test_upload_parse_failure`
  - [x] `test_upload_indexing_failure_graceful`
  - [x] `test_upload_response_schema`
- [x] Write failing service tests:
  - [x] `test_validate_pdf_file_valid`
  - [x] `test_validate_pdf_file_too_large`
  - [x] `test_validate_pdf_file_wrong_extension`
  - [x] `test_validate_pdf_file_wrong_magic_bytes`
  - [x] `test_process_upload_full_pipeline`
  - [x] `test_process_upload_extracts_title`
  - [x] `test_process_upload_extracts_abstract`
  - [x] `test_process_upload_no_abstract_fallback`
  - [x] `test_process_upload_creates_paper_record`
  - [x] `test_process_upload_chunks_and_indexes`
  - [x] `test_process_upload_indexing_fails_gracefully`
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Add `UploadSettings` — using UploadService constructor params (no env prefix needed yet)
- [x] Implement `src/schemas/api/upload.py` (UploadResponse)
- [x] Implement `src/services/upload/service.py` (UploadService):
  - [x] `validate_pdf()` — file validation (extension, size, magic bytes)
  - [x] `process_upload()` — full pipeline (parse → save → chunk → embed → index)
  - [x] `_extract_metadata()` — title, abstract, authors from parsed content
- [x] Implement `src/routers/upload.py` (POST /api/v1/upload)
- [x] Add DI entries to router (PDFParserDep, TextChunkerDep, UploadServiceDep)
- [x] Run tests -- expect pass (Green)
- [x] Refactor: cleaned up imports

## Phase 4: Integration
- [x] Register upload router in `src/main.py`
- [x] Run lint (`ruff check src/ tests/`)
- [x] Run full test suite (`pytest`) — 918 passed, 9 skipped

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook: `notebooks/specs/S8.1_upload.ipynb`
- [x] Update `roadmap.md` status to "done"
- [x] Append spec summary to `docs/spec-summaries.md`
