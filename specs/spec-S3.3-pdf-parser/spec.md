# Spec S3.3 -- PDF Parser (Docling)

## Overview
Section-aware PDF parsing service using Docling for structured content extraction from academic papers. Extracts sections (with hierarchy), tables, figures (captions), and raw text. Enforces file size (50MB max) and page limits (30 pages). Runs Docling synchronously in a thread pool with async wrapper and timeout protection. Lazy-initializes the Docling converter for fast startup.

## Dependencies
- **S1.2** (Environment configuration) — for `PDFParserSettings` in config

## Target Location
- `src/services/pdf_parser/service.py` — main service class
- `src/services/pdf_parser/factory.py` — singleton factory
- `src/services/pdf_parser/__init__.py` — public API
- `src/schemas/pdf.py` — Pydantic models (PDFContent, Section)

## Functional Requirements

### FR-1: PDF Validation
- **What**: Validate a PDF file before parsing — check existence, extension, file size, and PDF magic bytes
- **Inputs**: `pdf_path: Path`
- **Outputs**: Raises `PDFParserError` on failure, returns `None` on success
- **Edge cases**: File not found, non-PDF extension, file too large (>50MB), corrupted file (wrong magic bytes), empty file

### FR-2: Section-Aware Parsing
- **What**: Parse a PDF using Docling and extract structured content: raw text, sections (title + content + heading level), tables (text representation), and figure captions
- **Inputs**: `pdf_path: Path`
- **Outputs**: `PDFContent` with `raw_text`, `sections: list[Section]`, `tables: list[str]`, `figures: list[str]`, `page_count: int`, `parser_used: str`, `parser_time_seconds: float`
- **Edge cases**: PDF with no headings (entire content in one "Introduction" section), PDF with only images (empty text), Docling not installed (raise error)

### FR-3: Async Parsing with Timeout
- **What**: Run Docling (synchronous) in a thread pool executor, wrapped with `asyncio.wait_for` timeout
- **Inputs**: `pdf_path: Path`
- **Outputs**: `PDFContent` on success, raises `PDFParserTimeoutError` on timeout, raises `PDFParserError` on parse failure
- **Edge cases**: Timeout (default 120s), thread pool exhaustion, concurrent parsing requests

### FR-4: Batch Parsing
- **What**: Parse multiple PDFs sequentially with optional `continue_on_error`
- **Inputs**: `pdf_paths: list[Path]`, `continue_on_error: bool = True`
- **Outputs**: `dict[str, PDFContent | None]` mapping filename to result
- **Edge cases**: Empty list, all failures, partial failures with `continue_on_error=False`

### FR-5: Factory & Configuration
- **What**: Add `PDFParserSettings` to config (max_pages, max_file_size_mb, timeout) and create a singleton factory
- **Inputs**: Settings from environment variables (`PDF_PARSER__MAX_PAGES`, etc.)
- **Outputs**: Cached `PDFParserService` instance
- **Edge cases**: Missing config (use defaults: 30 pages, 50MB, 120s timeout)

### FR-6: Resource Cleanup
- **What**: Shut down thread pool executor and release Docling converter on close
- **Inputs**: None
- **Outputs**: Resources freed
- **Edge cases**: Double-close, close before any parsing

## Tangible Outcomes
- [ ] `src/services/pdf_parser/service.py` exists with `PDFParserService` class
- [ ] `src/services/pdf_parser/factory.py` exists with `make_pdf_parser_service()`
- [ ] `src/services/pdf_parser/__init__.py` exports public API
- [ ] `src/schemas/pdf.py` exists with `PDFContent` and `Section` models
- [ ] `src/config.py` has `PDFParserSettings` with `max_pages=30`, `max_file_size_mb=50`, `timeout=120`
- [ ] File validation rejects: missing files, non-PDF, >50MB, invalid magic bytes
- [ ] Docling converter is lazy-initialized (not at import time)
- [ ] Parsing runs in thread pool with async timeout
- [ ] Batch parsing supports `continue_on_error`
- [ ] `tests/unit/test_pdf_parser.py` — all tests passing
- [ ] `notebooks/specs/S3.3_pdf_parser.ipynb` — interactive demo
- [ ] Factory produces singleton instance from settings

## Test-Driven Requirements

### Tests to Write First
1. `test_validate_file_not_found` — raises error for missing file
2. `test_validate_file_not_pdf` — raises error for .txt file
3. `test_validate_file_too_large` — raises error for >50MB file
4. `test_validate_file_invalid_magic` — raises error for file without %PDF- header
5. `test_validate_file_success` — valid PDF passes validation
6. `test_parse_pdf_extracts_sections` — sections extracted with title, content, level
7. `test_parse_pdf_extracts_tables` — tables extracted as text
8. `test_parse_pdf_extracts_figures` — figure captions extracted
9. `test_parse_pdf_raw_text` — raw text extracted
10. `test_parse_pdf_no_headings` — content grouped into single section
11. `test_parse_pdf_timeout` — raises `PDFParserTimeoutError` on timeout
12. `test_parse_pdf_docling_not_installed` — raises `PDFParserError`
13. `test_parse_multiple_all_success` — batch parsing all succeed
14. `test_parse_multiple_partial_failure` — continue_on_error=True returns None for failures
15. `test_parse_multiple_stop_on_error` — continue_on_error=False raises on first failure
16. `test_factory_creates_singleton` — same instance returned
17. `test_factory_uses_settings` — settings applied correctly
18. `test_close_cleans_resources` — executor shut down, converter cleared
19. `test_section_model_validation` — Pydantic model for Section
20. `test_pdf_content_model_validation` — Pydantic model for PDFContent

### Mocking Strategy
- **Mock Docling** (`docling.document_converter.DocumentConverter`) — never call real Docling in unit tests
- Mock the converter's `convert()` return value with a fake document object that has `texts`, `tables`, `pictures`, `export_to_text()`
- Use `tmp_path` fixture for creating test PDF files (with valid %PDF- header)
- Mock `Path.stat()` for file size checks
- Use `AsyncMock` and `patch` for async tests

### Coverage
- All public methods tested
- All validation edge cases covered
- Timeout behavior verified
- Factory singleton behavior verified
