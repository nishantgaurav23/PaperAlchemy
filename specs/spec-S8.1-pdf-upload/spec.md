# Spec S8.1 -- PDF Upload Endpoint

## Overview
PDF upload endpoint that accepts user-uploaded academic papers via multipart form data, validates them (PDF-only, 50MB max), parses them with Docling, stores metadata in PostgreSQL, chunks the content, embeds the chunks, and indexes them in OpenSearch for retrieval. This is the entry point for Phase 8 (Paper Upload & Analysis) — all downstream analysis specs (S8.2-S8.5) depend on this.

## Dependencies
- **S3.3** (PDF Parser — Docling) — provides `PDFParserService` for parsing uploaded PDFs
- **S2.4** (Dependency Injection) — provides DI patterns and typed `Annotated[]` dependencies
- **S3.1** (Paper Model & Repository) — provides `Paper` ORM model and `PaperRepository` for persistence
- **S4.2** (Text Chunker) — provides `TextChunker` for section-aware chunking
- **S4.3** (Embedding Service) — provides `JinaEmbeddingService` for vectorizing chunks
- **S4.1** (OpenSearch Client) — provides `OpenSearchClient` for bulk indexing

## Target Location
- `src/routers/upload.py` — Upload router with POST endpoint
- `src/schemas/api/upload.py` — Request/response Pydantic models
- `src/services/upload/` — Upload orchestration service
- `tests/unit/test_upload_router.py` — Router tests
- `tests/unit/test_upload_service.py` — Service tests

## Functional Requirements

### FR-1: PDF File Validation
- **What**: Validate uploaded file is a valid PDF within size limits before any processing
- **Inputs**: `UploadFile` (FastAPI multipart form file)
- **Outputs**: Validated file bytes or `HTTPException(422)`
- **Validation rules**:
  - File must have `.pdf` extension (case-insensitive)
  - Content-type must be `application/pdf` (if provided)
  - File size must be ≤ 50MB (configurable via settings)
  - First bytes must match PDF magic bytes (`%PDF`)
  - Empty files rejected
- **Edge cases**: Missing file, non-PDF extension, corrupted file, exactly 50MB boundary

### FR-2: PDF Parsing & Metadata Extraction
- **What**: Parse uploaded PDF using Docling to extract text, sections, and metadata
- **Inputs**: Validated PDF file bytes
- **Outputs**: `PDFContent` (raw_text, sections, page_count) + extracted metadata (title, abstract)
- **Process**:
  1. Save uploaded bytes to temporary file
  2. Call `PDFParserService.parse_pdf(tmp_path)`
  3. Extract title from first section heading or filename
  4. Extract abstract from "Abstract" section if found
  5. Clean up temporary file
- **Edge cases**: Parse failure (return 422 with error), no extractable text, no sections found

### FR-3: Paper Record Creation
- **What**: Create a `Paper` database record for the uploaded document
- **Inputs**: Extracted metadata + parsed content
- **Outputs**: `Paper` ORM instance with UUID
- **Fields**:
  - `arxiv_id`: Generated as `upload_{uuid}` (not a real arXiv ID)
  - `title`: From PDF metadata or filename (sans extension)
  - `authors`: From PDF metadata if extractable, else empty list
  - `abstract`: From parsed "Abstract" section or first 500 chars
  - `pdf_content`: Full raw text
  - `sections`: Parsed section JSON
  - `parsing_status`: `"success"`
  - `source`: `"upload"` (to distinguish from arXiv-fetched papers)
- **Edge cases**: Duplicate upload (upsert based on content hash or allow duplicates with unique IDs)

### FR-4: Content Chunking & Indexing
- **What**: Chunk parsed content, embed chunks, and index in OpenSearch
- **Inputs**: Parsed `PDFContent`, paper metadata
- **Outputs**: Number of chunks indexed
- **Process**:
  1. `TextChunker.chunk_paper()` → list of `TextChunk`
  2. `EmbeddingService.embed_documents()` → vectors for each chunk
  3. `OpenSearchClient.bulk_index()` → index chunks with embeddings + metadata
- **Edge cases**: Empty content (0 chunks — still save paper, return warning), embedding API failure (save paper but report indexing failure), OpenSearch failure (save paper but report indexing failure)

### FR-5: Upload Response
- **What**: Return structured response with paper metadata and processing results
- **Inputs**: Paper record, chunk count, any warnings
- **Outputs**: `UploadResponse` JSON
- **Fields**:
  - `paper_id`: UUID of created paper
  - `title`: Extracted or derived title
  - `authors`: Extracted authors list
  - `abstract`: Extracted abstract (truncated if >500 chars)
  - `page_count`: Number of pages
  - `chunks_indexed`: Count of indexed chunks
  - `parsing_status`: "success" or "partial"
  - `indexing_status`: "success", "partial", or "failed"
  - `warnings`: List of non-fatal issues (e.g., "no abstract found", "indexing failed")
  - `message`: Human-readable status summary

### FR-6: Upload Settings Configuration
- **What**: Configurable upload settings via environment variables
- **Inputs**: Environment variables with `UPLOAD__` prefix
- **Outputs**: `UploadSettings` Pydantic model
- **Settings**:
  - `max_file_size_mb`: int = 50
  - `upload_dir`: str = "data/uploaded_pdfs"
  - `allowed_extensions`: list[str] = [".pdf"]
  - `auto_index`: bool = True (whether to chunk+embed+index after parsing)

## Tangible Outcomes
- [ ] `POST /api/v1/upload` accepts PDF via multipart form and returns `UploadResponse`
- [ ] Rejects non-PDF files with 422 and clear error message
- [ ] Rejects files > 50MB with 413 and clear error message
- [ ] Parses PDF and extracts text + sections via Docling
- [ ] Creates `Paper` record in PostgreSQL with `source="upload"`
- [ ] Chunks content, embeds, and indexes in OpenSearch
- [ ] Returns paper_id, title, chunks_indexed, and any warnings
- [ ] Graceful degradation: paper saved even if indexing fails (with warning)
- [ ] All services injected via FastAPI `Depends()` pattern
- [ ] Upload settings configurable via environment variables

## Test-Driven Requirements

### Tests to Write First

#### Router Tests (`tests/unit/test_upload_router.py`)
1. `test_upload_valid_pdf` — Upload valid PDF → 200 with paper_id and chunks_indexed
2. `test_upload_non_pdf_rejected` — Upload .txt file → 422 with "PDF only" message
3. `test_upload_oversized_rejected` — Upload >50MB file → 413 with size limit message
4. `test_upload_empty_file_rejected` — Upload empty file → 422
5. `test_upload_invalid_magic_bytes` — Upload file with .pdf ext but non-PDF content → 422
6. `test_upload_parse_failure` — PDF parser raises error → 422 with parse error detail
7. `test_upload_indexing_failure_graceful` — Indexing fails but paper saved → 200 with warning
8. `test_upload_response_schema` — Verify response matches `UploadResponse` schema exactly

#### Service Tests (`tests/unit/test_upload_service.py`)
9. `test_validate_pdf_file_valid` — Valid PDF passes all checks
10. `test_validate_pdf_file_too_large` — Raises validation error for oversized files
11. `test_validate_pdf_file_wrong_extension` — Raises for non-.pdf extension
12. `test_validate_pdf_file_wrong_magic_bytes` — Raises for non-PDF content
13. `test_process_upload_full_pipeline` — Mock all services → verify full pipeline executes
14. `test_process_upload_extracts_title` — Title extracted from PDF content
15. `test_process_upload_extracts_abstract` — Abstract extracted from sections
16. `test_process_upload_no_abstract_fallback` — Falls back to first N chars when no abstract section
17. `test_process_upload_creates_paper_record` — Verify Paper record created with correct fields
18. `test_process_upload_chunks_and_indexes` — Verify chunking + embedding + indexing flow
19. `test_process_upload_indexing_fails_gracefully` — Paper saved, warning returned on index failure

### Mocking Strategy
- **PDFParserService**: Mock `parse_pdf()` to return fixture `PDFContent`
- **PaperRepository**: Mock `create()` to return fixture `Paper` with UUID
- **TextChunker**: Mock `chunk_paper()` to return fixture chunks
- **EmbeddingService**: Mock `embed_documents()` to return fixture vectors
- **OpenSearchClient**: Mock `bulk_index()` to return success/failure
- **Database Session**: Mock async session with commit/rollback
- **UploadFile**: Use `UploadFile(file=BytesIO(pdf_bytes), filename="test.pdf")`

### Coverage
- All public functions tested
- All validation paths tested (valid + each rejection reason)
- Graceful degradation paths tested
- Error propagation tested
- Response schema tested
