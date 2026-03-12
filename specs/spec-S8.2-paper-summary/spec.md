# Spec S8.2 -- AI-Generated Paper Summary

## Overview
Generate structured AI-driven summaries for uploaded or indexed papers using the LLM client (Ollama/Gemini). Given a paper's abstract, sections, and metadata, produce a structured summary with: objective, method, key findings, contribution, and limitations. The summary is persisted and returned via an API endpoint.

## Dependencies
- **S8.1** (PDF Upload) -- provides `Paper` model with `pdf_content`, `sections`, `abstract`
- **S5.1** (LLM Client) -- provides `LLMProvider` protocol for text generation

## Target Location
- Service: `src/services/analysis/summarizer.py`
- Schemas: `src/schemas/api/analysis.py`
- Router: `src/routers/analysis.py`
- Tests: `tests/unit/test_analysis_summarizer.py`
- Notebook: `notebooks/specs/S8.2_summary.ipynb`

## Functional Requirements

### FR-1: Structured Summary Generation
- **What**: Generate a structured summary from a paper's content (abstract + sections + metadata)
- **Inputs**: `paper_id: UUID` -- identifies the paper in the database
- **Outputs**: `PaperSummary` with fields: `objective`, `method`, `key_findings` (list[str]), `contribution`, `limitations`
- **Edge cases**:
  - Paper not found in DB -> raise `PaperNotFoundError` (404)
  - Paper has no parsed content (only abstract) -> generate summary from abstract alone (with warning)
  - Paper has no abstract AND no sections -> raise `InsufficientContentError` (422)
  - LLM service unavailable -> raise `LLMServiceError` (503)
  - LLM returns malformed output -> attempt re-parse, fallback to raw text with warning

### FR-2: Content Extraction for Prompt
- **What**: Extract and format the most relevant paper content for the LLM prompt
- **Inputs**: `Paper` model instance
- **Outputs**: Formatted string with abstract + key sections (introduction, methodology, results, conclusion)
- **Edge cases**:
  - Very long papers -> truncate to ~4000 words to stay within context window
  - Missing sections -> use available content, note missing sections
  - Only abstract available -> use abstract with explicit instruction to summarize from abstract only

### FR-3: Summary API Endpoint
- **What**: REST endpoint to request and retrieve a paper summary
- **Inputs**: `POST /api/v1/papers/{paper_id}/summary`
- **Outputs**: `SummaryResponse` JSON with structured summary + metadata (model, provider, latency)
- **Edge cases**:
  - Invalid UUID format -> 422 validation error
  - Paper not found -> 404
  - LLM timeout -> 503 with retry suggestion

### FR-4: Summary Caching (Optional)
- **What**: Cache generated summaries to avoid redundant LLM calls
- **Inputs**: `paper_id`
- **Outputs**: Return cached summary if available, otherwise generate and cache
- **Edge cases**:
  - Cache miss -> generate fresh summary
  - Force regeneration via `?force=true` query param

## Tangible Outcomes
- [ ] `SummarizerService` class with `async def summarize(paper_id) -> PaperSummary`
- [ ] `PaperSummary` Pydantic model with 5 structured fields (objective, method, key_findings, contribution, limitations)
- [ ] `SummaryResponse` API response model with summary + metadata
- [ ] `POST /api/v1/papers/{paper_id}/summary` endpoint returning structured summary
- [ ] Content extraction handles abstract-only and full-paper cases
- [ ] LLM prompt enforces structured output format
- [ ] All external services mocked in tests
- [ ] Notebook `S8.2_summary.ipynb` with executable verification cells

## Test-Driven Requirements

### Tests to Write First
1. `test_summarize_full_paper`: Mock LLM + paper repo, verify structured summary returned with all 5 fields
2. `test_summarize_abstract_only`: Paper with abstract but no sections -> summary generated with warning
3. `test_summarize_paper_not_found`: Non-existent paper_id -> PaperNotFoundError
4. `test_summarize_insufficient_content`: Paper with no abstract and no sections -> InsufficientContentError
5. `test_summarize_llm_failure`: LLM provider raises exception -> LLMServiceError propagated
6. `test_extract_content_full_paper`: Verify content extraction formats abstract + key sections correctly
7. `test_extract_content_truncation`: Very long paper content truncated to ~4000 words
8. `test_summary_response_schema`: Verify API response model validates correctly
9. `test_summary_endpoint_success`: POST endpoint returns 200 with valid summary
10. `test_summary_endpoint_not_found`: POST endpoint returns 404 for missing paper
11. `test_force_regeneration`: `?force=true` bypasses cache and generates fresh summary

### Mocking Strategy
- Mock `LLMProvider.generate()` with `AsyncMock` returning structured text
- Mock `PaperRepository.get_by_id()` with `AsyncMock` returning `Paper` fixture
- Mock `AsyncSession` for database operations
- No real LLM calls in unit tests

### Coverage
- All public methods of `SummarizerService` tested
- All edge cases (missing content, LLM failure, malformed output) covered
- API endpoint tested via `httpx.AsyncClient` + `ASGITransport`
