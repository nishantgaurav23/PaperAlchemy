# Spec S8.3 -- Key Highlights Extraction

## Overview
Extract key highlights and insights from uploaded or indexed papers using the LLM client (Ollama/Gemini). Given a paper's abstract, sections, and metadata, produce a structured list of highlights covering: novel contributions, important findings, practical implications, and notable limitations. The highlights are returned via an API endpoint on the analysis router.

## Dependencies
- **S8.1** (PDF Upload) -- provides `Paper` model with `pdf_content`, `sections`, `abstract`
- **S5.1** (LLM Client) -- provides `LLMProvider` protocol for text generation

## Target Location
- Service: `src/services/analysis/highlights.py`
- Schemas: `src/schemas/api/analysis.py` (extend existing)
- Router: `src/routers/analysis.py` (add endpoint)
- Tests: `tests/unit/test_analysis_highlights.py`
- Notebook: `notebooks/specs/S8.3_highlights.ipynb`

## Functional Requirements

### FR-1: Key Highlights Extraction
- **What**: Extract structured highlights from a paper's content (abstract + sections + metadata)
- **Inputs**: `paper_id: UUID` -- identifies the paper in the database
- **Outputs**: `PaperHighlights` with fields:
  - `novel_contributions: list[str]` -- what's new in this paper (1-5 items)
  - `important_findings: list[str]` -- key results and discoveries (1-5 items)
  - `practical_implications: list[str]` -- real-world applications and impact (1-5 items)
  - `limitations: list[str]` -- noted limitations and caveats (1-3 items)
  - `keywords: list[str]` -- extracted key terms/topics (3-10 items)
- **Edge cases**:
  - Paper not found in DB -> raise `PaperNotFoundError` (404)
  - Paper has no parsed content (only abstract) -> extract highlights from abstract alone (with warning)
  - Paper has no abstract AND no sections -> raise `InsufficientContentError` (422)
  - LLM service unavailable -> raise `LLMServiceError` (503)
  - LLM returns malformed output -> attempt re-parse, fallback to raw extraction with warning

### FR-2: Content Preparation for Highlights
- **What**: Extract and format paper content optimized for highlight extraction
- **Inputs**: `Paper` model instance
- **Outputs**: Formatted string with abstract + results + conclusion + discussion sections (prioritized for highlights)
- **Edge cases**:
  - Very long papers -> truncate to ~4000 words, prioritizing abstract, results, and conclusion
  - Missing sections -> use available content, note missing sections
  - Only abstract available -> use abstract with explicit instruction to extract from abstract only

### FR-3: Highlights API Endpoint
- **What**: REST endpoint to request and retrieve paper highlights
- **Inputs**: `POST /api/v1/papers/{paper_id}/highlights`
- **Outputs**: `HighlightsResponse` JSON with structured highlights + metadata (model, provider, latency_ms)
- **Edge cases**:
  - Invalid UUID format -> 422 validation error
  - Paper not found -> 404
  - LLM timeout -> 503 with retry suggestion

### FR-4: Highlights Caching (Optional)
- **What**: Cache generated highlights to avoid redundant LLM calls
- **Inputs**: `paper_id`
- **Outputs**: Return cached highlights if available, otherwise generate and cache
- **Edge cases**:
  - Cache miss -> generate fresh highlights
  - Force regeneration via `?force=true` query param

## Tangible Outcomes
- [ ] `HighlightsService` class with `async def extract_highlights(paper_id) -> PaperHighlights`
- [ ] `PaperHighlights` Pydantic model with 5 structured fields (novel_contributions, important_findings, practical_implications, limitations, keywords)
- [ ] `HighlightsResponse` API response model with highlights + metadata
- [ ] `POST /api/v1/papers/{paper_id}/highlights` endpoint returning structured highlights
- [ ] Content preparation handles abstract-only and full-paper cases
- [ ] LLM prompt enforces structured output format (JSON-parseable)
- [ ] All external services mocked in tests
- [ ] Notebook `S8.3_highlights.ipynb` with executable verification cells

## Test-Driven Requirements

### Tests to Write First
1. `test_extract_highlights_full_paper`: Mock LLM + paper repo, verify structured highlights returned with all 5 fields
2. `test_extract_highlights_abstract_only`: Paper with abstract but no sections -> highlights generated with warning
3. `test_extract_highlights_paper_not_found`: Non-existent paper_id -> PaperNotFoundError
4. `test_extract_highlights_insufficient_content`: Paper with no abstract and no sections -> InsufficientContentError
5. `test_extract_highlights_llm_failure`: LLM provider raises exception -> LLMServiceError propagated
6. `test_prepare_content_full_paper`: Verify content preparation formats abstract + key sections correctly
7. `test_prepare_content_truncation`: Very long paper content truncated to ~4000 words
8. `test_highlights_response_schema`: Verify API response model validates correctly
9. `test_highlights_endpoint_success`: POST endpoint returns 200 with valid highlights
10. `test_highlights_endpoint_not_found`: POST endpoint returns 404 for missing paper
11. `test_force_regeneration`: `?force=true` bypasses cache and generates fresh highlights

### Mocking Strategy
- Mock `LLMProvider.generate()` with `AsyncMock` returning structured JSON text
- Mock `PaperRepository.get_by_id()` with `AsyncMock` returning `Paper` fixture
- Mock `AsyncSession` for database operations
- No real LLM calls in unit tests

### Coverage
- All public methods of `HighlightsService` tested
- All edge cases (missing content, LLM failure, malformed output) covered
- API endpoint tested via `httpx.AsyncClient` + `ASGITransport`
