# Spec S8.5 -- Side-by-Side Paper Comparison

## Overview
Compare 2 or more academic papers side-by-side using the LLM client (Ollama/Gemini). Given multiple paper IDs, generate a structured comparison covering: methods, results, contributions, and limitations. The comparison is returned via an API endpoint on the analysis router.

## Dependencies
- **S8.2** (Paper Summary) -- provides `SummarizerService` and establishes the analysis service pattern with `LLMProvider`, `PaperRepository`, content extraction, and structured JSON output parsing

## Target Location
- Service: `src/services/analysis/comparator.py`
- Schemas: `src/schemas/api/analysis.py` (extend existing)
- Router: `src/routers/analysis.py` (add endpoint)
- Tests: `tests/unit/test_analysis_comparator.py`
- Notebook: `notebooks/specs/S8.5_comparison.ipynb`

## Functional Requirements

### FR-1: Multi-Paper Comparison Generation
- **What**: Generate a structured side-by-side comparison from 2+ papers' content (abstract + sections + metadata)
- **Inputs**: `paper_ids: list[UUID]` -- identifies 2+ papers in the database (minimum 2, maximum 5)
- **Outputs**: `PaperComparison` with fields:
  - `papers: list[ComparedPaper]` -- metadata for each paper (id, title, authors)
  - `methods_comparison: str` -- how the approaches/methodologies differ and overlap
  - `results_comparison: str` -- comparative analysis of results and findings
  - `contributions_comparison: str` -- what each paper uniquely contributes
  - `limitations_comparison: str` -- comparative limitations and gaps
  - `common_themes: list[str]` -- shared themes, topics, or approaches (1-5 items)
  - `key_differences: list[str]` -- most notable differences (1-5 items)
  - `verdict: str` -- brief overall synthesis (which excels at what, complementarity)
- **Edge cases**:
  - Fewer than 2 paper_ids -> raise `ValidationError` (422)
  - More than 5 paper_ids -> raise `ValidationError` (422)
  - Any paper not found in DB -> raise `PaperNotFoundError` (404) with the missing paper_id
  - Any paper has no abstract AND no sections -> raise `InsufficientContentError` (422)
  - LLM service unavailable -> raise `LLMServiceError` (503)
  - LLM returns malformed output -> attempt re-parse, fallback to raw text with warning
  - Duplicate paper_ids -> deduplicate silently

### FR-2: Multi-Paper Content Extraction
- **What**: Extract and format content from multiple papers into a single comparative prompt
- **Inputs**: List of `Paper` model instances
- **Outputs**: Formatted string with each paper's content clearly labelled (Paper 1, Paper 2, etc.)
- **Edge cases**:
  - Very long combined content -> truncate each paper proportionally to stay within ~6000 words total
  - Missing sections in some papers -> use available content, note gaps
  - Only abstracts available for some papers -> proceed with abstracts, warn in output

### FR-3: Comparison API Endpoint
- **What**: REST endpoint to request a paper comparison
- **Inputs**: `POST /api/v1/papers/compare` with JSON body `{"paper_ids": ["uuid1", "uuid2", ...]}`
- **Outputs**: `ComparisonResponse` JSON with structured comparison + metadata (model, provider, latency_ms)
- **Edge cases**:
  - Invalid UUID format -> 422 validation error
  - Fewer than 2 or more than 5 paper IDs -> 422 validation error
  - Any paper not found -> 404
  - LLM timeout -> 503 with retry suggestion

## Tangible Outcomes
- [ ] `ComparatorService` class with `async def compare(paper_ids) -> ComparisonResponse`
- [ ] `PaperComparison` Pydantic model with comparison fields (methods, results, contributions, limitations, common_themes, key_differences, verdict)
- [ ] `ComparedPaper` Pydantic model (id, title, authors)
- [ ] `ComparisonRequest` Pydantic model with `paper_ids: list[UUID]` (min 2, max 5)
- [ ] `ComparisonResponse` API response model with comparison + metadata
- [ ] Multi-paper content extraction with per-paper labelling and proportional truncation
- [ ] LLM prompt enforces structured JSON comparison output
- [ ] `POST /api/v1/papers/compare` endpoint returning structured comparison
- [ ] All external services mocked in tests
- [ ] Notebook `S8.5_comparison.ipynb` with executable verification cells

## Test-Driven Requirements

### Tests to Write First
1. `test_compare_two_papers`: Mock LLM + paper repo, verify structured comparison returned with all fields
2. `test_compare_three_papers`: Compare 3 papers, verify all are included in output
3. `test_compare_paper_not_found`: One paper_id doesn't exist -> PaperNotFoundError
4. `test_compare_insufficient_content`: One paper has no abstract/sections -> InsufficientContentError
5. `test_compare_fewer_than_two`: Single paper_id -> validation error
6. `test_compare_more_than_five`: Six paper_ids -> validation error
7. `test_compare_duplicate_ids`: Duplicate IDs deduplicated
8. `test_compare_llm_failure`: LLM provider raises exception -> LLMServiceError
9. `test_compare_malformed_llm_output`: LLM returns non-JSON -> fallback with warning
10. `test_extract_multi_content`: Verify multi-paper content extraction formats correctly with labels
11. `test_extract_multi_content_truncation`: Combined content truncated proportionally
12. `test_comparison_endpoint_success`: POST endpoint returns 200 with valid comparison
13. `test_comparison_endpoint_not_found`: POST endpoint returns 404 for missing paper
14. `test_comparison_endpoint_validation`: POST with <2 papers returns 422

### Mocking Strategy
- Mock `LLMProvider.generate()` with `AsyncMock` returning structured JSON text
- Mock `PaperRepository.get_by_id()` with `AsyncMock` returning `Paper` fixtures
- Mock `AsyncSession` for database operations
- No real LLM calls in unit tests

### Coverage
- All public methods of `ComparatorService` tested
- All edge cases (missing content, LLM failure, malformed output, validation) covered
- API endpoint tested via `httpx.AsyncClient` + `ASGITransport`
