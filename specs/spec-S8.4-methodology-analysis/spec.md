# Spec S8.4 -- Methodology & Findings Deep-Dive

## Overview
Analyze a paper's methodology and findings in depth using the LLM client (Ollama/Gemini). Given a paper's content (abstract, sections, metadata), produce a structured analysis covering: research design, datasets used, baselines compared, key results (with metrics), statistical significance, and reproducibility notes. The analysis is returned via an API endpoint on the analysis router.

## Dependencies
- **S8.1** (PDF Upload) -- provides `Paper` model with `pdf_content`, `sections`, `abstract`
- **S5.1** (LLM Client) -- provides `LLMProvider` protocol for text generation

## Target Location
- Service: `src/services/analysis/methodology.py`
- Schemas: `src/schemas/api/analysis.py` (extend existing)
- Router: `src/routers/analysis.py` (add endpoint)
- Tests: `tests/unit/test_analysis_methodology.py`
- Notebook: `notebooks/specs/S8.4_methodology.ipynb`

## Functional Requirements

### FR-1: Methodology & Findings Analysis
- **What**: Extract structured methodology and findings analysis from a paper's content
- **Inputs**: `paper_id: UUID` -- identifies the paper in the database
- **Outputs**: `MethodologyAnalysis` with fields:
  - `research_design: str` -- type of study (experimental, observational, theoretical, survey, etc.) and overall approach (1-3 sentences)
  - `datasets: list[DatasetInfo]` -- datasets used, each with `name`, `description`, `size` (optional)
  - `baselines: list[str]` -- baseline methods/models compared against (1-10 items)
  - `key_results: list[ResultEntry]` -- key results, each with `metric`, `value`, `context` (e.g., "BLEU: 28.4 on WMT 2014 EN-DE")
  - `statistical_significance: str | None` -- notes on statistical tests, confidence intervals, p-values (null if not reported)
  - `reproducibility_notes: str | None` -- code availability, hyperparameters, compute resources (null if not mentioned)
- **Edge cases**:
  - Paper not found in DB -> raise `PaperNotFoundError` (404)
  - Paper has no parsed content (only abstract) -> extract what's possible from abstract (with warning)
  - Paper has no abstract AND no sections -> raise `InsufficientContentError` (422)
  - LLM service unavailable -> raise `LLMServiceError` (503)
  - LLM returns malformed output -> attempt re-parse, fallback to raw extraction with warning
  - Theoretical papers with no experiments -> `datasets` and `key_results` may be empty, `research_design` notes "theoretical"

### FR-2: Content Preparation for Methodology Analysis
- **What**: Extract and format paper content optimized for methodology/findings extraction
- **Inputs**: `Paper` model instance
- **Outputs**: Formatted string with abstract + methodology + experiments + results + discussion sections (prioritized)
- **Edge cases**:
  - Very long papers -> truncate to ~4000 words, prioritizing methodology, experiments, and results
  - Missing sections -> use available content, note missing sections
  - Only abstract available -> use abstract with explicit instruction to extract from abstract only

### FR-3: Methodology Analysis API Endpoint
- **What**: REST endpoint to request methodology & findings analysis
- **Inputs**: `POST /api/v1/papers/{paper_id}/methodology`
- **Outputs**: `MethodologyResponse` JSON with structured analysis + metadata (model, provider, latency_ms)
- **Edge cases**:
  - Invalid UUID format -> 422 validation error
  - Paper not found -> 404
  - LLM timeout -> 503 with retry suggestion

### FR-4: Analysis Caching (Optional)
- **What**: Cache generated analysis to avoid redundant LLM calls
- **Inputs**: `paper_id`
- **Outputs**: Return cached analysis if available, otherwise generate and cache
- **Edge cases**:
  - Cache miss -> generate fresh analysis
  - Force regeneration via `?force=true` query param

## Tangible Outcomes
- [ ] `MethodologyService` class with `async def analyze_methodology(paper_id) -> MethodologyResponse`
- [ ] `DatasetInfo` Pydantic model with `name`, `description`, `size` (optional)
- [ ] `ResultEntry` Pydantic model with `metric`, `value`, `context`
- [ ] `MethodologyAnalysis` Pydantic model with 6 structured fields
- [ ] `MethodologyResponse` API response model with analysis + metadata
- [ ] `POST /api/v1/papers/{paper_id}/methodology` endpoint returning structured analysis
- [ ] Content preparation handles abstract-only and full-paper cases
- [ ] LLM prompt enforces structured output format (JSON-parseable)
- [ ] All external services mocked in tests
- [ ] Notebook `S8.4_methodology.ipynb` with executable verification cells

## Test-Driven Requirements

### Tests to Write First
1. `test_analyze_methodology_full_paper`: Mock LLM + paper repo, verify structured analysis with all 6 fields
2. `test_analyze_methodology_abstract_only`: Paper with abstract but no sections -> analysis with warning
3. `test_analyze_methodology_paper_not_found`: Non-existent paper_id -> PaperNotFoundError
4. `test_analyze_methodology_insufficient_content`: No abstract and no sections -> InsufficientContentError
5. `test_analyze_methodology_llm_failure`: LLM provider raises exception -> LLMServiceError
6. `test_analyze_methodology_theoretical_paper`: Theoretical paper -> empty datasets/results, research_design notes "theoretical"
7. `test_prepare_content_methodology_focus`: Verify content preparation prioritizes methodology/results sections
8. `test_prepare_content_truncation`: Very long paper content truncated to ~4000 words
9. `test_methodology_response_schema`: Verify API response model validates correctly
10. `test_methodology_endpoint_success`: POST endpoint returns 200 with valid analysis
11. `test_methodology_endpoint_not_found`: POST endpoint returns 404 for missing paper
12. `test_force_regeneration`: `?force=true` bypasses cache and generates fresh analysis

### Mocking Strategy
- Mock `LLMProvider.generate()` with `AsyncMock` returning structured JSON text
- Mock `PaperRepository.get_by_id()` with `AsyncMock` returning `Paper` fixture
- Mock `AsyncSession` for database operations
- No real LLM calls in unit tests

### Coverage
- All public methods of `MethodologyService` tested
- All edge cases (missing content, LLM failure, malformed output, theoretical papers) covered
- API endpoint tested via `httpx.AsyncClient` + `ASGITransport`
