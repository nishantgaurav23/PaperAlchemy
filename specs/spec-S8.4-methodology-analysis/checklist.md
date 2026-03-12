# Checklist S8.4 -- Methodology & Findings Deep-Dive

## Phase 1: Schemas
- [x] `DatasetInfo` model (name, description, size)
- [x] `ResultEntry` model (metric, value, context)
- [x] `MethodologyAnalysis` model (research_design, datasets, baselines, key_results, statistical_significance, reproducibility_notes)
- [x] `MethodologyResponse` model (paper_id, analysis, model, provider, latency_ms, warning)

## Phase 2: Service (TDD)
- [x] Write failing tests first (16 tests)
- [x] `MethodologyService.__init__` (llm_provider, paper_repo)
- [x] `MethodologyService.analyze_methodology(paper_id, force)` — main method
- [x] `MethodologyService._prepare_content(paper)` — format content for methodology analysis
- [x] `MethodologyService._parse_analysis(text)` — parse LLM JSON output with fallback
- [x] Paper not found -> PaperNotFoundError
- [x] Insufficient content -> InsufficientContentError
- [x] LLM failure -> LLMServiceError
- [x] Malformed LLM output -> fallback parsing with warning
- [x] All 16 tests passing

## Phase 3: Router Endpoint
- [x] `POST /api/v1/papers/{paper_id}/methodology` endpoint
- [x] Error handling (404, 422, 503)
- [x] Endpoint tests passing

## Phase 4: Integration & Verification
- [x] Service registered in `__init__.py`
- [x] Router wired in analysis router
- [x] Notebook `S8.4_methodology.ipynb` created with executable cells
- [x] All tests pass (`pytest tests/unit/test_analysis_methodology.py -v`)
- [x] Lint clean (`ruff check src/services/analysis/methodology.py`)
- [x] Roadmap updated to `done`
- [x] Spec summary appended to `docs/spec-summaries.md`
