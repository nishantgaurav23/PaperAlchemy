# Checklist -- Spec S3.2: ArXiv API Client

## Phase 1: Setup & Dependencies
- [x] Verify S1.2 (Environment Configuration) is "done"
- [x] Create `src/services/arxiv/` directory
- [x] Create `src/schemas/arxiv.py` schema file
- [x] Verify httpx, feedparser in pyproject.toml

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_arxiv_client.py`
- [x] Write failing tests for ArxivPaper schema (FR-1)
- [x] Write failing tests for rate limiting (FR-2)
- [x] Write failing tests for retry logic (FR-3)
- [x] Write failing tests for query building (FR-4)
- [x] Write failing tests for fetch_papers (FR-5)
- [x] Write failing tests for download_pdf (FR-6)
- [x] Write failing tests for factory (FR-7)
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement ArxivPaper schema -- pass FR-1 tests
- [x] Implement ArxivClient._wait_for_rate_limit -- pass FR-2 tests
- [x] Implement ArxivClient._make_request -- pass FR-3 tests
- [x] Implement ArxivClient._build_query -- pass FR-4 tests
- [x] Implement ArxivClient._parse_entry -- helper for FR-5
- [x] Implement ArxivClient.fetch_papers -- pass FR-5 tests
- [x] Implement ArxivClient.download_pdf -- pass FR-6 tests
- [x] Implement factory make_arxiv_client -- pass FR-7 tests
- [x] Run tests -- expect pass (Green)
- [x] Refactor if needed

## Phase 4: Integration
- [x] Wire factory into dependency.py (register provider)
- [x] Export from `src/services/arxiv/__init__.py`
- [x] Run lint (ruff check src/ tests/)
- [x] Run full test suite

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Notebook created: `notebooks/specs/S3.2_arxiv_client.ipynb`
- [x] Update roadmap.md status to "done"
- [x] Append spec summary to docs/spec-summaries.md
