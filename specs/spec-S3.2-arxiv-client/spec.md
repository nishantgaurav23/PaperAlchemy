# Spec S3.2 -- ArXiv API Client

## Overview
Async arXiv API client with rate limiting (3s between requests), exponential backoff retry, category filtering, date-range queries, and PDF download with local caching. Uses `httpx` for HTTP and `feedparser` for Atom XML parsing.

## Dependencies
- S1.2 (Environment Configuration) — ArxivSettings for base_url, rate_limit_delay, max_results, category, timeout, max_retries

## Target Location
- `src/services/arxiv/client.py` — ArxivClient class
- `src/services/arxiv/factory.py` — Factory function (make_arxiv_client)
- `src/services/arxiv/__init__.py` — Public exports
- `src/schemas/arxiv.py` — ArxivPaper Pydantic schema (API response model)

## Functional Requirements

### FR-1: ArxivPaper Schema
- **What**: Pydantic model representing a parsed arXiv paper
- **Fields**: arxiv_id, title, authors (list[str]), abstract, categories (list[str]), published_date (str), updated_date (str | None), pdf_url (str)
- **Validation**: arxiv_id must be non-empty; title stripped of newlines

### FR-2: Rate Limiting
- **What**: Enforce minimum 3.0s delay between consecutive HTTP requests
- **How**: Track `_last_request_time`, sleep for remaining time if elapsed < delay
- **Edge cases**: First request has no delay; concurrent access (single-client assumption)

### FR-3: HTTP Request with Retry + Backoff
- **What**: Make GET requests with exponential backoff on failure
- **Retries**: Up to `max_retries` (default 3) attempts
- **Backoff**: `retry_delay * 2^attempt` seconds on 503, timeout, connection error
- **Rate limit**: On HTTP 429, wait `retry_delay * 2^attempt * 10` then retry
- **Fatal**: Other non-200 status codes raise ArxivAPIError immediately
- **Headers**: User-Agent header required by arXiv API guidelines

### FR-4: Query Building
- **What**: Build arXiv API search query string
- **Inputs**: category (str), from_date (YYYYMMDD), to_date (YYYYMMDD), search_query (str)
- **Output**: Query string like `cat:cs.AI AND submittedDate:[20240101 TO 20240131] AND all:transformers`
- **Default**: Falls back to configured category if none specified

### FR-5: Fetch Papers
- **What**: Query arXiv API and parse Atom XML response into ArxivPaper list
- **Inputs**: max_results, category, from_date, to_date, search_query, sort_by, sort_order, start (pagination)
- **Outputs**: list[ArxivPaper]
- **Parsing**: Use feedparser; extract arxiv_id from URL, strip version suffix, parse authors/categories/dates
- **Edge cases**: Malformed entries skipped with warning; empty results return []

### FR-6: PDF Download
- **What**: Download PDF to local cache directory with validation
- **Inputs**: ArxivPaper (or arxiv_id + pdf_url), force (bool)
- **Outputs**: Path | None (None on failure)
- **Caching**: Skip download if file exists (unless force=True)
- **Validation**: Check Content-Type contains "pdf", max 50MB, verify %PDF- magic bytes
- **Atomic**: Write to .tmp then rename to prevent partial files

### FR-7: Factory Function
- **What**: Create cached singleton ArxivClient from settings
- **How**: `make_arxiv_client()` reads ArxivSettings, returns cached instance

## Tangible Outcomes
- [ ] `ArxivClient` class with async `fetch_papers()` returning list[ArxivPaper]
- [ ] Rate limiting enforces >= 3.0s between requests
- [ ] Retry with exponential backoff on 429, 503, timeout, connection errors
- [ ] `download_pdf()` caches PDFs locally, validates format
- [ ] `ArxivPaper` Pydantic schema with all fields
- [ ] Factory `make_arxiv_client()` creates client from settings
- [ ] All external HTTP calls mocked in tests

## Test-Driven Requirements

### Tests to Write First
1. `test_arxiv_paper_schema`: Validate ArxivPaper model creation + field types
2. `test_rate_limiting`: Verify >= 3s delay between consecutive requests
3. `test_retry_on_503`: Retry with backoff on service unavailable
4. `test_retry_on_429`: Retry with extended wait on rate limit
5. `test_retry_on_timeout`: Retry on request timeout
6. `test_fatal_error_no_retry`: Non-retryable errors raise immediately
7. `test_max_retries_exceeded`: Raise ArxivAPIError after all retries fail
8. `test_build_query_category_only`: Query with just category
9. `test_build_query_date_range`: Query with date range
10. `test_build_query_search_terms`: Query with search_query
11. `test_build_query_combined`: All filters combined
12. `test_fetch_papers_success`: Parse valid Atom feed into ArxivPaper list
13. `test_fetch_papers_empty`: Empty feed returns []
14. `test_fetch_papers_malformed_entry`: Skips bad entries gracefully
15. `test_download_pdf_success`: Download, validate, cache PDF
16. `test_download_pdf_cached`: Return cached path without download
17. `test_download_pdf_force`: Re-download even if cached
18. `test_download_pdf_invalid_content_type`: Return None on non-PDF response
19. `test_download_pdf_too_large`: Return None for >50MB
20. `test_download_pdf_invalid_magic`: Return None + cleanup on bad magic bytes
21. `test_factory_creates_client`: make_arxiv_client returns ArxivClient
22. `test_factory_caches_instance`: Same instance on multiple calls

### Mocking Strategy
- Mock `httpx.AsyncClient` responses for all HTTP calls
- Use `feedparser.parse()` on constructed Atom XML strings for integration-level parsing tests
- Mock `asyncio.sleep` / time tracking for rate limit tests
- Use `tmp_path` fixture for PDF cache directory

### Coverage
- All public methods tested
- All retry paths (429, 503, timeout, connection error) tested
- All PDF validation paths tested
- Edge cases: empty results, malformed XML entries, first-request-no-delay
