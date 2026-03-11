# Spec S2.3 -- Error Handling & Middleware

## Overview
Global error handling and request logging middleware for the FastAPI application. Provides custom exception classes organized by subsystem, a global exception handler that returns structured JSON error responses, and request logging middleware with timing and correlation IDs.

## Dependencies
- S2.1 (FastAPI app factory) — provides `create_app()` to register exception handlers and middleware

## Target Location
- `src/exceptions.py` — custom exception hierarchy
- `src/middlewares.py` — request logging middleware with timing + correlation IDs

## Functional Requirements

### FR-1: Custom Exception Hierarchy
- **What**: Define a hierarchy of exception classes organized by subsystem (repository, parsing, search, API, LLM, config, embeddings, cache)
- **Inputs**: N/A (class definitions)
- **Outputs**: Exception classes that carry `detail` message, optional `status_code` hint, and optional `context` dict
- **Edge cases**: All exceptions must be picklable (for multiprocessing); base class provides consistent `__str__`

#### Exception Hierarchy:
```
PaperAlchemyError (base)
├── RepositoryError (database layer)
│   ├── PaperNotFoundError (404)
│   └── PaperSaveError (500)
├── ParsingError (document parsing)
│   ├── PDFParsingError (PDF extraction failed)
│   └── PDFValidationError (invalid PDF file)
├── ExternalServiceError (external API calls)
│   ├── ArxivAPIError (arXiv API)
│   │   ├── ArxivTimeoutError
│   │   └── ArxivRateLimitError
│   ├── EmbeddingServiceError (Jina API)
│   ├── LLMServiceError (Ollama / Gemini)
│   │   ├── LLMConnectionError
│   │   └── LLMTimeoutError
│   ├── SearchEngineError (OpenSearch)
│   └── CacheServiceError (Redis)
├── PipelineError (ingestion pipeline)
└── ConfigurationError (invalid settings at startup)
```

### FR-2: Structured Error Response Schema
- **What**: Pydantic model for error responses returned by the API
- **Inputs**: Exception instance
- **Outputs**: JSON body: `{"error": {"type": str, "message": str, "request_id": str | None, "detail": dict | None}}`
- **Edge cases**: Never leak stack traces or internal paths in production (only in debug mode)

### FR-3: Global Exception Handlers
- **What**: FastAPI exception handlers registered via `create_app()` that catch custom exceptions and return structured JSON
- **Inputs**: Raised exception during request processing
- **Outputs**: `JSONResponse` with appropriate HTTP status code and `ErrorResponse` body
- **Mapping**:
  - `PaperNotFoundError` → 404
  - `PDFValidationError` → 422
  - `ArxivRateLimitError` → 429
  - `ExternalServiceError` subclasses → 503
  - `ConfigurationError` → 500
  - `PaperAlchemyError` (catch-all) → 500
  - Unhandled `Exception` → 500 (generic "Internal Server Error")
- **Edge cases**: Handler must log the full traceback at ERROR level; response must include `request_id` if available

### FR-4: Request Logging Middleware
- **What**: Starlette `BaseHTTPMiddleware` that logs every request with method, path, status code, duration, and correlation ID
- **Inputs**: Every incoming HTTP request
- **Outputs**: Log entry at INFO level; adds `X-Request-ID` response header
- **Behavior**:
  - Generate UUID4 `request_id` (or use incoming `X-Request-ID` header)
  - Store `request_id` in request state for access in exception handlers
  - Log: `{method} {path} → {status_code} ({duration_ms}ms) [request_id={id}]`
  - Errors: log at ERROR level with exception type
- **Edge cases**: Exclude `/api/v1/ping` and `/docs` from logging to avoid noise

### FR-5: Wire into App Factory
- **What**: Register exception handlers and middleware in `create_app()`
- **Inputs**: N/A
- **Outputs**: Updated `src/main.py` that includes error handlers + middleware
- **Edge cases**: Middleware order matters — logging middleware must be outermost

## Tangible Outcomes
- [ ] `src/exceptions.py` exists with full exception hierarchy (10+ exception classes)
- [ ] `src/schemas/api/error.py` exists with `ErrorResponse` Pydantic model
- [ ] `src/middlewares.py` exists with `RequestLoggingMiddleware` class
- [ ] Global exception handlers registered in `create_app()`
- [ ] `PaperNotFoundError` → 404 JSON response (tested)
- [ ] `ExternalServiceError` → 503 JSON response (tested)
- [ ] Unhandled exceptions → 500 JSON response with no stack trace leak (tested)
- [ ] Every response has `X-Request-ID` header (tested)
- [ ] Request duration logged in milliseconds (tested)
- [ ] Ping endpoint excluded from request logging (tested)
- [ ] All tests pass with `pytest`
- [ ] Notebook: `notebooks/specs/S2.3_error_handling.ipynb`

## Test-Driven Requirements

### Tests to Write First
1. `test_paper_alchemy_error_base`: PaperAlchemyError stores detail, status_code, context
2. `test_exception_hierarchy_inheritance`: All exceptions inherit correctly
3. `test_error_response_schema`: ErrorResponse serializes correctly
4. `test_not_found_returns_404`: Endpoint raising PaperNotFoundError returns 404 JSON
5. `test_external_service_returns_503`: Endpoint raising ExternalServiceError returns 503 JSON
6. `test_rate_limit_returns_429`: ArxivRateLimitError returns 429 JSON
7. `test_validation_returns_422`: PDFValidationError returns 422 JSON
8. `test_unhandled_returns_500`: Generic Exception returns 500 with no stack trace
9. `test_debug_mode_includes_traceback`: In debug mode, 500 response includes traceback
10. `test_request_id_header_generated`: Response includes X-Request-ID
11. `test_request_id_header_forwarded`: Existing X-Request-ID is preserved
12. `test_request_logging_format`: Logged message includes method, path, status, duration
13. `test_ping_excluded_from_logging`: /api/v1/ping requests not logged
14. `test_error_response_includes_request_id`: Error JSON includes request_id field

### Mocking Strategy
- Use `httpx.AsyncClient` + `ASGITransport` for testing FastAPI endpoints
- Create temporary test routes that raise specific exceptions
- Mock `logging.getLogger` to capture log output for assertions

### Coverage
- All public exception classes tested for instantiation and inheritance
- All HTTP status code mappings tested end-to-end
- Middleware tested for header injection and logging
- Debug vs production mode error detail tested
