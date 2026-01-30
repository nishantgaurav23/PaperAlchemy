"""
PaperAlchemy Custom Exception Hierarchy.

Why it's needed:
    Without custom exceptions, every failure raises generic Exception, making
    it impossible to distinguish between "paper not found" (return 404) and
    "OpenSearch is down" (return 503). Custom exceptions enable precise error
    handling at every layer of the application.

How it helps:
    - Routers catch specific exceptions → return correct HTTP status codes
    - Services catch and re-raise domain exceptions → clean error propagation
    - Pipeline code retries on transient errors (timeout, rate limit)
    - Startup code fails fast on ConfigurationError before serving requests
    - Logging includes exception class name → instant failure identification

Architecture:
    Each subsystem gets its own exception tree. Child exceptions inherit from
    parent, so you can catch broadly (ArxivAPIException) or narrowly
    (ArxivAPIRateLimitError) depending on the handler's needs.

Hierarchy:
    Exception
    ├── RepositoryException        ─ database layer (CRUD failures)
    │   ├── PaperNotFound          ─ SELECT returned no rows → HTTP 404
    │   └── PaperNotSaved          ─ INSERT/UPDATE failed → HTTP 500
    ├── ParsingException           ─ document parsing layer
    │   └── PDFParsingException    ─ Docling extraction failed
    │       └── PDFValidationError ─ file isn't a valid PDF
    ├── PDFDownloadException       ─ HTTP download from arXiv
    │   └── PDFDownloadTimeoutError─ download exceeded timeout
    ├── PDFCacheException          ─ local PDF file cache errors
    ├── OpenSearchException        ─ search engine layer → HTTP 503
    ├── ArxivAPIException          ─ arXiv external API
    │   ├── ArxivAPITimeoutError   ─ request timed out → retry with backoff
    │   ├── ArxivAPIRateLimitError ─ HTTP 429 → wait 3+ seconds
    │   └── ArxivParseError        ─ malformed XML response
    ├── MetadataFetchingException  ─ ingestion pipeline
    │   └── PipelineException      ─ DAG/step failure
    ├── LLMException               ─ language model layer
    │   └── OllamaException        ─ Ollama-specific errors
    │       ├── OllamaConnectionError ─ can't reach Ollama → HTTP 503
    │       └── OllamaTimeoutError    ─ generation timed out → retry/truncate
    └── ConfigurationError         ─ invalid settings → fail at startup
"""


# =============================================================
# Database / Repository Exceptions
# =============================================================
# These exceptions are raised by the repository layer (src/repositories/)
# when CRUD operations on the papers table fail. The router layer
# catches these to return appropriate HTTP responses.

class RepositoryException(Exception):
    """Base exception for all database repository operations.

    Raised when CRUD operations on the papers table fail.
    Routers should catch this to return 500 Internal Server Error.
    """


class PaperNotFound(RepositoryException):
    """Raised when a SELECT query returns no matching paper.

    Typically caught in routers to return HTTP 404.
    Example: GET /papers/{arxiv_id} where arxiv_id doesn't exist.
    """


class PaperNotSaved(RepositoryException):
    """Raised when an INSERT or UPDATE fails to persist a paper.

    Could happen due to unique constraint violations (duplicate arxiv_id),
    connection drops, or transaction rollbacks.
    """


# =============================================================
# PDF Parsing Exceptions
# =============================================================
# These exceptions are raised by the PDF parsing service
# (src/services/pdf_parser/). The ingestion pipeline catches these
# to mark papers as 'failed' and continue processing the batch.

class ParsingException(Exception):
    """Base exception for all document parsing operations.

    Covers PDF, HTML, or any future document format parsing.
    """


class PDFParsingException(ParsingException):
    """Raised when Docling fails to extract content from a PDF.

    Common causes: corrupted PDF, unsupported encoding, timeout.
    The ingestion pipeline catches this to mark papers as 'failed'
    and continue processing remaining papers in the batch.
    """


class PDFValidationError(PDFParsingException):
    """Raised when a file fails PDF format validation before parsing.

    Checks: file size > 0, valid PDF header (%PDF-), not encrypted.
    Failing validation skips the expensive Docling parsing step entirely,
    saving compute time on obviously invalid files.
    """


# =============================================================
# PDF Download Exceptions
# =============================================================
# These exceptions are raised when downloading PDFs from arXiv.
# The pipeline retries with exponential backoff on timeouts.

class PDFDownloadException(Exception):
    """Raised when downloading a PDF from arXiv fails.

    Covers HTTP errors (403, 500), network timeouts, and SSL errors.
    """


class PDFDownloadTimeoutError(PDFDownloadException):
    """Raised when a PDF download exceeds the configured timeout.

    Default timeout is set in config. The pipeline can retry with
    exponential backoff when this occurs.
    """


# =============================================================
# PDF Cache Exceptions
# =============================================================
# PDF caching stores downloaded PDFs locally to avoid re-downloading.
# Cache failures are non-fatal — the pipeline falls back to re-downloading.

class PDFCacheException(Exception):
    """Raised for local PDF file cache errors.

    Covers: disk full, permission denied, corrupted cached file.
    Non-fatal — the pipeline falls back to re-downloading from arXiv.
    """


# =============================================================
# OpenSearch Exceptions
# =============================================================
# These exceptions are raised by the OpenSearch client
# (src/services/opensearch/client.py) when search operations fail.
# Routers catch these to return 503 Service Unavailable.

class OpenSearchException(Exception):
    """Raised for OpenSearch client and indexing errors.

    Covers: connection refused, index not found, mapping conflicts,
    bulk indexing failures, and query syntax errors.
    Routers catch this to return 503 Service Unavailable.
    """


# =============================================================
# ArXiv API Exceptions
# =============================================================
# These exceptions model the different failure modes of the arXiv API.
# Each failure mode has a different recovery strategy:
#   - Timeout → retry with exponential backoff
#   - Rate limit → wait 3+ seconds before retry
#   - Parse error → log and skip the malformed entry

class ArxivAPIException(Exception):
    """Base exception for all arXiv API interactions.

    The arXiv API has strict rate limits (1 request per 3 seconds)
    and can return malformed XML. These exceptions help the pipeline
    handle each failure mode differently.
    """


class ArxivAPITimeoutError(ArxivAPIException):
    """Raised when an arXiv API request exceeds the timeout.

    The pipeline retries with exponential backoff (3s, 6s, 12s).
    """


class ArxivAPIRateLimitError(ArxivAPIException):
    """Raised when arXiv returns HTTP 429 (Too Many Requests).

    The pipeline must wait before retrying. arXiv recommends
    a minimum 3-second delay between requests.
    """


class ArxivParseError(ArxivAPIException):
    """Raised when the arXiv API response XML cannot be parsed.

    Happens when arXiv returns HTML error pages instead of Atom XML,
    or when the feed structure changes unexpectedly.
    """


# =============================================================
# Metadata / Pipeline Exceptions
# =============================================================
# These exceptions cover the full ingestion flow managed by
# MetadataFetcher: arXiv fetch → PDF download → parse → store.

class MetadataFetchingException(Exception):
    """Raised during the metadata fetching pipeline.

    Covers the full ingestion flow: arXiv fetch → parse → store.
    """


class PipelineException(MetadataFetchingException):
    """Raised when an Airflow DAG step or pipeline stage fails.

    Includes context about which step failed and partial results
    so the pipeline can resume from the last successful step.
    """


# =============================================================
# LLM / Ollama Exceptions
# =============================================================
# These exceptions are raised by the Ollama client
# (src/services/ollama/client.py) during LLM inference.
# The router can retry with shorter context or return a fallback response.

class LLMException(Exception):
    """Base exception for all language model operations.

    Covers local (Ollama) and future remote (OpenAI) LLM providers.
    """


class OllamaException(LLMException):
    """Raised for Ollama-specific service errors.

    Covers: model not loaded, inference errors, malformed responses.
    """


class OllamaConnectionError(OllamaException):
    """Raised when the Ollama service is unreachable.

    Common causes: Docker container not running, wrong port,
    network partition. The router returns 503 with a helpful message.
    """


class OllamaTimeoutError(OllamaException):
    """Raised when Ollama generation exceeds the timeout.

    Large prompts or complex queries can cause this. The router
    can retry with a shorter context or return a partial response.
    """


# =============================================================
# Configuration Exceptions
# =============================================================
# Raised at startup when settings are invalid. Causes immediate
# application shutdown before any requests are served.

class ConfigurationError(Exception):
    """Raised when application settings are invalid at startup.

    Examples: missing required env vars, invalid database URL,
    OpenSearch host unreachable. Causes fast failure before
    serving any requests — better to crash early than serve
    broken responses.
    """
