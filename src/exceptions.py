"""PaperAlchemy custom exception hierarchy.

Organized by subsystem so each service layer has its own exception tree.
This enables precise error handling in routers and middleware:
- RepositoryError       -> 404/500 responses
- ParsingError          -> 422 for validation, 500 for parse failures
- ExternalServiceError  -> 503 (service unavailable)
- ArxivRateLimitError   -> 429 (too many requests)
- ConfigurationError    -> 500 (fail-fast on startup)
"""

from __future__ import annotations


class PaperAlchemyError(Exception):
    """Base exception for all PaperAlchemy errors."""

    def __init__(
        self,
        detail: str = "An unexpected error occurred",
        *,
        status_code: int = 500,
        context: dict | None = None,
    ) -> None:
        self.detail = detail
        self.status_code = status_code
        self.context = context
        super().__init__(detail)


# -- Repository (database layer) --


class RepositoryError(PaperAlchemyError):
    """Base exception for database repository operations."""

    def __init__(self, detail: str = "Database operation failed", **kwargs) -> None:
        super().__init__(detail, **kwargs)


class PaperNotFoundError(RepositoryError):
    """Raised when a SELECT query returns no matching paper."""

    def __init__(self, detail: str = "Paper not found", **kwargs) -> None:
        kwargs.setdefault("status_code", 404)
        super().__init__(detail, **kwargs)


class PaperSaveError(RepositoryError):
    """Raised when an INSERT or UPDATE fails to persist a paper."""

    def __init__(self, detail: str = "Failed to save paper", **kwargs) -> None:
        super().__init__(detail, **kwargs)


# -- Parsing (document parsing layer) --


class ParsingError(PaperAlchemyError):
    """Base exception for document parsing operations."""

    def __init__(self, detail: str = "Document parsing failed", **kwargs) -> None:
        super().__init__(detail, **kwargs)


class PDFParsingError(ParsingError):
    """Raised when PDF content extraction fails."""

    def __init__(self, detail: str = "PDF parsing failed", **kwargs) -> None:
        super().__init__(detail, **kwargs)


class PDFValidationError(PDFParsingError):
    """Raised when a file fails PDF format validation before parsing."""

    def __init__(self, detail: str = "Invalid PDF file", **kwargs) -> None:
        kwargs.setdefault("status_code", 422)
        super().__init__(detail, **kwargs)


# -- External Services --


class ExternalServiceError(PaperAlchemyError):
    """Base exception for external service failures."""

    def __init__(self, detail: str = "External service unavailable", **kwargs) -> None:
        kwargs.setdefault("status_code", 503)
        super().__init__(detail, **kwargs)


class ArxivAPIError(ExternalServiceError):
    """Base exception for arXiv API interactions."""

    def __init__(self, detail: str = "arXiv API error", **kwargs) -> None:
        super().__init__(detail, **kwargs)


class ArxivTimeoutError(ArxivAPIError):
    """Raised when an arXiv API request exceeds the timeout."""

    def __init__(self, detail: str = "arXiv API request timed out", **kwargs) -> None:
        super().__init__(detail, **kwargs)


class ArxivRateLimitError(ArxivAPIError):
    """Raised when arXiv returns HTTP 429 (Too Many Requests)."""

    def __init__(self, detail: str = "arXiv rate limit exceeded", **kwargs) -> None:
        kwargs.setdefault("status_code", 429)
        super().__init__(detail, **kwargs)


class EmbeddingServiceError(ExternalServiceError):
    """Raised for Jina embedding API errors."""

    def __init__(self, detail: str = "Embedding service error", **kwargs) -> None:
        super().__init__(detail, **kwargs)


class LLMServiceError(ExternalServiceError):
    """Base exception for LLM provider errors (Ollama, Gemini)."""

    def __init__(self, detail: str = "LLM service error", **kwargs) -> None:
        super().__init__(detail, **kwargs)


class LLMConnectionError(LLMServiceError):
    """Raised when the LLM service is unreachable."""

    def __init__(self, detail: str = "LLM service unreachable", **kwargs) -> None:
        super().__init__(detail, **kwargs)


class LLMTimeoutError(LLMServiceError):
    """Raised when LLM generation exceeds the timeout."""

    def __init__(self, detail: str = "LLM generation timed out", **kwargs) -> None:
        super().__init__(detail, **kwargs)


class RerankerError(ExternalServiceError):
    """Raised for cross-encoder re-ranking errors."""

    def __init__(self, detail: str = "Re-ranker error", **kwargs) -> None:
        super().__init__(detail, **kwargs)


class SearchEngineError(ExternalServiceError):
    """Raised for OpenSearch client and indexing errors."""

    def __init__(self, detail: str = "Search engine error", **kwargs) -> None:
        super().__init__(detail, **kwargs)


class CacheServiceError(ExternalServiceError):
    """Raised for Redis cache errors."""

    def __init__(self, detail: str = "Cache service error", **kwargs) -> None:
        super().__init__(detail, **kwargs)


# -- Pipeline --


class PipelineError(PaperAlchemyError):
    """Raised when an ingestion pipeline step fails."""

    def __init__(self, detail: str = "Pipeline step failed", **kwargs) -> None:
        super().__init__(detail, **kwargs)


# -- Configuration --


class ConfigurationError(PaperAlchemyError):
    """Raised when application settings are invalid at startup."""

    def __init__(self, detail: str = "Invalid configuration", **kwargs) -> None:
        super().__init__(detail, **kwargs)
