"""Tests for S2.3 — Custom Exception Hierarchy.

TDD: These tests are written BEFORE the implementation.
"""

from __future__ import annotations


class TestPaperAlchemyErrorBase:
    def test_base_error_stores_message(self):
        from src.exceptions import PaperAlchemyError

        err = PaperAlchemyError("something broke")
        assert str(err) == "something broke"
        assert err.detail == "something broke"

    def test_base_error_default_status_code(self):
        from src.exceptions import PaperAlchemyError

        err = PaperAlchemyError("fail")
        assert err.status_code == 500

    def test_base_error_custom_status_code(self):
        from src.exceptions import PaperAlchemyError

        err = PaperAlchemyError("not found", status_code=404)
        assert err.status_code == 404

    def test_base_error_optional_context(self):
        from src.exceptions import PaperAlchemyError

        err = PaperAlchemyError("fail", context={"paper_id": "123"})
        assert err.context == {"paper_id": "123"}

    def test_base_error_context_defaults_to_none(self):
        from src.exceptions import PaperAlchemyError

        err = PaperAlchemyError("fail")
        assert err.context is None


class TestExceptionHierarchyInheritance:
    def test_repository_error_inherits(self):
        from src.exceptions import PaperAlchemyError, RepositoryError

        assert issubclass(RepositoryError, PaperAlchemyError)

    def test_paper_not_found_inherits(self):
        from src.exceptions import PaperNotFoundError, RepositoryError

        assert issubclass(PaperNotFoundError, RepositoryError)
        err = PaperNotFoundError("paper xyz")
        assert err.status_code == 404

    def test_paper_save_error_inherits(self):
        from src.exceptions import PaperSaveError, RepositoryError

        assert issubclass(PaperSaveError, RepositoryError)

    def test_parsing_error_inherits(self):
        from src.exceptions import PaperAlchemyError, ParsingError

        assert issubclass(ParsingError, PaperAlchemyError)

    def test_pdf_parsing_error_inherits(self):
        from src.exceptions import ParsingError, PDFParsingError

        assert issubclass(PDFParsingError, ParsingError)

    def test_pdf_validation_error_inherits(self):
        from src.exceptions import PDFParsingError, PDFValidationError

        assert issubclass(PDFValidationError, PDFParsingError)
        err = PDFValidationError("bad pdf")
        assert err.status_code == 422

    def test_external_service_error_inherits(self):
        from src.exceptions import ExternalServiceError, PaperAlchemyError

        assert issubclass(ExternalServiceError, PaperAlchemyError)
        err = ExternalServiceError("service down")
        assert err.status_code == 503

    def test_arxiv_errors_inherit(self):
        from src.exceptions import ArxivAPIError, ArxivRateLimitError, ArxivTimeoutError, ExternalServiceError

        assert issubclass(ArxivAPIError, ExternalServiceError)
        assert issubclass(ArxivTimeoutError, ArxivAPIError)
        assert issubclass(ArxivRateLimitError, ArxivAPIError)
        err = ArxivRateLimitError("429")
        assert err.status_code == 429

    def test_embedding_service_error_inherits(self):
        from src.exceptions import EmbeddingServiceError, ExternalServiceError

        assert issubclass(EmbeddingServiceError, ExternalServiceError)

    def test_llm_errors_inherit(self):
        from src.exceptions import ExternalServiceError, LLMConnectionError, LLMServiceError, LLMTimeoutError

        assert issubclass(LLMServiceError, ExternalServiceError)
        assert issubclass(LLMConnectionError, LLMServiceError)
        assert issubclass(LLMTimeoutError, LLMServiceError)

    def test_search_engine_error_inherits(self):
        from src.exceptions import ExternalServiceError, SearchEngineError

        assert issubclass(SearchEngineError, ExternalServiceError)

    def test_cache_service_error_inherits(self):
        from src.exceptions import CacheServiceError, ExternalServiceError

        assert issubclass(CacheServiceError, ExternalServiceError)

    def test_pipeline_error_inherits(self):
        from src.exceptions import PaperAlchemyError, PipelineError

        assert issubclass(PipelineError, PaperAlchemyError)

    def test_configuration_error_inherits(self):
        from src.exceptions import ConfigurationError, PaperAlchemyError

        assert issubclass(ConfigurationError, PaperAlchemyError)


class TestErrorResponseSchema:
    def test_error_response_serializes(self):
        from src.schemas.api.error import ErrorDetail, ErrorResponse

        resp = ErrorResponse(error=ErrorDetail(type="PaperNotFoundError", message="not found"))
        data = resp.model_dump()
        assert data["error"]["type"] == "PaperNotFoundError"
        assert data["error"]["message"] == "not found"
        assert data["error"]["request_id"] is None
        assert data["error"]["detail"] is None

    def test_error_response_with_request_id(self):
        from src.schemas.api.error import ErrorDetail, ErrorResponse

        resp = ErrorResponse(error=ErrorDetail(type="Error", message="fail", request_id="abc-123"))
        assert resp.error.request_id == "abc-123"

    def test_error_response_with_detail(self):
        from src.schemas.api.error import ErrorDetail, ErrorResponse

        resp = ErrorResponse(error=ErrorDetail(type="Error", message="fail", detail={"key": "val"}))
        assert resp.error.detail == {"key": "val"}
