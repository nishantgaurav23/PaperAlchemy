"""Tests for S2.3 — Error Handlers & Request Logging Middleware.

TDD: These tests are written BEFORE the implementation.
"""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def app_with_error_routes():
    """Create an app with test routes that raise specific exceptions."""
    from src.main import create_app

    app = create_app()

    from src.exceptions import (
        ArxivRateLimitError,
        ExternalServiceError,
        PaperAlchemyError,
        PaperNotFoundError,
        PDFValidationError,
    )

    @app.get("/api/v1/test/not-found")
    async def raise_not_found():
        raise PaperNotFoundError("paper abc not found")

    @app.get("/api/v1/test/service-error")
    async def raise_service_error():
        raise ExternalServiceError("OpenSearch is down")

    @app.get("/api/v1/test/rate-limit")
    async def raise_rate_limit():
        raise ArxivRateLimitError("429 Too Many Requests")

    @app.get("/api/v1/test/validation")
    async def raise_validation():
        raise PDFValidationError("not a valid PDF")

    @app.get("/api/v1/test/generic-app-error")
    async def raise_generic_app():
        raise PaperAlchemyError("generic app error")

    @app.get("/api/v1/test/unhandled")
    async def raise_unhandled():
        raise RuntimeError("unexpected crash")

    return app


@pytest.fixture
async def client(app_with_error_routes: FastAPI):
    async with AsyncClient(transport=ASGITransport(app=app_with_error_routes), base_url="http://test") as ac:
        yield ac


class TestGlobalExceptionHandlers:
    @pytest.mark.asyncio
    async def test_not_found_returns_404(self, client: AsyncClient):
        resp = await client.get("/api/v1/test/not-found")
        assert resp.status_code == 404
        data = resp.json()
        assert data["error"]["type"] == "PaperNotFoundError"
        assert "paper abc not found" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_external_service_returns_503(self, client: AsyncClient):
        resp = await client.get("/api/v1/test/service-error")
        assert resp.status_code == 503
        data = resp.json()
        assert data["error"]["type"] == "ExternalServiceError"

    @pytest.mark.asyncio
    async def test_rate_limit_returns_429(self, client: AsyncClient):
        resp = await client.get("/api/v1/test/rate-limit")
        assert resp.status_code == 429
        data = resp.json()
        assert data["error"]["type"] == "ArxivRateLimitError"

    @pytest.mark.asyncio
    async def test_validation_returns_422(self, client: AsyncClient):
        resp = await client.get("/api/v1/test/validation")
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["type"] == "PDFValidationError"

    @pytest.mark.asyncio
    async def test_generic_app_error_returns_500(self, client: AsyncClient):
        resp = await client.get("/api/v1/test/generic-app-error")
        assert resp.status_code == 500
        data = resp.json()
        assert data["error"]["type"] == "PaperAlchemyError"

    @pytest.mark.asyncio
    async def test_unhandled_returns_500_no_stack_trace(self, client: AsyncClient):
        resp = await client.get("/api/v1/test/unhandled")
        assert resp.status_code == 500
        data = resp.json()
        assert data["error"]["type"] == "InternalServerError"
        assert "Internal Server Error" in data["error"]["message"]
        # Must NOT leak stack trace in production mode
        assert "Traceback" not in data["error"]["message"]
        assert "RuntimeError" not in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_error_response_includes_request_id(self, client: AsyncClient):
        resp = await client.get("/api/v1/test/not-found")
        data = resp.json()
        assert data["error"]["request_id"] is not None
        assert len(data["error"]["request_id"]) > 0


class TestDebugModeErrors:
    @pytest.mark.asyncio
    async def test_debug_mode_includes_traceback(self):
        """In debug mode, unhandled 500 errors include traceback detail."""
        from src.config import Settings
        from src.main import create_app

        app = create_app(settings_override=Settings(app={"title": "Test", "version": "0.0.1", "debug": True}))

        @app.get("/api/v1/test/crash")
        async def crash():
            raise RuntimeError("debug crash")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/v1/test/crash")
            assert resp.status_code == 500
            data = resp.json()
            assert data["error"]["detail"] is not None
            assert "traceback" in data["error"]["detail"]


class TestRequestLoggingMiddleware:
    @pytest.mark.asyncio
    async def test_request_id_header_generated(self, client: AsyncClient):
        resp = await client.get("/api/v1/ping")
        assert "x-request-id" in resp.headers
        # UUID4 format: 8-4-4-4-12
        req_id = resp.headers["x-request-id"]
        assert len(req_id) == 36

    @pytest.mark.asyncio
    async def test_request_id_header_forwarded(self, client: AsyncClient):
        custom_id = "my-custom-request-id-123"
        resp = await client.get("/api/v1/ping", headers={"X-Request-ID": custom_id})
        assert resp.headers["x-request-id"] == custom_id

    @pytest.mark.asyncio
    async def test_request_logging_format(self, client: AsyncClient, caplog):
        with caplog.at_level(logging.INFO, logger="src.middlewares"):
            await client.get("/api/v1/test/not-found")
        # Filter for INFO-level request log (not ERROR-level exception log)
        log_messages = [r.message for r in caplog.records if "src.middlewares" in r.name and r.levelno == logging.INFO]
        assert len(log_messages) >= 1
        log_msg = log_messages[0]
        assert "GET" in log_msg
        assert "/api/v1/test/not-found" in log_msg
        assert "404" in log_msg
        assert "ms" in log_msg

    @pytest.mark.asyncio
    async def test_ping_excluded_from_logging(self, client: AsyncClient, caplog):
        with caplog.at_level(logging.INFO, logger="src.middlewares"):
            await client.get("/api/v1/ping")
        log_messages = [r.message for r in caplog.records if "src.middlewares" in r.name]
        # Ping should be excluded
        ping_logs = [m for m in log_messages if "/api/v1/ping" in m]
        assert len(ping_logs) == 0
