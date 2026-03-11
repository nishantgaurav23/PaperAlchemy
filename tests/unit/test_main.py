"""Tests for S2.1 — FastAPI App Factory.

TDD: These tests are written BEFORE the implementation.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from src.config import Settings


@pytest.fixture
def app():
    """Create a fresh app instance for each test."""
    from src.main import create_app

    return create_app()


@pytest.fixture
async def client(app: FastAPI):
    """Async test client using httpx."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


class TestCreateApp:
    def test_create_app_returns_fastapi_instance(self, app: FastAPI):
        assert isinstance(app, FastAPI)

    def test_create_app_sets_title_and_version(self, app: FastAPI):
        assert app.title == "PaperAlchemy"
        assert app.version == "0.1.0"

    def test_create_app_with_custom_settings(self):
        from src.main import create_app

        custom = Settings(app={"title": "CustomApp", "version": "9.9.9", "debug": False})
        custom_app = create_app(settings_override=custom)
        assert custom_app.title == "CustomApp"
        assert custom_app.version == "9.9.9"


class TestPingEndpoint:
    @pytest.mark.asyncio
    async def test_ping_endpoint_returns_ok(self, client: AsyncClient):
        resp = await client.get("/api/v1/ping")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_ping_response_schema(self, client: AsyncClient):
        from src.schemas.api.health import PingResponse

        resp = await client.get("/api/v1/ping")
        data = resp.json()
        ping = PingResponse(**data)
        assert ping.status == "ok"
        assert ping.version == "0.1.0"


class TestCORS:
    @pytest.mark.asyncio
    async def test_cors_headers_present(self, client: AsyncClient):
        resp = await client.options(
            "/api/v1/ping",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" in resp.headers


class TestLifespan:
    @pytest.mark.asyncio
    async def test_lifespan_startup_shutdown(self, app: FastAPI):
        """Verify lifespan runs without errors by using the app in a client context."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/api/v1/ping")
            assert resp.status_code == 200


class TestRouting:
    @pytest.mark.asyncio
    async def test_404_for_unknown_routes(self, client: AsyncClient):
        resp = await client.get("/api/v1/nonexistent")
        assert resp.status_code == 404
