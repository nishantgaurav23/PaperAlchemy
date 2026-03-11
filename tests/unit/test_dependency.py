"""Tests for S2.4 — Dependency Injection.

TDD: These tests are written BEFORE the implementation.
Note: Do NOT use `from __future__ import annotations` here — it breaks
FastAPI's runtime inspection of Annotated[] type aliases.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from src.config import Settings  # noqa: F401
from src.db.database import Database

# ---------------------------------------------------------------------------
# FR-1: Settings Dependency
# ---------------------------------------------------------------------------


class TestGetSettings:
    def test_get_settings_returns_settings_instance(self):
        """get_settings() returns a Settings object."""
        from src.dependency import get_settings

        result = get_settings()
        assert isinstance(result, Settings)

    def test_get_settings_returns_same_instance(self):
        """get_settings() returns the cached singleton (via lru_cache in config)."""
        from src.dependency import get_settings

        a = get_settings()
        b = get_settings()
        assert a is b


# ---------------------------------------------------------------------------
# FR-2: Database Dependency
# ---------------------------------------------------------------------------


class TestGetDatabase:
    def test_get_database_returns_database_when_initialized(self):
        """get_database() returns Database after init_database() was called."""
        from src.dependency import get_database

        mock_db = MagicMock(spec=Database)
        with patch("src.dependency._get_database", return_value=mock_db):
            result = get_database()
            assert result is mock_db

    def test_get_database_raises_when_not_initialized(self):
        """get_database() raises RuntimeError if database not initialized."""
        from src.dependency import get_database

        with (
            patch("src.dependency._get_database", side_effect=RuntimeError("Database not initialised")),
            pytest.raises(RuntimeError, match="Database not initialised"),
        ):
            get_database()


# ---------------------------------------------------------------------------
# FR-3: Async Session Dependency
# ---------------------------------------------------------------------------


class TestGetDbSession:
    @pytest.mark.asyncio
    async def test_get_db_session_yields_async_session(self):
        """get_db_session() yields an AsyncSession."""
        from src.dependency import get_db_session

        mock_session = AsyncMock(spec=AsyncSession)
        mock_db = MagicMock(spec=Database)

        # get_session() returns an async context manager
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_db.get_session.return_value = mock_ctx

        with patch("src.db._get_database", return_value=mock_db):
            sessions = []
            async for session in get_db_session():
                sessions.append(session)

            assert len(sessions) == 1
            assert sessions[0] is mock_session

    @pytest.mark.asyncio
    async def test_get_db_session_raises_when_db_not_initialized(self):
        """get_db_session() propagates RuntimeError from uninitialized DB."""
        from src.dependency import get_db_session

        with (
            patch("src.db._get_database", side_effect=RuntimeError("Database not initialised")),
            pytest.raises(RuntimeError, match="Database not initialised"),
        ):
            async for _ in get_db_session():
                pass


# ---------------------------------------------------------------------------
# FR-4: Annotated Type Aliases
# ---------------------------------------------------------------------------


class TestAnnotatedAliases:
    def test_settings_dep_importable(self):
        """SettingsDep is importable and is an Annotated type."""
        from src.dependency import SettingsDep

        assert hasattr(SettingsDep, "__metadata__")

    def test_database_dep_importable(self):
        """DatabaseDep is importable and is an Annotated type."""
        from src.dependency import DatabaseDep

        assert hasattr(DatabaseDep, "__metadata__")

    def test_session_dep_importable(self):
        """SessionDep is importable and is an Annotated type."""
        from src.dependency import SessionDep

        assert hasattr(SessionDep, "__metadata__")

    def test_all_aliases_in_module_exports(self):
        """All *Dep aliases are listed in __all__."""
        import src.dependency as dep_module

        for name in ("SettingsDep", "DatabaseDep", "SessionDep"):
            assert name in dep_module.__all__, f"{name} not in __all__"

    def test_getter_functions_in_module_exports(self):
        """All getter functions are listed in __all__."""
        import src.dependency as dep_module

        for name in ("get_settings", "get_database", "get_db_session"):
            assert name in dep_module.__all__, f"{name} not in __all__"


# ---------------------------------------------------------------------------
# FR-5: Dependency Overrides (Integration)
# ---------------------------------------------------------------------------


class TestDependencyOverrides:
    @pytest.fixture
    def test_app(self):
        """Create a minimal app with a test route using dependencies."""
        from src.dependency import SessionDep, SettingsDep

        app = FastAPI()

        @app.get("/test-settings")
        async def test_settings_route(settings: SettingsDep):
            return {"title": settings.app.title}

        @app.get("/test-session")
        async def test_session_route(session: SessionDep):
            return {"session_type": type(session).__name__}

        return app

    @pytest.mark.asyncio
    async def test_override_settings_dependency(self, test_app: FastAPI):
        """app.dependency_overrides replaces get_settings with mock."""
        from src.dependency import get_settings

        mock_settings = MagicMock()
        mock_settings.app.title = "OverriddenApp"

        test_app.dependency_overrides[get_settings] = lambda: mock_settings

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
            resp = await ac.get("/test-settings")
            assert resp.status_code == 200
            assert resp.json()["title"] == "OverriddenApp"

        test_app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_override_session_dependency(self, test_app: FastAPI):
        """app.dependency_overrides replaces get_db_session with mock session."""
        from src.dependency import get_db_session

        mock_session = AsyncMock(spec=AsyncSession)

        async def mock_get_db_session():
            yield mock_session

        test_app.dependency_overrides[get_db_session] = mock_get_db_session

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
            resp = await ac.get("/test-session")
            assert resp.status_code == 200
            assert resp.json()["session_type"] == "AsyncMock"

        test_app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Integration: Router with injected dependencies
# ---------------------------------------------------------------------------


class TestRouterIntegration:
    @pytest.mark.asyncio
    async def test_router_with_settings_dep(self):
        """Full integration: a router endpoint receives SettingsDep correctly."""
        from src.dependency import SettingsDep, get_settings

        app = FastAPI()

        @app.get("/info")
        async def info_route(settings: SettingsDep):
            return {"title": settings.app.title, "version": settings.app.version}

        mock_settings = MagicMock()
        mock_settings.app.title = "TestApp"
        mock_settings.app.version = "1.0.0"
        app.dependency_overrides[get_settings] = lambda: mock_settings

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/info")
            assert resp.status_code == 200
            data = resp.json()
            assert data["title"] == "TestApp"
            assert data["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_router_with_database_dep(self):
        """Full integration: a router endpoint receives DatabaseDep correctly."""
        from src.dependency import DatabaseDep, get_database

        app = FastAPI()

        @app.get("/db-check")
        async def db_check_route(db: DatabaseDep):
            return {"db_type": type(db).__name__}

        mock_db = MagicMock(spec=Database)
        app.dependency_overrides[get_database] = lambda: mock_db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/db-check")
            assert resp.status_code == 200
            assert resp.json()["db_type"] == "MagicMock"
