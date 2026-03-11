"""Tests for S2.2 — Database Layer (async SQLAlchemy)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from src.config import PostgresSettings
from src.db.base import Base
from src.db.database import Database

# ---------------------------------------------------------------------------
# FR-1: Declarative Base
# ---------------------------------------------------------------------------


class TestDeclarativeBase:
    def test_base_is_importable(self):
        """Base can be imported without side effects (no engine created)."""
        assert Base is not None
        assert hasattr(Base, "metadata")
        assert hasattr(Base, "__subclasses__")

    def test_base_metadata_exists(self):
        """Base.metadata is a MetaData instance usable for table creation."""
        from sqlalchemy import MetaData

        assert isinstance(Base.metadata, MetaData)


# ---------------------------------------------------------------------------
# FR-2: Async Database Engine
# ---------------------------------------------------------------------------


class TestDatabaseCreation:
    def test_database_stores_url(self):
        """Database object stores the provided URL."""
        db = Database(database_url="postgresql+asyncpg://user:pass@localhost:5432/testdb")
        assert db.database_url == "postgresql+asyncpg://user:pass@localhost:5432/testdb"

    def test_database_creates_async_engine(self):
        """Database creates an AsyncEngine instance."""
        db = Database(database_url="postgresql+asyncpg://user:pass@localhost:5432/testdb")
        assert isinstance(db.engine, AsyncEngine)

    def test_database_engine_echo_off_by_default(self):
        """Engine echo is off by default."""
        db = Database(database_url="postgresql+asyncpg://user:pass@localhost:5432/testdb")
        assert db.engine.echo is False

    def test_database_engine_echo_on(self):
        """Engine echo can be turned on."""
        db = Database(database_url="postgresql+asyncpg://user:pass@localhost:5432/testdb", echo=True)
        assert db.engine.echo is True

    def test_database_pool_settings(self):
        """Engine has correct pool configuration."""
        db = Database(database_url="postgresql+asyncpg://user:pass@localhost:5432/testdb")
        pool = db.engine.pool
        assert pool.size() == 5
        assert pool._max_overflow == 10
        assert pool._recycle == 3600

    def test_database_url_from_settings(self):
        """Database URL is correctly constructed from PostgresSettings."""
        settings = PostgresSettings(host="db-host", port=5432, user="admin", password="secret", db="mydb")
        assert settings.url == "postgresql+asyncpg://admin:secret@db-host:5432/mydb"


# ---------------------------------------------------------------------------
# FR-3: Async Session Factory
# ---------------------------------------------------------------------------


class TestAsyncSessionLifecycle:
    async def test_session_yields_async_session(self):
        """get_session() yields an AsyncSession."""
        db = Database(database_url="postgresql+asyncpg://user:pass@localhost:5432/testdb")
        mock_session = AsyncMock(spec=AsyncSession)
        with patch.object(db, "_session_factory", return_value=mock_session):
            async with db.get_session() as session:
                assert session is mock_session

    async def test_session_commits_on_success(self):
        """Session is committed when context exits normally."""
        db = Database(database_url="postgresql+asyncpg://user:pass@localhost:5432/testdb")
        mock_session = AsyncMock(spec=AsyncSession)
        with patch.object(db, "_session_factory", return_value=mock_session):
            async with db.get_session() as _session:
                pass  # successful exit
            mock_session.commit.assert_awaited_once()

    async def test_session_rollbacks_on_exception(self):
        """Session is rolled back when an exception occurs."""
        db = Database(database_url="postgresql+asyncpg://user:pass@localhost:5432/testdb")
        mock_session = AsyncMock(spec=AsyncSession)
        with patch.object(db, "_session_factory", return_value=mock_session):
            with pytest.raises(SQLAlchemyError):
                async with db.get_session() as _session:
                    raise SQLAlchemyError("test error")
            mock_session.rollback.assert_awaited_once()
            mock_session.commit.assert_not_awaited()

    async def test_session_closes_always(self):
        """Session is closed regardless of success or failure."""
        db = Database(database_url="postgresql+asyncpg://user:pass@localhost:5432/testdb")
        mock_session = AsyncMock(spec=AsyncSession)
        with patch.object(db, "_session_factory", return_value=mock_session):
            async with db.get_session() as _session:
                pass
            mock_session.close.assert_awaited_once()

    async def test_session_closes_on_error(self):
        """Session is closed even when an error occurs."""
        db = Database(database_url="postgresql+asyncpg://user:pass@localhost:5432/testdb")
        mock_session = AsyncMock(spec=AsyncSession)
        with patch.object(db, "_session_factory", return_value=mock_session):
            with pytest.raises(SQLAlchemyError):
                async with db.get_session() as _session:
                    raise SQLAlchemyError("test error")
            mock_session.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# FR-4: Health Check
# ---------------------------------------------------------------------------


class _AsyncCtx:
    """Helper to build an async context manager returning a mock."""

    def __init__(self, mock_obj):
        self._mock = mock_obj

    async def __aenter__(self):
        return self._mock

    async def __aexit__(self, *args):
        pass


class TestHealthCheck:
    async def test_health_check_success(self):
        """health_check() returns True when DB is reachable."""
        mock_conn = AsyncMock()
        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(return_value=_AsyncCtx(mock_conn))
        mock_engine.dispose = AsyncMock()

        with patch("src.db.database.create_async_engine", return_value=mock_engine):
            db = Database(database_url="postgresql+asyncpg://user:pass@localhost:5432/testdb")
            result = await db.health_check()
        assert result is True
        mock_conn.execute.assert_awaited_once()

    async def test_health_check_failure(self):
        """health_check() returns False when DB is unreachable."""
        mock_engine = MagicMock()
        mock_engine.connect = MagicMock(side_effect=Exception("connection refused"))
        mock_engine.dispose = AsyncMock()

        with patch("src.db.database.create_async_engine", return_value=mock_engine):
            db = Database(database_url="postgresql+asyncpg://user:pass@localhost:5432/testdb")
            result = await db.health_check()
        assert result is False


# ---------------------------------------------------------------------------
# FR-5: Table Management
# ---------------------------------------------------------------------------


class TestTableManagement:
    async def test_create_tables(self):
        """create_tables() runs Base.metadata.create_all via run_sync."""
        mock_conn = AsyncMock()
        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=_AsyncCtx(mock_conn))
        mock_engine.dispose = AsyncMock()

        with patch("src.db.database.create_async_engine", return_value=mock_engine):
            db = Database(database_url="postgresql+asyncpg://user:pass@localhost:5432/testdb")
            await db.create_tables()
        mock_conn.run_sync.assert_awaited_once()
        func = mock_conn.run_sync.call_args[0][0]
        assert func.__name__ == "create_all"

    async def test_drop_tables(self):
        """drop_tables() runs Base.metadata.drop_all via run_sync."""
        mock_conn = AsyncMock()
        mock_engine = MagicMock()
        mock_engine.begin = MagicMock(return_value=_AsyncCtx(mock_conn))
        mock_engine.dispose = AsyncMock()

        with patch("src.db.database.create_async_engine", return_value=mock_engine):
            db = Database(database_url="postgresql+asyncpg://user:pass@localhost:5432/testdb")
            await db.drop_tables()
        mock_conn.run_sync.assert_awaited_once()
        func = mock_conn.run_sync.call_args[0][0]
        assert func.__name__ == "drop_all"


# ---------------------------------------------------------------------------
# FR-6: Engine Disposal
# ---------------------------------------------------------------------------


class TestEngineDisposal:
    async def test_close_disposes_engine(self):
        """close() disposes the engine."""
        mock_engine = AsyncMock()

        with patch("src.db.database.create_async_engine", return_value=mock_engine):
            db = Database(database_url="postgresql+asyncpg://user:pass@localhost:5432/testdb")
            await db.close()
        mock_engine.dispose.assert_awaited_once()


# ---------------------------------------------------------------------------
# FR-7: FastAPI Session Dependency
# ---------------------------------------------------------------------------


class TestGetDbSessionDependency:
    async def test_get_db_session_yields_session(self):
        """get_db_session() yields an AsyncSession and cleans up."""
        from src.db import get_db_session

        mock_session = AsyncMock(spec=AsyncSession)
        mock_db = MagicMock(spec=Database)

        class FakeCtx:
            async def __aenter__(self):  # noqa: N805
                return mock_session

            async def __aexit__(self, *args):  # noqa: N805
                await mock_session.commit()
                await mock_session.close()

        mock_db.get_session = FakeCtx

        with patch("src.db._get_database", return_value=mock_db):
            collected_sessions = []
            async for session in get_db_session():
                collected_sessions.append(session)
            assert len(collected_sessions) == 1
            assert collected_sessions[0] is mock_session

    async def test_get_db_session_raises_if_not_initialized(self):
        """get_db_session() raises RuntimeError if database not initialised."""
        import src.db as db_module

        original = db_module._database
        db_module._database = None
        try:
            with pytest.raises(RuntimeError, match="Database not initialised"):
                async for _ in db_module.get_db_session():
                    pass
        finally:
            db_module._database = original

    def test_init_database_creates_singleton(self):
        """init_database() creates and stores a Database instance."""
        import src.db as db_module

        original = db_module._database
        try:
            db = db_module.init_database("postgresql+asyncpg://user:pass@localhost:5432/testdb")
            assert isinstance(db, Database)
            assert db_module._database is db
        finally:
            db_module._database = original
