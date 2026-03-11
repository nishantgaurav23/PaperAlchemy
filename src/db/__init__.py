"""Database module — async SQLAlchemy engine, sessions, and ORM base."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

# Import models so Base.metadata sees them for create_tables / Alembic autogenerate.
import src.models.paper  # noqa: F401
from src.db.base import Base
from src.db.database import Database

_database: Database | None = None


def init_database(database_url: str, echo: bool = False) -> Database:
    """Create and store the singleton Database instance."""
    global _database  # noqa: PLW0603
    _database = Database(database_url=database_url, echo=echo)
    return _database


def _get_database() -> Database:
    """Return the singleton Database — raises if not initialised."""
    if _database is None:
        raise RuntimeError("Database not initialised. Call init_database() first.")
    return _database


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI Depends() async generator — yields an AsyncSession."""
    db = _get_database()
    async with db.get_session() as session:
        yield session


__all__ = [
    "Base",
    "Database",
    "get_db_session",
    "init_database",
]
