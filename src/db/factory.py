"""Database factory for dependency injection."""

from typing import Generator
from functools import lru_cache

from sqlalchemy.orm import Session

from src.config import get_settings
from src.db.database import Database

@lru_cache(maxsize=1)
def make_database() -> Database:
    """
    Create and cache a database instance.

    Returns:
        Database instance (singleton)
    """

    settings = get_settings()
    database = Database(
        database_url=settings.postgres.url,
        echo=settings.app.debug,
    )

    # Create tables on startup
    database.create_tables()
    return database

def get_db_session() -> Generator[Session, None,None]:
    """
    FastAPI dependency for database sessions.

    Usage:
        def get_papers(db: Session = Depends(get_db_session)):
            ...

    Yields:
        SQLAlchemy Session object
    """
    database = make_database()
    with database.get_session() as session:
        yield session

def reset_database_cache() -> None:
    """
    Reset the cached database instance.

    Useful for testing or reconnecting after config changes.
    """
    make_database.cache_clear()