"""Database connection and session management."""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from src.models.base import Base

class Database:
    """Database connection manager with session handling."""

    def __init__(self, database_url: str, echo: bool = False):
        """
        Initialize database connection.

        Args:
            database_url: PostgreSQL connection URL
            echi: If True, log all SQL statements.
        """

        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            echo=echo,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True, # Verify connections before use
            pool_recycle=3600, # Recycle connections after 1 hour
        )
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )
    
    def create_tables(self) -> None:
        """Create all tables defined in the models."""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self) -> None:
        """Drop all tables. USE WITH CAUTIONS."""
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a database session with automatic cleanup.

        Usage:
            with self.get_session() as session:
                # do database operations
                session.commit()
        Yields:
            SQLAlchemy Session object

        Raises:
            SQLAlchemyError: On database errors (auto-rollback)
        """

        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_session_no_commit(self) -> Session:
        """
        Get a session without auto-commit (for manual control).

        Returns:
            SQLAlchemy Session object

        Notes:
            Caller is responsible for commit/rollack/close
        """
        return self.SessionLocal()
    
    def health_check(self) -> bool:
        """
        Check database connectivity.

        Returns:
            True if database is accessible, False otherwise.
        """

        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
        
    def close(self) -> None:
        """Close the database engine and all connections."""
        self.engine.dispose()
    
            
        