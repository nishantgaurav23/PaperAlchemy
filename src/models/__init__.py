"""SQLAlchemy models."""

from src.models.base import Base, TimestampMixin
from src.models.paper import Paper

__all__ = [
    "Base",
    "TimestampMixin",
    "Paper",
]