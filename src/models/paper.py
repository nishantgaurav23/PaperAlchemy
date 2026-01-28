"""SQLAlchemy model for arXiv papers."""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import String, Text, DateTime, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin

class Paper(Base, TimestampMixin):
    """arXiv paper model for PostgreSQL storage."""

    __tablename__ = 'papers'

    # Primary Key
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # arXiv metadata (required)
    arxiv_id: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        index=True
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[List[str]] = mapped_column(JSONB, nullable=False)
    abstract: Mapped[str] = mapped_column(Text, nullable=False)
    categories: Mapped[List[str]] = mapped_column(JSONB, nullable=False)
    published_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    pdf_url: Mapped[str] = mapped_column(String(500), nullable=False)

    # PDF content (populated after parsing)
    pdf_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sections: Mapped[Optional[list[dict]]] = mapped_column(JSONB, nullable=True)

    # Parsing metadata
    parsing_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending"  # pending, success, failed
    )
    parsing_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Indexes for common queries
    __table_args__ = (
        Index("idx_papers_published_date", "published_date"),
        Index("idx_papers_parsing_status", "parsing_status"),
        Index("idx_papers_categories", "categories", postgresql_using="gin")
    )

    def __repr__(self) -> str:
        return f"<Paper_id={self.id}, arxiv_id='{self.arxiv_id}', title='{self.title[50:]}...')>"
    
    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "categories": self.categories,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "updated_date": self.updated_date.isoformat() if self.updated_date else None,
            "pdf_url": self.pdf_url,
            "pdf_content": self.pdf_content,
            "sections": self.sections,
            "parsing_status": self.parsing_status,
            "parsing_error": self.parsing_error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
