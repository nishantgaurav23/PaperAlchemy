"""SQLAlchemy ORM model for arXiv papers."""

import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base

# Use JSON (portable) rather than JSONB so tests can use SQLite.
# PostgreSQL treats JSON columns efficiently enough; JSONB can be
# swapped in via Alembic migration if GIN-index performance matters.


class Paper(Base):
    """arXiv paper stored in PostgreSQL."""

    __tablename__ = "papers"

    # Primary key — UUID
    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )

    # arXiv metadata (required)
    arxiv_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[list] = mapped_column(JSON, nullable=False)
    abstract: Mapped[str] = mapped_column(Text, nullable=False)
    categories: Mapped[list] = mapped_column(JSON, nullable=False)
    published_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pdf_url: Mapped[str] = mapped_column(String(500), nullable=False)

    # PDF content (populated after parsing)
    pdf_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    sections: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # AI analysis (populated after LLM analysis)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    highlights: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    methodology: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Parsing metadata
    parsing_status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", insert_default="pending")
    parsing_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=True
    )

    __table_args__ = (
        Index("idx_papers_published_date", "published_date"),
        Index("idx_papers_parsing_status", "parsing_status"),
    )

    def __repr__(self) -> str:
        return f"<Paper arxiv_id='{self.arxiv_id}' title='{self.title[:50]}...'>"
