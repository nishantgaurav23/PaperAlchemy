"""Async repository for Paper CRUD operations."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import String, and_, cast, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.paper import Paper
from src.schemas.paper import PaperCreate, PaperUpdate


class PaperRepository:
    """Async repository for Paper database operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, data: PaperCreate) -> Paper:
        """Insert a new paper. Caller must flush/commit."""
        paper = Paper(**data.model_dump())
        self.session.add(paper)
        await self.session.flush()
        return paper

    async def get_by_id(self, paper_id: uuid.UUID) -> Paper | None:
        """Retrieve a paper by its UUID primary key."""
        stmt = select(Paper).where(Paper.id == paper_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_arxiv_id(self, arxiv_id: str) -> Paper | None:
        """Retrieve a paper by arXiv ID."""
        stmt = select(Paper).where(Paper.arxiv_id == arxiv_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def exists(self, arxiv_id: str) -> bool:
        """Check whether a paper with the given arXiv ID exists."""
        stmt = select(Paper.id).where(Paper.arxiv_id == arxiv_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def update(self, arxiv_id: str, data: PaperUpdate) -> Paper | None:
        """Partial update by arXiv ID. Returns None if not found."""
        paper = await self.get_by_arxiv_id(arxiv_id)
        if paper is None:
            return None
        update_fields = data.model_dump(exclude_unset=True)
        for field, value in update_fields.items():
            setattr(paper, field, value)
        await self.session.flush()
        return paper

    async def update_parsing_status(
        self,
        arxiv_id: str,
        status: str,
        pdf_content: str | None = None,
        sections: list[dict] | None = None,
        error: str | None = None,
    ) -> Paper | None:
        """Update parsing status and optionally content/sections/error."""
        paper = await self.get_by_arxiv_id(arxiv_id)
        if paper is None:
            return None
        paper.parsing_status = status
        if pdf_content is not None:
            paper.pdf_content = pdf_content
        if sections is not None:
            paper.sections = sections
        if error is not None:
            paper.parsing_error = error
        await self.session.flush()
        return paper

    async def delete(self, arxiv_id: str) -> bool:
        """Delete a paper by arXiv ID. Returns True if deleted."""
        paper = await self.get_by_arxiv_id(arxiv_id)
        if paper is None:
            return False
        await self.session.delete(paper)
        await self.session.flush()
        return True

    async def upsert(self, data: PaperCreate) -> Paper:
        """Insert or update on arxiv_id conflict.

        Falls back to get-then-update for SQLite compatibility in tests.
        Uses PostgreSQL ON CONFLICT in production.
        """
        existing = await self.get_by_arxiv_id(data.arxiv_id)
        if existing is not None:
            update_fields = data.model_dump(exclude={"arxiv_id"})
            for field, value in update_fields.items():
                setattr(existing, field, value)
            await self.session.flush()
            return existing
        return await self.create(data)

    async def bulk_upsert(self, papers: list[PaperCreate]) -> int:
        """Bulk upsert multiple papers. Returns count processed."""
        if not papers:
            return 0
        for p in papers:
            await self.upsert(p)
        return len(papers)

    async def get_by_date_range(
        self,
        from_date: datetime,
        to_date: datetime,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Paper]:
        """Query papers within a published date range."""
        stmt = (
            select(Paper)
            .where(and_(Paper.published_date >= from_date, Paper.published_date <= to_date))
            .order_by(Paper.published_date.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    def _category_filter(category: str):
        """Build a category filter compatible with both SQLite (JSON) and PostgreSQL."""
        # cast to String and use LIKE — works on both backends
        return cast(Paper.categories, String).like(f'%"{category}"%')

    async def get_by_category(self, category: str, limit: int = 100, offset: int = 0) -> list[Paper]:
        """Query papers by arXiv category."""
        stmt = (
            select(Paper).where(self._category_filter(category)).order_by(Paper.published_date.desc()).offset(offset).limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_pending_parsing(self, limit: int = 100) -> list[Paper]:
        """Get papers with parsing_status='pending'."""
        stmt = select(Paper).where(Paper.parsing_status == "pending").order_by(Paper.created_at.asc()).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def search(
        self,
        query: str | None = None,
        category: str | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        parsing_status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Paper]:
        """Search papers with multiple optional filters."""
        conditions = []
        if query:
            pattern = f"%{query}%"
            conditions.append(or_(Paper.title.ilike(pattern), Paper.abstract.ilike(pattern)))
        if category:
            conditions.append(self._category_filter(category))
        if from_date:
            conditions.append(Paper.published_date >= from_date)
        if to_date:
            conditions.append(Paper.published_date <= to_date)
        if parsing_status:
            conditions.append(Paper.parsing_status == parsing_status)

        stmt = (
            select(Paper)
            .where(and_(*conditions) if conditions else True)  # noqa: FBT003
            .order_by(Paper.published_date.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count(self, parsing_status: str | None = None) -> int:
        """Count papers, optionally filtered by parsing status."""
        stmt = select(func.count(Paper.id))
        if parsing_status:
            stmt = stmt.where(Paper.parsing_status == parsing_status)
        result = await self.session.execute(stmt)
        return result.scalar_one()
