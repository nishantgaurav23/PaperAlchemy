"""Async repository for Collection CRUD operations."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.collection import Collection
from src.models.paper import Paper
from src.schemas.collection import CollectionCreate, CollectionUpdate


class CollectionRepository:
    """Async repository for Collection database operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, data: CollectionCreate) -> Collection:
        """Insert a new collection. Caller must flush/commit."""
        collection = Collection(**data.model_dump())
        self.session.add(collection)
        await self.session.flush()
        return collection

    async def get_by_id(self, collection_id: uuid.UUID) -> Collection | None:
        """Retrieve a collection by UUID (papers eagerly loaded via selectin)."""
        stmt = select(Collection).where(Collection.id == collection_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(
        self,
        user_id: uuid.UUID | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Collection]:
        """List collections, optionally filtered by user_id."""
        stmt = select(Collection).order_by(Collection.created_at.desc()).offset(offset).limit(limit)
        if user_id is not None:
            stmt = stmt.where(Collection.user_id == user_id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update(self, collection_id: uuid.UUID, data: CollectionUpdate) -> Collection | None:
        """Partial update by collection ID. Returns None if not found."""
        collection = await self.get_by_id(collection_id)
        if collection is None:
            return None
        update_fields = data.model_dump(exclude_unset=True)
        for field, value in update_fields.items():
            setattr(collection, field, value)
        await self.session.flush()
        await self.session.refresh(collection)
        return collection

    async def delete(self, collection_id: uuid.UUID) -> bool:
        """Delete a collection. Returns True if deleted."""
        collection = await self.get_by_id(collection_id)
        if collection is None:
            return False
        await self.session.delete(collection)
        await self.session.flush()
        return True

    async def add_paper(self, collection_id: uuid.UUID, paper_id: uuid.UUID) -> bool:
        """Add a paper to a collection. Idempotent — returns True on success, False if not found."""
        collection = await self.get_by_id(collection_id)
        if collection is None:
            return False

        # Check paper exists
        paper_stmt = select(Paper).where(Paper.id == paper_id)
        paper_result = await self.session.execute(paper_stmt)
        paper = paper_result.scalar_one_or_none()
        if paper is None:
            return False

        # Check if already in collection (idempotent)
        if paper in collection.papers:
            return True

        collection.papers.append(paper)
        await self.session.flush()
        return True

    async def remove_paper(self, collection_id: uuid.UUID, paper_id: uuid.UUID) -> bool:
        """Remove a paper from a collection. Returns False if not found or not in collection."""
        collection = await self.get_by_id(collection_id)
        if collection is None:
            return False

        # Find the paper in the collection
        for paper in collection.papers:
            if paper.id == paper_id:
                collection.papers.remove(paper)
                await self.session.flush()
                return True
        return False

    async def get_collection_papers(self, collection_id: uuid.UUID) -> list[Paper]:
        """Get all papers in a collection."""
        collection = await self.get_by_id(collection_id)
        if collection is None:
            return []
        return list(collection.papers)

    async def count(self, user_id: uuid.UUID | None = None) -> int:
        """Count collections, optionally filtered by user_id."""
        stmt = select(func.count(Collection.id))
        if user_id is not None:
            stmt = stmt.where(Collection.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalar_one()
