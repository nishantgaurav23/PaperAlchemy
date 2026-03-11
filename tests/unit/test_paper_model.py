"""Tests for S3.1 — Paper ORM Model, Repository, and Schemas.

Uses an in-memory SQLite async engine for isolation.
JSONB columns fall back to JSON on SQLite automatically via SQLAlchemy.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from src.db.base import Base

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def engine():
    """Create an in-memory SQLite async engine."""
    eng = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


@pytest.fixture
async def session(engine) -> AsyncSession:
    """Yield an async session bound to the in-memory engine."""
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as sess:
        yield sess


def _paper_data(**overrides) -> dict:
    """Helper to build valid PaperCreate kwargs."""
    defaults = {
        "arxiv_id": "2401.00001",
        "title": "Attention Is All You Need",
        "authors": ["Vaswani", "Shazeer"],
        "abstract": "We propose a new simple network architecture.",
        "categories": ["cs.AI", "cs.CL"],
        "published_date": datetime(2024, 1, 15, tzinfo=UTC),
        "pdf_url": "https://arxiv.org/pdf/2401.00001v1",
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# FR-1: Paper ORM Model
# ---------------------------------------------------------------------------


class TestPaperModel:
    def test_paper_table_name(self):
        """Paper model table name is 'papers'."""
        from src.models.paper import Paper

        assert Paper.__tablename__ == "papers"

    def test_paper_columns_exist(self):
        """Paper model has all required columns."""
        from src.models.paper import Paper

        mapper = inspect(Paper)
        column_names = {c.key for c in mapper.column_attrs}
        expected = {
            "id",
            "arxiv_id",
            "title",
            "authors",
            "abstract",
            "categories",
            "published_date",
            "updated_date",
            "pdf_url",
            "pdf_content",
            "sections",
            "parsing_status",
            "parsing_error",
            "created_at",
            "updated_at",
        }
        assert expected.issubset(column_names), f"Missing columns: {expected - column_names}"

    def test_paper_inherits_from_base(self):
        """Paper inherits from our declarative Base."""
        from src.models.paper import Paper

        assert issubclass(Paper, Base)

    def test_paper_repr(self):
        """Paper has a useful __repr__."""
        from src.models.paper import Paper

        p = Paper(arxiv_id="2401.00001", title="Test Paper Title")
        r = repr(p)
        assert "2401.00001" in r

    async def test_paper_default_parsing_status(self, session):
        """Paper default parsing_status is 'pending' after insert."""
        from src.models.paper import Paper

        p = Paper(
            arxiv_id="2401.99999",
            title="Test",
            authors=["A"],
            abstract="Abs",
            categories=["cs.AI"],
            published_date=datetime(2024, 1, 1, tzinfo=UTC),
            pdf_url="https://arxiv.org/pdf/2401.99999",
        )
        session.add(p)
        await session.flush()
        assert p.parsing_status == "pending"


# ---------------------------------------------------------------------------
# FR-2: Paper Repository (Async CRUD)
# ---------------------------------------------------------------------------


class TestPaperRepositoryCreate:
    async def test_create_paper(self, session):
        """Create a paper and verify fields are persisted."""
        from src.repositories.paper import PaperRepository
        from src.schemas.paper import PaperCreate

        repo = PaperRepository(session)
        data = PaperCreate(**_paper_data())
        paper = await repo.create(data)

        assert paper.arxiv_id == "2401.00001"
        assert paper.title == "Attention Is All You Need"
        assert paper.authors == ["Vaswani", "Shazeer"]
        assert paper.categories == ["cs.AI", "cs.CL"]
        assert paper.parsing_status == "pending"
        assert paper.id is not None

    async def test_create_duplicate_raises(self, session):
        """Creating a paper with a duplicate arxiv_id raises IntegrityError."""
        from sqlalchemy.exc import IntegrityError
        from src.repositories.paper import PaperRepository
        from src.schemas.paper import PaperCreate

        repo = PaperRepository(session)
        data = PaperCreate(**_paper_data())
        await repo.create(data)
        await session.flush()

        with pytest.raises(IntegrityError):
            await repo.create(data)
            await session.flush()


class TestPaperRepositoryRead:
    async def test_get_by_id(self, session):
        """Retrieve a paper by its UUID."""
        from src.repositories.paper import PaperRepository
        from src.schemas.paper import PaperCreate

        repo = PaperRepository(session)
        created = await repo.create(PaperCreate(**_paper_data()))
        await session.flush()

        found = await repo.get_by_id(created.id)
        assert found is not None
        assert found.arxiv_id == "2401.00001"

    async def test_get_by_id_not_found(self, session):
        """get_by_id returns None for non-existent UUID."""
        from src.repositories.paper import PaperRepository

        repo = PaperRepository(session)
        result = await repo.get_by_id(uuid.uuid4())
        assert result is None

    async def test_get_by_arxiv_id(self, session):
        """Retrieve a paper by arXiv ID."""
        from src.repositories.paper import PaperRepository
        from src.schemas.paper import PaperCreate

        repo = PaperRepository(session)
        await repo.create(PaperCreate(**_paper_data()))
        await session.flush()

        found = await repo.get_by_arxiv_id("2401.00001")
        assert found is not None
        assert found.title == "Attention Is All You Need"

    async def test_get_by_arxiv_id_not_found(self, session):
        """get_by_arxiv_id returns None for non-existent arXiv ID."""
        from src.repositories.paper import PaperRepository

        repo = PaperRepository(session)
        result = await repo.get_by_arxiv_id("nonexistent")
        assert result is None

    async def test_exists(self, session):
        """exists() returns True for existing paper."""
        from src.repositories.paper import PaperRepository
        from src.schemas.paper import PaperCreate

        repo = PaperRepository(session)
        await repo.create(PaperCreate(**_paper_data()))
        await session.flush()

        assert await repo.exists("2401.00001") is True

    async def test_exists_not_found(self, session):
        """exists() returns False for non-existent paper."""
        from src.repositories.paper import PaperRepository

        repo = PaperRepository(session)
        assert await repo.exists("nonexistent") is False


class TestPaperRepositoryUpdate:
    async def test_update(self, session):
        """Partial update on a paper."""
        from src.repositories.paper import PaperRepository
        from src.schemas.paper import PaperCreate, PaperUpdate

        repo = PaperRepository(session)
        await repo.create(PaperCreate(**_paper_data()))
        await session.flush()

        updated = await repo.update("2401.00001", PaperUpdate(title="New Title"))
        assert updated is not None
        assert updated.title == "New Title"
        assert updated.authors == ["Vaswani", "Shazeer"]  # unchanged

    async def test_update_not_found(self, session):
        """update() returns None for non-existent paper."""
        from src.repositories.paper import PaperRepository
        from src.schemas.paper import PaperUpdate

        repo = PaperRepository(session)
        result = await repo.update("nonexistent", PaperUpdate(title="X"))
        assert result is None

    async def test_update_parsing_status(self, session):
        """Update parsing status and content."""
        from src.repositories.paper import PaperRepository
        from src.schemas.paper import PaperCreate

        repo = PaperRepository(session)
        await repo.create(PaperCreate(**_paper_data()))
        await session.flush()

        updated = await repo.update_parsing_status(
            "2401.00001",
            status="success",
            pdf_content="Parsed text content here",
            sections=[{"title": "Introduction", "content": "..."}],
        )
        assert updated is not None
        assert updated.parsing_status == "success"
        assert updated.pdf_content == "Parsed text content here"
        assert updated.sections == [{"title": "Introduction", "content": "..."}]

    async def test_update_parsing_status_failed(self, session):
        """Update parsing status to failed with error message."""
        from src.repositories.paper import PaperRepository
        from src.schemas.paper import PaperCreate

        repo = PaperRepository(session)
        await repo.create(PaperCreate(**_paper_data()))
        await session.flush()

        updated = await repo.update_parsing_status("2401.00001", status="failed", error="PDF corrupted")
        assert updated is not None
        assert updated.parsing_status == "failed"
        assert updated.parsing_error == "PDF corrupted"

    async def test_update_parsing_status_not_found(self, session):
        """update_parsing_status returns None for non-existent paper."""
        from src.repositories.paper import PaperRepository

        repo = PaperRepository(session)
        result = await repo.update_parsing_status("nonexistent", status="success")
        assert result is None


class TestPaperRepositoryDelete:
    async def test_delete(self, session):
        """Delete a paper by arXiv ID."""
        from src.repositories.paper import PaperRepository
        from src.schemas.paper import PaperCreate

        repo = PaperRepository(session)
        await repo.create(PaperCreate(**_paper_data()))
        await session.flush()

        assert await repo.delete("2401.00001") is True
        assert await repo.exists("2401.00001") is False

    async def test_delete_not_found(self, session):
        """delete() returns False for non-existent paper."""
        from src.repositories.paper import PaperRepository

        repo = PaperRepository(session)
        assert await repo.delete("nonexistent") is False


class TestPaperRepositoryUpsert:
    async def test_upsert_insert(self, session):
        """Upsert creates a new paper when none exists."""
        from src.repositories.paper import PaperRepository
        from src.schemas.paper import PaperCreate

        repo = PaperRepository(session)
        data = PaperCreate(**_paper_data())
        paper = await repo.upsert(data)
        assert paper.arxiv_id == "2401.00001"

    async def test_upsert_update(self, session):
        """Upsert updates an existing paper's fields."""
        from src.repositories.paper import PaperRepository
        from src.schemas.paper import PaperCreate

        repo = PaperRepository(session)

        # First insert
        data1 = PaperCreate(**_paper_data())
        await repo.upsert(data1)
        await session.flush()

        # Update via upsert
        data2 = PaperCreate(**_paper_data(title="Updated Title", abstract="Updated abstract"))
        paper = await repo.upsert(data2)
        assert paper.title == "Updated Title"
        assert paper.abstract == "Updated abstract"

    async def test_bulk_upsert(self, session):
        """Bulk upsert multiple papers."""
        from src.repositories.paper import PaperRepository
        from src.schemas.paper import PaperCreate

        repo = PaperRepository(session)
        papers = [PaperCreate(**_paper_data(arxiv_id=f"2401.{i:05d}")) for i in range(5)]
        count = await repo.bulk_upsert(papers)
        assert count == 5

    async def test_bulk_upsert_empty(self, session):
        """Bulk upsert with empty list returns 0."""
        from src.repositories.paper import PaperRepository

        repo = PaperRepository(session)
        count = await repo.bulk_upsert([])
        assert count == 0


class TestPaperRepositoryQueries:
    async def _seed_papers(self, session):
        """Insert a few papers for query tests."""
        from src.repositories.paper import PaperRepository
        from src.schemas.paper import PaperCreate

        repo = PaperRepository(session)
        papers = [
            PaperCreate(
                **_paper_data(
                    arxiv_id="2401.00001",
                    title="Transformers in NLP",
                    categories=["cs.CL"],
                    published_date=datetime(2024, 1, 10, tzinfo=UTC),
                )
            ),
            PaperCreate(
                **_paper_data(
                    arxiv_id="2401.00002",
                    title="Deep Reinforcement Learning",
                    categories=["cs.AI", "cs.LG"],
                    published_date=datetime(2024, 1, 20, tzinfo=UTC),
                )
            ),
            PaperCreate(
                **_paper_data(
                    arxiv_id="2401.00003",
                    title="Quantum Computing Survey",
                    categories=["quant-ph"],
                    published_date=datetime(2024, 2, 5, tzinfo=UTC),
                    parsing_status="success",
                )
            ),
        ]
        for p in papers:
            await repo.create(p)
        await session.flush()
        return repo

    async def test_get_by_date_range(self, session):
        """Query papers within a date range."""
        repo = await self._seed_papers(session)
        results = await repo.get_by_date_range(
            from_date=datetime(2024, 1, 1, tzinfo=UTC),
            to_date=datetime(2024, 1, 31, tzinfo=UTC),
        )
        assert len(results) == 2
        # Ordered by published_date desc
        assert results[0].arxiv_id == "2401.00002"

    async def test_get_by_category(self, session):
        """Query papers by category."""
        repo = await self._seed_papers(session)
        results = await repo.get_by_category("cs.AI")
        assert len(results) == 1
        assert results[0].arxiv_id == "2401.00002"

    async def test_get_pending_parsing(self, session):
        """Query papers with pending parsing status."""
        repo = await self._seed_papers(session)
        results = await repo.get_pending_parsing()
        assert len(results) == 2  # papers 1 and 2 are pending
        for p in results:
            assert p.parsing_status == "pending"

    async def test_search_by_title(self, session):
        """Search papers by title text."""
        repo = await self._seed_papers(session)
        results = await repo.search(query="Transformer")
        assert len(results) == 1
        assert results[0].arxiv_id == "2401.00001"

    async def test_search_multi_filter(self, session):
        """Search with multiple filters combined."""
        repo = await self._seed_papers(session)
        results = await repo.search(
            parsing_status="pending",
            from_date=datetime(2024, 1, 15, tzinfo=UTC),
        )
        assert len(results) == 1
        assert results[0].arxiv_id == "2401.00002"

    async def test_count(self, session):
        """Count all papers."""
        repo = await self._seed_papers(session)
        total = await repo.count()
        assert total == 3

    async def test_count_with_status(self, session):
        """Count papers filtered by parsing status."""
        repo = await self._seed_papers(session)
        pending = await repo.count(parsing_status="pending")
        assert pending == 2
        success = await repo.count(parsing_status="success")
        assert success == 1


# ---------------------------------------------------------------------------
# FR-3: Pydantic Schemas
# ---------------------------------------------------------------------------


class TestPydanticSchemas:
    def test_paper_create_validation(self):
        """PaperCreate validates required fields."""
        from src.schemas.paper import PaperCreate

        data = PaperCreate(**_paper_data())
        assert data.arxiv_id == "2401.00001"
        assert data.parsing_status == "pending"

    def test_paper_create_missing_required(self):
        """PaperCreate raises ValidationError for missing required fields."""
        from pydantic import ValidationError
        from src.schemas.paper import PaperCreate

        with pytest.raises(ValidationError):
            PaperCreate(arxiv_id="test")  # missing other required fields

    def test_paper_update_all_optional(self):
        """PaperUpdate allows all fields to be optional."""
        from src.schemas.paper import PaperUpdate

        update = PaperUpdate()
        data = update.model_dump(exclude_unset=True)
        assert data == {}

    def test_paper_update_partial(self):
        """PaperUpdate captures only set fields."""
        from src.schemas.paper import PaperUpdate

        update = PaperUpdate(title="New Title")
        data = update.model_dump(exclude_unset=True)
        assert data == {"title": "New Title"}

    def test_paper_response_from_attributes(self):
        """PaperResponse can be created from ORM model attributes."""
        from src.schemas.paper import PaperResponse

        assert PaperResponse.model_config.get("from_attributes") is True
