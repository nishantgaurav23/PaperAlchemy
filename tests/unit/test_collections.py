"""Tests for S9b.6 — Collections Backend API.

Tests the Collection ORM model, CollectionRepository (CRUD + paper operations),
Pydantic schemas, and REST API endpoints. Uses an in-memory SQLite async engine.
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
    """Create an in-memory SQLite async engine with all tables."""
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


async def _create_paper(session, **overrides):
    """Helper to create a paper in the database."""
    from src.repositories.paper import PaperRepository
    from src.schemas.paper import PaperCreate

    repo = PaperRepository(session)
    paper = await repo.create(PaperCreate(**_paper_data(**overrides)))
    await session.flush()
    return paper


# ---------------------------------------------------------------------------
# FR-1: Collection Model
# ---------------------------------------------------------------------------


class TestCollectionModel:
    def test_collection_table_name(self):
        """Collection model table name is 'collections'."""
        from src.models.collection import Collection

        assert Collection.__tablename__ == "collections"

    def test_collection_columns_exist(self):
        """Collection model has all required columns."""
        from src.models.collection import Collection

        mapper = inspect(Collection)
        column_names = {c.key for c in mapper.column_attrs}
        expected = {"id", "name", "description", "user_id", "created_at", "updated_at"}
        assert expected.issubset(column_names), f"Missing columns: {expected - column_names}"

    def test_collection_inherits_from_base(self):
        """Collection inherits from our declarative Base."""
        from src.models.collection import Collection

        assert issubclass(Collection, Base)

    def test_collection_repr(self):
        """Collection has a useful __repr__."""
        from src.models.collection import Collection

        c = Collection(name="My Reading List")
        r = repr(c)
        assert "My Reading List" in r

    def test_association_table_exists(self):
        """collection_papers association table is registered in metadata."""
        assert "collection_papers" in Base.metadata.tables

    async def test_collection_create_and_persist(self, session):
        """Collection can be created and persisted."""
        from src.models.collection import Collection

        c = Collection(name="Test Collection", description="A test")
        session.add(c)
        await session.flush()
        assert c.id is not None
        assert c.name == "Test Collection"


# ---------------------------------------------------------------------------
# FR-2: Collection Repository
# ---------------------------------------------------------------------------


class TestCollectionRepositoryCreate:
    async def test_create_collection(self, session):
        """Create a collection and verify fields."""
        from src.repositories.collection import CollectionRepository
        from src.schemas.collection import CollectionCreate

        repo = CollectionRepository(session)
        data = CollectionCreate(name="ML Papers", description="Machine learning papers")
        collection = await repo.create(data)

        assert collection.name == "ML Papers"
        assert collection.description == "Machine learning papers"
        assert collection.id is not None

    async def test_create_collection_minimal(self, session):
        """Create collection with only required fields."""
        from src.repositories.collection import CollectionRepository
        from src.schemas.collection import CollectionCreate

        repo = CollectionRepository(session)
        data = CollectionCreate(name="Minimal")
        collection = await repo.create(data)

        assert collection.name == "Minimal"
        assert collection.description is None


class TestCollectionRepositoryRead:
    async def test_get_by_id(self, session):
        """Retrieve a collection by UUID."""
        from src.repositories.collection import CollectionRepository
        from src.schemas.collection import CollectionCreate

        repo = CollectionRepository(session)
        created = await repo.create(CollectionCreate(name="Test"))

        found = await repo.get_by_id(created.id)
        assert found is not None
        assert found.name == "Test"

    async def test_get_by_id_not_found(self, session):
        """get_by_id returns None for non-existent UUID."""
        from src.repositories.collection import CollectionRepository

        repo = CollectionRepository(session)
        result = await repo.get_by_id(uuid.uuid4())
        assert result is None

    async def test_list_all(self, session):
        """List all collections."""
        from src.repositories.collection import CollectionRepository
        from src.schemas.collection import CollectionCreate

        repo = CollectionRepository(session)
        await repo.create(CollectionCreate(name="First"))
        await repo.create(CollectionCreate(name="Second"))

        results = await repo.list_all()
        assert len(results) == 2

    async def test_list_all_with_pagination(self, session):
        """List collections with limit and offset."""
        from src.repositories.collection import CollectionRepository
        from src.schemas.collection import CollectionCreate

        repo = CollectionRepository(session)
        for i in range(5):
            await repo.create(CollectionCreate(name=f"Collection {i}"))

        results = await repo.list_all(limit=2, offset=1)
        assert len(results) == 2

    async def test_list_all_filter_by_user_id(self, session):
        """List collections filtered by user_id."""
        from src.repositories.collection import CollectionRepository
        from src.schemas.collection import CollectionCreate

        repo = CollectionRepository(session)
        user_id = uuid.uuid4()
        c1 = await repo.create(CollectionCreate(name="User's Collection"))
        # Manually set user_id since CollectionCreate doesn't expose it
        c1.user_id = user_id
        await session.flush()

        await repo.create(CollectionCreate(name="Another Collection"))

        results = await repo.list_all(user_id=user_id)
        assert len(results) == 1
        assert results[0].name == "User's Collection"

    async def test_count(self, session):
        """Count all collections."""
        from src.repositories.collection import CollectionRepository
        from src.schemas.collection import CollectionCreate

        repo = CollectionRepository(session)
        await repo.create(CollectionCreate(name="A"))
        await repo.create(CollectionCreate(name="B"))

        assert await repo.count() == 2

    async def test_count_with_user_id(self, session):
        """Count collections filtered by user_id."""
        from src.repositories.collection import CollectionRepository
        from src.schemas.collection import CollectionCreate

        repo = CollectionRepository(session)
        user_id = uuid.uuid4()
        c = await repo.create(CollectionCreate(name="User's"))
        c.user_id = user_id
        await session.flush()
        await repo.create(CollectionCreate(name="Other"))

        assert await repo.count(user_id=user_id) == 1


class TestCollectionRepositoryUpdate:
    async def test_update_collection(self, session):
        """Update collection name and description."""
        from src.repositories.collection import CollectionRepository
        from src.schemas.collection import CollectionCreate, CollectionUpdate

        repo = CollectionRepository(session)
        created = await repo.create(CollectionCreate(name="Old Name", description="Old desc"))

        updated = await repo.update(created.id, CollectionUpdate(name="New Name"))
        assert updated is not None
        assert updated.name == "New Name"
        assert updated.description == "Old desc"  # unchanged

    async def test_update_not_found(self, session):
        """update returns None for non-existent collection."""
        from src.repositories.collection import CollectionRepository
        from src.schemas.collection import CollectionUpdate

        repo = CollectionRepository(session)
        result = await repo.update(uuid.uuid4(), CollectionUpdate(name="X"))
        assert result is None


class TestCollectionRepositoryDelete:
    async def test_delete_collection(self, session):
        """Delete a collection."""
        from src.repositories.collection import CollectionRepository
        from src.schemas.collection import CollectionCreate

        repo = CollectionRepository(session)
        created = await repo.create(CollectionCreate(name="To Delete"))

        assert await repo.delete(created.id) is True
        assert await repo.get_by_id(created.id) is None

    async def test_delete_not_found(self, session):
        """delete returns False for non-existent collection."""
        from src.repositories.collection import CollectionRepository

        repo = CollectionRepository(session)
        assert await repo.delete(uuid.uuid4()) is False


class TestCollectionRepositoryPaperOps:
    async def test_add_paper(self, session):
        """Add a paper to a collection."""
        from src.repositories.collection import CollectionRepository
        from src.schemas.collection import CollectionCreate

        paper = await _create_paper(session)
        repo = CollectionRepository(session)
        collection = await repo.create(CollectionCreate(name="Reading List"))

        result = await repo.add_paper(collection.id, paper.id)
        assert result is True

        papers = await repo.get_collection_papers(collection.id)
        assert len(papers) == 1
        assert papers[0].id == paper.id

    async def test_add_paper_idempotent(self, session):
        """Adding the same paper twice is idempotent."""
        from src.repositories.collection import CollectionRepository
        from src.schemas.collection import CollectionCreate

        paper = await _create_paper(session)
        repo = CollectionRepository(session)
        collection = await repo.create(CollectionCreate(name="List"))

        await repo.add_paper(collection.id, paper.id)
        result = await repo.add_paper(collection.id, paper.id)
        assert result is True  # succeeds silently

        papers = await repo.get_collection_papers(collection.id)
        assert len(papers) == 1  # still only one

    async def test_add_paper_collection_not_found(self, session):
        """add_paper returns False for non-existent collection."""
        from src.repositories.collection import CollectionRepository

        paper = await _create_paper(session)
        repo = CollectionRepository(session)
        result = await repo.add_paper(uuid.uuid4(), paper.id)
        assert result is False

    async def test_add_paper_paper_not_found(self, session):
        """add_paper returns False for non-existent paper."""
        from src.repositories.collection import CollectionRepository
        from src.schemas.collection import CollectionCreate

        repo = CollectionRepository(session)
        collection = await repo.create(CollectionCreate(name="List"))
        result = await repo.add_paper(collection.id, uuid.uuid4())
        assert result is False

    async def test_remove_paper(self, session):
        """Remove a paper from a collection."""
        from src.repositories.collection import CollectionRepository
        from src.schemas.collection import CollectionCreate

        paper = await _create_paper(session)
        repo = CollectionRepository(session)
        collection = await repo.create(CollectionCreate(name="List"))
        await repo.add_paper(collection.id, paper.id)

        result = await repo.remove_paper(collection.id, paper.id)
        assert result is True

        papers = await repo.get_collection_papers(collection.id)
        assert len(papers) == 0

    async def test_remove_paper_not_in_collection(self, session):
        """remove_paper returns False if paper not in collection."""
        from src.repositories.collection import CollectionRepository
        from src.schemas.collection import CollectionCreate

        paper = await _create_paper(session)
        repo = CollectionRepository(session)
        collection = await repo.create(CollectionCreate(name="List"))

        result = await repo.remove_paper(collection.id, paper.id)
        assert result is False

    async def test_get_collection_papers_empty(self, session):
        """get_collection_papers returns empty list for collection with no papers."""
        from src.repositories.collection import CollectionRepository
        from src.schemas.collection import CollectionCreate

        repo = CollectionRepository(session)
        collection = await repo.create(CollectionCreate(name="Empty"))

        papers = await repo.get_collection_papers(collection.id)
        assert papers == []

    async def test_get_collection_papers_not_found(self, session):
        """get_collection_papers returns empty list for non-existent collection."""
        from src.repositories.collection import CollectionRepository

        repo = CollectionRepository(session)
        papers = await repo.get_collection_papers(uuid.uuid4())
        assert papers == []

    async def test_delete_collection_cascades_m2m(self, session):
        """Deleting a collection removes M2M links but not the papers."""
        from src.repositories.collection import CollectionRepository
        from src.repositories.paper import PaperRepository
        from src.schemas.collection import CollectionCreate

        paper = await _create_paper(session)
        repo = CollectionRepository(session)
        collection = await repo.create(CollectionCreate(name="Will Delete"))
        await repo.add_paper(collection.id, paper.id)

        await repo.delete(collection.id)

        # Paper should still exist
        paper_repo = PaperRepository(session)
        found = await paper_repo.get_by_id(paper.id)
        assert found is not None

    async def test_multiple_papers_in_collection(self, session):
        """Add multiple papers to a collection."""
        from src.repositories.collection import CollectionRepository
        from src.schemas.collection import CollectionCreate

        p1 = await _create_paper(session, arxiv_id="2401.00001")
        p2 = await _create_paper(session, arxiv_id="2401.00002")

        repo = CollectionRepository(session)
        collection = await repo.create(CollectionCreate(name="Multi"))
        await repo.add_paper(collection.id, p1.id)
        await repo.add_paper(collection.id, p2.id)

        papers = await repo.get_collection_papers(collection.id)
        assert len(papers) == 2


# ---------------------------------------------------------------------------
# FR-3: Pydantic Schemas
# ---------------------------------------------------------------------------


class TestCollectionSchemas:
    def test_collection_create_valid(self):
        """CollectionCreate validates required name field."""
        from src.schemas.collection import CollectionCreate

        data = CollectionCreate(name="Test", description="Desc")
        assert data.name == "Test"
        assert data.description == "Desc"

    def test_collection_create_name_required(self):
        """CollectionCreate raises for missing name."""
        from pydantic import ValidationError
        from src.schemas.collection import CollectionCreate

        with pytest.raises(ValidationError):
            CollectionCreate()

    def test_collection_update_all_optional(self):
        """CollectionUpdate allows all fields to be optional."""
        from src.schemas.collection import CollectionUpdate

        update = CollectionUpdate()
        data = update.model_dump(exclude_unset=True)
        assert data == {}

    def test_collection_response_from_attributes(self):
        """CollectionResponse has from_attributes config."""
        from src.schemas.collection import CollectionResponse

        assert CollectionResponse.model_config.get("from_attributes") is True

    def test_collection_detail_response_has_papers(self):
        """CollectionDetailResponse includes papers list."""
        from src.schemas.collection import CollectionDetailResponse

        fields = CollectionDetailResponse.model_fields
        assert "papers" in fields

    def test_collection_paper_action_valid(self):
        """CollectionPaperAction accepts a UUID."""
        from src.schemas.collection import CollectionPaperAction

        action = CollectionPaperAction(paper_id=uuid.uuid4())
        assert action.paper_id is not None


# ---------------------------------------------------------------------------
# FR-4: REST API Endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
async def api_client(engine):
    """Create a FastAPI test client with real DB session."""
    from httpx import ASGITransport, AsyncClient
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from src.dependency import get_db_session
    from src.main import create_app

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async def override_session():
        async with session_factory() as sess:
            yield sess

    app = create_app()
    app.dependency_overrides[get_db_session] = override_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestCollectionAPI:
    async def test_create_collection(self, api_client):
        """POST /api/v1/collections creates a collection."""
        resp = await api_client.post("/api/v1/collections", json={"name": "My List", "description": "Test"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "My List"
        assert data["description"] == "Test"
        assert "id" in data

    async def test_list_collections_empty(self, api_client):
        """GET /api/v1/collections returns empty list."""
        resp = await api_client.get("/api/v1/collections")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_collections(self, api_client):
        """GET /api/v1/collections returns created collections."""
        await api_client.post("/api/v1/collections", json={"name": "A"})
        await api_client.post("/api/v1/collections", json={"name": "B"})

        resp = await api_client.get("/api/v1/collections")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    async def test_get_collection(self, api_client):
        """GET /api/v1/collections/{id} returns collection with papers."""
        create_resp = await api_client.post("/api/v1/collections", json={"name": "Detail Test"})
        cid = create_resp.json()["id"]

        resp = await api_client.get(f"/api/v1/collections/{cid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Detail Test"
        assert "papers" in data

    async def test_get_collection_not_found(self, api_client):
        """GET /api/v1/collections/{id} returns 404 for non-existent."""
        resp = await api_client.get(f"/api/v1/collections/{uuid.uuid4()}")
        assert resp.status_code == 404

    async def test_update_collection(self, api_client):
        """PUT /api/v1/collections/{id} updates collection."""
        create_resp = await api_client.post("/api/v1/collections", json={"name": "Old"})
        cid = create_resp.json()["id"]

        resp = await api_client.put(f"/api/v1/collections/{cid}", json={"name": "New"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "New"

    async def test_update_collection_not_found(self, api_client):
        """PUT /api/v1/collections/{id} returns 404."""
        resp = await api_client.put(f"/api/v1/collections/{uuid.uuid4()}", json={"name": "X"})
        assert resp.status_code == 404

    async def test_delete_collection(self, api_client):
        """DELETE /api/v1/collections/{id} deletes collection."""
        create_resp = await api_client.post("/api/v1/collections", json={"name": "Delete Me"})
        cid = create_resp.json()["id"]

        resp = await api_client.delete(f"/api/v1/collections/{cid}")
        assert resp.status_code == 204

        # Verify deleted
        resp = await api_client.get(f"/api/v1/collections/{cid}")
        assert resp.status_code == 404

    async def test_delete_collection_not_found(self, api_client):
        """DELETE /api/v1/collections/{id} returns 404."""
        resp = await api_client.delete(f"/api/v1/collections/{uuid.uuid4()}")
        assert resp.status_code == 404


class TestCollectionPaperAPI:
    async def _create_collection_and_paper(self, api_client, session_factory=None):
        """Helper: create a collection and a paper via API / direct DB."""
        # Create collection via API
        resp = await api_client.post("/api/v1/collections", json={"name": "Papers"})
        cid = resp.json()["id"]
        return cid

    async def test_add_paper_to_collection(self, api_client, engine):
        """POST /api/v1/collections/{id}/papers adds paper."""
        # Create paper directly in DB
        session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        async with session_factory() as sess:
            paper = await _create_paper(sess, arxiv_id="2401.99001")
            await sess.commit()
            paper_id = str(paper.id)

        resp = await api_client.post("/api/v1/collections", json={"name": "Test"})
        cid = resp.json()["id"]

        resp = await api_client.post(f"/api/v1/collections/{cid}/papers", json={"paper_id": paper_id})
        assert resp.status_code == 200
        assert resp.json()["message"] == "Paper added to collection"

    async def test_add_paper_collection_not_found(self, api_client):
        """POST /api/v1/collections/{id}/papers returns 404 for bad collection."""
        resp = await api_client.post(
            f"/api/v1/collections/{uuid.uuid4()}/papers", json={"paper_id": str(uuid.uuid4())}
        )
        assert resp.status_code == 404

    async def test_remove_paper_from_collection(self, api_client, engine):
        """DELETE /api/v1/collections/{id}/papers/{paper_id} removes paper."""
        session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        async with session_factory() as sess:
            paper = await _create_paper(sess, arxiv_id="2401.99002")
            await sess.commit()
            paper_id = str(paper.id)

        resp = await api_client.post("/api/v1/collections", json={"name": "Remove Test"})
        cid = resp.json()["id"]

        # Add then remove
        await api_client.post(f"/api/v1/collections/{cid}/papers", json={"paper_id": paper_id})

        resp = await api_client.delete(f"/api/v1/collections/{cid}/papers/{paper_id}")
        assert resp.status_code == 200
        assert resp.json()["message"] == "Paper removed from collection"

    async def test_remove_paper_not_in_collection(self, api_client, engine):
        """DELETE /api/v1/collections/{id}/papers/{paper_id} returns 404 if not in collection."""
        session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        async with session_factory() as sess:
            paper = await _create_paper(sess, arxiv_id="2401.99003")
            await sess.commit()
            paper_id = str(paper.id)

        resp = await api_client.post("/api/v1/collections", json={"name": "Empty"})
        cid = resp.json()["id"]

        resp = await api_client.delete(f"/api/v1/collections/{cid}/papers/{paper_id}")
        assert resp.status_code == 404
