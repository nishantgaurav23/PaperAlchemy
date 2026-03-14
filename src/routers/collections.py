"""Collection CRUD and paper management endpoints."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query

from src.dependency import CollectionRepoDep, SessionDep
from src.schemas.collection import (
    CollectionCreate,
    CollectionDetailResponse,
    CollectionPaperAction,
    CollectionResponse,
    CollectionUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collections", tags=["collections"])


@router.get("", response_model=list[CollectionResponse])
async def list_collections(
    collection_repo: CollectionRepoDep,
    session: SessionDep,
    user_id: uuid.UUID | None = Query(default=None, description="Filter by user ID"),  # noqa: B008
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[CollectionResponse]:
    """List collections with optional user_id filter and pagination."""
    collections = await collection_repo.list_all(user_id=user_id, limit=limit, offset=offset)
    return [
        CollectionResponse(
            id=c.id,
            name=c.name,
            description=c.description,
            user_id=c.user_id,
            paper_count=len(c.papers),
            created_at=c.created_at,
            updated_at=c.updated_at,
        )
        for c in collections
    ]


@router.post("", response_model=CollectionResponse, status_code=201)
async def create_collection(
    body: CollectionCreate,
    collection_repo: CollectionRepoDep,
    session: SessionDep,
) -> CollectionResponse:
    """Create a new collection."""
    collection = await collection_repo.create(body)
    response = CollectionResponse(
        id=collection.id,
        name=collection.name,
        description=collection.description,
        user_id=collection.user_id,
        paper_count=0,
        created_at=collection.created_at,
        updated_at=collection.updated_at,
    )
    await session.commit()
    return response


@router.get("/{collection_id}", response_model=CollectionDetailResponse)
async def get_collection(
    collection_id: uuid.UUID,
    collection_repo: CollectionRepoDep,
    session: SessionDep,
) -> CollectionDetailResponse:
    """Get a collection with its papers."""
    collection = await collection_repo.get_by_id(collection_id)
    if collection is None:
        raise HTTPException(status_code=404, detail=f"Collection {collection_id} not found")
    return CollectionDetailResponse.model_validate(collection)


@router.put("/{collection_id}", response_model=CollectionResponse)
async def update_collection(
    collection_id: uuid.UUID,
    body: CollectionUpdate,
    collection_repo: CollectionRepoDep,
    session: SessionDep,
) -> CollectionResponse:
    """Update a collection's name or description."""
    collection = await collection_repo.update(collection_id, body)
    if collection is None:
        raise HTTPException(status_code=404, detail=f"Collection {collection_id} not found")
    response = CollectionResponse(
        id=collection.id,
        name=collection.name,
        description=collection.description,
        user_id=collection.user_id,
        paper_count=len(collection.papers),
        created_at=collection.created_at,
        updated_at=collection.updated_at,
    )
    await session.commit()
    return response


@router.delete("/{collection_id}", status_code=204)
async def delete_collection(
    collection_id: uuid.UUID,
    collection_repo: CollectionRepoDep,
    session: SessionDep,
) -> None:
    """Delete a collection."""
    deleted = await collection_repo.delete(collection_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Collection {collection_id} not found")
    await session.commit()


@router.post("/{collection_id}/papers")
async def add_paper_to_collection(
    collection_id: uuid.UUID,
    body: CollectionPaperAction,
    collection_repo: CollectionRepoDep,
    session: SessionDep,
) -> dict:
    """Add a paper to a collection."""
    result = await collection_repo.add_paper(collection_id, body.paper_id)
    if not result:
        raise HTTPException(status_code=404, detail="Collection or paper not found")
    await session.commit()
    return {"message": "Paper added to collection"}


@router.delete("/{collection_id}/papers/{paper_id}")
async def remove_paper_from_collection(
    collection_id: uuid.UUID,
    paper_id: uuid.UUID,
    collection_repo: CollectionRepoDep,
    session: SessionDep,
) -> dict:
    """Remove a paper from a collection."""
    result = await collection_repo.remove_paper(collection_id, paper_id)
    if not result:
        raise HTTPException(status_code=404, detail="Paper not found in collection")
    await session.commit()
    return {"message": "Paper removed from collection"}
