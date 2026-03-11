"""Pydantic schemas for the Jina AI Embeddings API."""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class JinaEmbeddingRequest(BaseModel):
    """Request payload for Jina /embeddings endpoint."""

    model: str = "jina-embeddings-v3"
    task: str = "retrieval.passage"
    dimensions: int = 1024
    input: list[str]

    @field_validator("input")
    @classmethod
    def input_not_empty(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("input must contain at least one text")
        return v


class JinaEmbeddingData(BaseModel):
    """Single embedding item in the Jina response."""

    object: str = "embedding"
    index: int
    embedding: list[float]


class JinaUsage(BaseModel):
    """Token usage info from the Jina response."""

    total_tokens: int


class JinaEmbeddingResponse(BaseModel):
    """Response from Jina /embeddings endpoint."""

    model: str
    data: list[JinaEmbeddingData]
    usage: JinaUsage
