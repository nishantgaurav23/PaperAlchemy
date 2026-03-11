"""Request/response schemas for the /ask endpoint (S5.3).

Supports both streaming (SSE) and non-streaming (JSON) modes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from src.services.rag.models import LLMMetadata, RetrievalMetadata, SourceReference


class AskRequest(BaseModel):
    """Research question request body."""

    query: str = Field(..., min_length=1, max_length=500)
    top_k: int | None = Field(None, gt=0)
    categories: list[str] | None = None
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    stream: bool = True

    @field_validator("query")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        v = v.strip()
        if not v:
            msg = "Query must not be empty or whitespace-only"
            raise ValueError(msg)
        return v


class AskResponse(BaseModel):
    """Non-streaming response with answer, sources, and metadata."""

    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    query: str
    retrieval_metadata: RetrievalMetadata = Field(default_factory=RetrievalMetadata)
    llm_metadata: LLMMetadata = Field(default_factory=LLMMetadata)
