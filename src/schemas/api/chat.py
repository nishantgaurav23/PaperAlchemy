"""Request/response schemas for the /chat endpoint (S7.3).

Supports both streaming (SSE) and non-streaming (JSON) modes with
session management and follow-up detection.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from src.services.rag.models import SourceReference


class ChatRequest(BaseModel):
    """Chat message request body."""

    session_id: str = Field(..., min_length=1, max_length=128)
    query: str = Field(..., min_length=1, max_length=500)
    stream: bool = True
    top_k: int | None = Field(None, gt=0)
    categories: list[str] | None = None
    temperature: float | None = Field(None, ge=0.0, le=2.0)

    @field_validator("query")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        v = v.strip()
        if not v:
            msg = "Query must not be empty or whitespace-only"
            raise ValueError(msg)
        return v


class ChatResponse(BaseModel):
    """Non-streaming chat response with answer, sources, and follow-up info."""

    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    session_id: str
    is_follow_up: bool
    rewritten_query: str | None = None
    query: str


class ChatMessageOut(BaseModel):
    """A single chat message for history responses."""

    role: str
    content: str
    timestamp: str | None = None


class SessionHistoryResponse(BaseModel):
    """Response for GET /chat/sessions/{id}/history."""

    session_id: str
    messages: list[ChatMessageOut] = Field(default_factory=list)


class SessionClearResponse(BaseModel):
    """Response for DELETE /chat/sessions/{id}."""

    session_id: str
    cleared: bool
