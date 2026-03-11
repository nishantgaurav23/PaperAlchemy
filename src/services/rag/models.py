"""RAG response models (S5.2).

Structured output types for the RAG chain: answer with citations,
source references, and metadata about the retrieval + LLM stages.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    """A single cited paper source."""

    index: int
    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    arxiv_url: str
    chunk_text: str = ""
    score: float = 0.0


class RetrievalMetadata(BaseModel):
    """Metadata about the retrieval pipeline execution."""

    stages_executed: list[str] = Field(default_factory=list)
    total_candidates: int = 0
    timings: dict[str, float] = Field(default_factory=dict)
    expanded_queries: list[str] = Field(default_factory=list)


class LLMMetadata(BaseModel):
    """Metadata about the LLM generation."""

    provider: str = ""
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None


class RAGResponse(BaseModel):
    """Full RAG response with answer, sources, and metadata."""

    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    query: str
    retrieval_metadata: RetrievalMetadata = Field(default_factory=RetrievalMetadata)
    llm_metadata: LLMMetadata = Field(default_factory=LLMMetadata)
