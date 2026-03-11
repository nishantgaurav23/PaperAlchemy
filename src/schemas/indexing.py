"""Pydantic models for text chunking and indexing."""

from __future__ import annotations

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """Metadata describing a chunk's position and context within a paper."""

    chunk_index: int
    start_char: int
    end_char: int
    word_count: int
    overlap_with_previous: int
    overlap_with_next: int
    section_title: str | None = None


class TextChunk(BaseModel):
    """A chunk of text with its metadata and paper identifiers."""

    text: str
    metadata: ChunkMetadata
    arxiv_id: str
    paper_id: str


class ChildChunk(BaseModel):
    """A child chunk linking back to its parent chunk."""

    text: str
    metadata: ChunkMetadata
    arxiv_id: str
    paper_id: str
    parent_chunk_index: int


class ParentChildResult(BaseModel):
    """Result of parent-child chunking: parents and their children."""

    parents: list[TextChunk] = []
    children: list[ChildChunk] = []
