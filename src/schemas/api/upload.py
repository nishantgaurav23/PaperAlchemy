"""Pydantic schemas for the PDF upload endpoint."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response from PDF upload endpoint."""

    paper_id: uuid.UUID = Field(description="UUID of the created paper record")
    arxiv_id: str = Field(description="Generated ID for the uploaded paper (upload_<uuid>)")
    title: str = Field(description="Extracted or derived paper title")
    authors: list[str] = Field(default_factory=list, description="Extracted authors")
    abstract: str = Field(default="", description="Extracted abstract or first 500 chars")
    page_count: int = Field(default=0, description="Number of pages in the PDF")
    chunks_indexed: int = Field(default=0, description="Number of chunks indexed in OpenSearch")
    parsing_status: str = Field(default="success", description="success or partial")
    indexing_status: str = Field(default="success", description="success, partial, or failed")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal issues encountered")
    message: str = Field(default="", description="Human-readable status summary")
