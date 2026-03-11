"""Pydantic models for PDF parsing results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Section(BaseModel):
    """A section extracted from a PDF document."""

    title: str = Field(description="Section heading text")
    content: str = Field(default="", description="Section body text")
    level: int = Field(default=1, ge=1, le=6, description="Heading level (1-6)")


class PDFContent(BaseModel):
    """Structured content extracted from a PDF file."""

    raw_text: str = Field(default="", description="Full plain-text content")
    sections: list[Section] = Field(default_factory=list, description="Extracted sections")
    tables: list[str] = Field(default_factory=list, description="Tables as text")
    figures: list[str] = Field(default_factory=list, description="Figure captions")
    page_count: int = Field(default=0, ge=0, description="Number of pages")
    parser_used: str = Field(default="docling", description="Parser backend name")
    parser_time_seconds: float = Field(default=0.0, ge=0.0, description="Parse duration")
