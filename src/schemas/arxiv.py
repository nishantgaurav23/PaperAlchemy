"""Pydantic schema for arXiv paper metadata."""

from __future__ import annotations

from pydantic import BaseModel


class ArxivPaper(BaseModel):
    """Parsed arXiv paper metadata from Atom API response."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published_date: str
    updated_date: str | None = None
    pdf_url: str

    @property
    def arxiv_url(self) -> str:
        """Return the arXiv abstract page URL."""
        return f"https://arxiv.org/abs/{self.arxiv_id}"
