"""Structured output models for agent LLM responses.

Pydantic models used with `llm.with_structured_output(Model)` for validated,
type-safe LLM outputs across all agent nodes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GuardrailScoring(BaseModel):
    """Domain relevance score returned by the guardrail node."""

    score: int = Field(ge=0, le=100)
    reason: str


class GradeDocuments(BaseModel):
    """Binary relevance grade for a retrieved document."""

    binary_score: Literal["yes", "no"]
    reasoning: str = ""


class GradingResult(BaseModel):
    """Detailed grading result for a single document."""

    document_id: str
    is_relevant: bool
    score: float = 0.0
    reasoning: str = ""


class SourceItem(BaseModel):
    """A retrieved source document with metadata for citation."""

    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    url: str
    relevance_score: float = 0.0
    chunk_text: str = ""


class RoutingDecision(BaseModel):
    """Routing decision made by the agent orchestrator."""

    route: Literal["retrieve", "out_of_scope", "generate_answer", "rewrite_query"]
    reason: str = ""
