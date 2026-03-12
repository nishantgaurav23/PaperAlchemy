"""Pydantic schemas for paper analysis endpoints."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


class PaperSummary(BaseModel):
    """Structured AI-generated paper summary."""

    objective: str = Field(description="Main research objective or question")
    method: str = Field(description="Methodology or approach used")
    key_findings: list[str] = Field(description="Key findings or results")
    contribution: str = Field(description="Main contribution to the field")
    limitations: str = Field(description="Limitations or caveats")


class SummaryResponse(BaseModel):
    """Response from paper summary endpoint."""

    paper_id: uuid.UUID = Field(description="UUID of the paper")
    title: str = Field(description="Paper title")
    summary: PaperSummary = Field(description="Structured summary")
    model: str = Field(description="LLM model used")
    provider: str = Field(description="LLM provider used")
    latency_ms: float | None = Field(default=None, description="Generation latency in milliseconds")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal issues encountered")


class PaperHighlights(BaseModel):
    """Structured key highlights extracted from a paper."""

    novel_contributions: list[str] = Field(description="Novel contributions (1-5 items)")
    important_findings: list[str] = Field(description="Key results and discoveries (1-5 items)")
    practical_implications: list[str] = Field(description="Real-world applications and impact (1-5 items)")
    limitations: list[str] = Field(default_factory=list, description="Noted limitations and caveats (0-3 items)")
    keywords: list[str] = Field(description="Extracted key terms/topics (3-10 items)")


class HighlightsResponse(BaseModel):
    """Response from paper highlights endpoint."""

    paper_id: uuid.UUID = Field(description="UUID of the paper")
    highlights: PaperHighlights = Field(description="Structured highlights")
    model: str = Field(description="LLM model used")
    provider: str = Field(description="LLM provider used")
    latency_ms: float | None = Field(default=None, description="Generation latency in milliseconds")
    warning: str | None = Field(default=None, description="Non-fatal issue (e.g. abstract-only, malformed output)")


# ---------------------------------------------------------------------------
# S8.4: Methodology & Findings Analysis
# ---------------------------------------------------------------------------


class DatasetInfo(BaseModel):
    """Information about a dataset used in a paper."""

    name: str = Field(description="Dataset name")
    description: str = Field(default="", description="Brief description of the dataset")
    size: str | None = Field(default=None, description="Dataset size (e.g., '10k samples', '1.5M images')")


class ResultEntry(BaseModel):
    """A single key result with metric and value."""

    metric: str = Field(description="Metric name (e.g., BLEU, accuracy, F1)")
    value: str = Field(description="Metric value (e.g., '28.4', '95.2%')")
    context: str = Field(default="", description="Context for the result (e.g., 'on WMT 2014 EN-DE')")


class MethodologyAnalysis(BaseModel):
    """Structured methodology and findings analysis of a paper."""

    research_design: str = Field(description="Type of study and overall approach (1-3 sentences)")
    datasets: list[DatasetInfo] = Field(default_factory=list, description="Datasets used in the study")
    baselines: list[str] = Field(default_factory=list, description="Baseline methods/models compared against")
    key_results: list[ResultEntry] = Field(default_factory=list, description="Key results with metrics")
    statistical_significance: str | None = Field(default=None, description="Statistical tests, confidence intervals, p-values")
    reproducibility_notes: str | None = Field(default=None, description="Code availability, hyperparameters, compute resources")


class MethodologyResponse(BaseModel):
    """Response from methodology analysis endpoint."""

    paper_id: uuid.UUID = Field(description="UUID of the paper")
    analysis: MethodologyAnalysis = Field(description="Structured methodology analysis")
    model: str = Field(description="LLM model used")
    provider: str = Field(description="LLM provider used")
    latency_ms: float | None = Field(default=None, description="Generation latency in milliseconds")
    warning: str | None = Field(default=None, description="Non-fatal issue (e.g. abstract-only, malformed output)")


# ---------------------------------------------------------------------------
# S8.5: Paper Comparison
# ---------------------------------------------------------------------------


class ComparedPaper(BaseModel):
    """Metadata for a paper included in a comparison."""

    id: uuid.UUID = Field(description="Paper UUID")
    title: str = Field(description="Paper title")
    authors: list[str] = Field(default_factory=list, description="Paper authors")


class PaperComparison(BaseModel):
    """Structured side-by-side comparison of 2+ papers."""

    papers: list[ComparedPaper] = Field(description="Metadata for each compared paper")
    methods_comparison: str = Field(description="How the approaches/methodologies differ and overlap")
    results_comparison: str = Field(description="Comparative analysis of results and findings")
    contributions_comparison: str = Field(description="What each paper uniquely contributes")
    limitations_comparison: str = Field(description="Comparative limitations and gaps")
    common_themes: list[str] = Field(description="Shared themes, topics, or approaches (1-5 items)")
    key_differences: list[str] = Field(description="Most notable differences (1-5 items)")
    verdict: str = Field(description="Brief overall synthesis")


class ComparisonRequest(BaseModel):
    """Request body for paper comparison endpoint."""

    paper_ids: list[uuid.UUID] = Field(min_length=2, max_length=5, description="UUIDs of papers to compare (2-5)")


class ComparisonResponse(BaseModel):
    """Response from paper comparison endpoint."""

    paper_ids: list[uuid.UUID] = Field(description="UUIDs of compared papers")
    comparison: PaperComparison = Field(description="Structured comparison")
    model: str = Field(description="LLM model used")
    provider: str = Field(description="LLM provider used")
    latency_ms: float | None = Field(default=None, description="Generation latency in milliseconds")
    warning: str | None = Field(default=None, description="Non-fatal issue (e.g. malformed output)")
