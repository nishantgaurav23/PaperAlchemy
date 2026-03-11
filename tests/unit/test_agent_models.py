"""Tests for agent structured output models (FR-3)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.services.agents.models import (
    GradeDocuments,
    GradingResult,
    GuardrailScoring,
    RoutingDecision,
    SourceItem,
)


class TestGuardrailScoring:
    def test_valid(self):
        gs = GuardrailScoring(score=75, reason="relevant to ML research")
        assert gs.score == 75
        assert gs.reason == "relevant to ML research"

    def test_boundary_zero(self):
        gs = GuardrailScoring(score=0, reason="completely off-topic")
        assert gs.score == 0

    def test_boundary_hundred(self):
        gs = GuardrailScoring(score=100, reason="perfect match")
        assert gs.score == 100

    def test_score_below_zero_raises(self):
        with pytest.raises(ValidationError):
            GuardrailScoring(score=-1, reason="invalid")

    def test_score_above_hundred_raises(self):
        with pytest.raises(ValidationError):
            GuardrailScoring(score=101, reason="invalid")

    def test_missing_reason_raises(self):
        with pytest.raises(ValidationError):
            GuardrailScoring(score=50)  # type: ignore[call-arg]


class TestGradeDocuments:
    def test_valid_yes(self):
        gd = GradeDocuments(binary_score="yes", reasoning="relevant content")
        assert gd.binary_score == "yes"
        assert gd.reasoning == "relevant content"

    def test_valid_no(self):
        gd = GradeDocuments(binary_score="no")
        assert gd.binary_score == "no"
        assert gd.reasoning == ""

    def test_invalid_literal_raises(self):
        with pytest.raises(ValidationError):
            GradeDocuments(binary_score="maybe")  # type: ignore[arg-type]

    def test_default_reasoning(self):
        gd = GradeDocuments(binary_score="yes")
        assert gd.reasoning == ""


class TestGradingResult:
    def test_full_fields(self):
        gr = GradingResult(
            document_id="doc-123",
            is_relevant=True,
            score=0.85,
            reasoning="Highly relevant to the query",
        )
        assert gr.document_id == "doc-123"
        assert gr.is_relevant is True
        assert gr.score == 0.85
        assert gr.reasoning == "Highly relevant to the query"

    def test_defaults(self):
        gr = GradingResult(document_id="doc-456", is_relevant=False)
        assert gr.score == 0.0
        assert gr.reasoning == ""


class TestSourceItem:
    def test_minimal_fields(self):
        si = SourceItem(arxiv_id="1706.03762", title="Attention Is All You Need", url="https://arxiv.org/abs/1706.03762")
        assert si.arxiv_id == "1706.03762"
        assert si.title == "Attention Is All You Need"
        assert si.authors == []
        assert si.relevance_score == 0.0
        assert si.chunk_text == ""

    def test_full_fields(self):
        si = SourceItem(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            authors=["Vaswani", "Shazeer"],
            url="https://arxiv.org/abs/1706.03762",
            relevance_score=0.95,
            chunk_text="The dominant sequence transduction models...",
        )
        assert si.authors == ["Vaswani", "Shazeer"]
        assert si.relevance_score == 0.95

    def test_serialization(self):
        si = SourceItem(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            url="https://arxiv.org/abs/1706.03762",
        )
        d = si.model_dump()
        assert isinstance(d, dict)
        assert d["arxiv_id"] == "1706.03762"
        assert d["title"] == "Attention Is All You Need"
        assert "authors" in d
        assert "relevance_score" in d


class TestRoutingDecision:
    @pytest.mark.parametrize(
        "route",
        ["retrieve", "out_of_scope", "generate_answer", "rewrite_query"],
    )
    def test_valid_routes(self, route: str):
        rd = RoutingDecision(route=route, reason="test")  # type: ignore[arg-type]
        assert rd.route == route

    def test_invalid_route_raises(self):
        with pytest.raises(ValidationError):
            RoutingDecision(route="invalid_route", reason="bad")  # type: ignore[arg-type]

    def test_default_reason(self):
        rd = RoutingDecision(route="retrieve")  # type: ignore[arg-type]
        assert rd.reason == ""
