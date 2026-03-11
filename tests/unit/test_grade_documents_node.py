"""Tests for the document grading node (S6.4).

Tests cover:
- ainvoke_grade_documents_step: LLM-based binary relevance grading
- continue_after_grading: conditional edge routing (generate / rewrite)
- GRADING_PROMPT: prompt template validation
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage
from src.services.agents.context import AgentContext
from src.services.agents.models import GradeDocuments, GradingResult, SourceItem
from src.services.agents.nodes.grade_documents_node import (
    GRADING_PROMPT,
    ainvoke_grade_documents_step,
    continue_after_grading,
)
from src.services.agents.state import AgentState

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_source(
    arxiv_id: str = "1706.03762",
    title: str = "Attention Is All You Need",
    chunk: str = "Transformers use self-attention.",
) -> SourceItem:
    return SourceItem(
        arxiv_id=arxiv_id,
        title=title,
        authors=["Vaswani et al."],
        url=f"https://arxiv.org/abs/{arxiv_id}",
        relevance_score=0.9,
        chunk_text=chunk,
    )


@pytest.fixture
def source_a() -> SourceItem:
    return _make_source("1706.03762", "Attention Is All You Need", "Transformers use self-attention mechanisms.")


@pytest.fixture
def source_b() -> SourceItem:
    return _make_source("1810.04805", "BERT", "BERT is a bidirectional transformer model.")


@pytest.fixture
def source_c() -> SourceItem:
    return _make_source("2005.14165", "GPT-3", "GPT-3 is a large language model.")


def _make_state(sources: list[SourceItem], query: str = "How do transformers work?", retrieval_attempts: int = 1) -> AgentState:
    return AgentState(
        messages=[HumanMessage(content=query)],
        original_query=query,
        sources=sources,
        retrieval_attempts=retrieval_attempts,
        grading_results=[],
        relevant_sources=[],
    )


def _make_context(llm_responses: list[GradeDocuments] | Exception | None = None) -> tuple[AgentContext, MagicMock]:
    """Create context with mock LLM that returns given responses sequentially."""
    provider = MagicMock()
    mock_model = MagicMock()
    mock_structured = AsyncMock()

    if isinstance(llm_responses, Exception) or llm_responses is not None:
        mock_structured.ainvoke = AsyncMock(side_effect=llm_responses)
    else:
        mock_structured.ainvoke = AsyncMock(return_value=GradeDocuments(binary_score="yes", reasoning="relevant"))

    mock_model.with_structured_output.return_value = mock_structured
    provider.get_langchain_model.return_value = mock_model

    ctx = AgentContext(llm_provider=provider, model_name="test-model")
    return ctx, provider


# ---------------------------------------------------------------------------
# Tests: ainvoke_grade_documents_step — FR-1: All relevant
# ---------------------------------------------------------------------------


class TestGradeDocumentsAllRelevant:
    @pytest.mark.asyncio
    async def test_all_sources_graded_yes(self, source_a, source_b):
        """All sources graded 'yes' → all in relevant_sources."""
        responses = [
            GradeDocuments(binary_score="yes", reasoning="Directly about transformers"),
            GradeDocuments(binary_score="yes", reasoning="BERT is relevant to NLP"),
        ]
        ctx, _ = _make_context(responses)
        state = _make_state([source_a, source_b])

        result = await ainvoke_grade_documents_step(state, ctx)

        assert len(result["relevant_sources"]) == 2
        assert len(result["grading_results"]) == 2
        assert all(gr.is_relevant for gr in result["grading_results"])


# ---------------------------------------------------------------------------
# Tests: ainvoke_grade_documents_step — FR-1: Mixed relevance
# ---------------------------------------------------------------------------


class TestGradeDocumentsMixedRelevance:
    @pytest.mark.asyncio
    async def test_mixed_yes_no(self, source_a, source_b, source_c):
        """Some 'yes', some 'no' → only 'yes' in relevant_sources."""
        responses = [
            GradeDocuments(binary_score="yes", reasoning="Relevant"),
            GradeDocuments(binary_score="no", reasoning="Not relevant"),
            GradeDocuments(binary_score="yes", reasoning="Relevant"),
        ]
        ctx, _ = _make_context(responses)
        state = _make_state([source_a, source_b, source_c])

        result = await ainvoke_grade_documents_step(state, ctx)

        assert len(result["relevant_sources"]) == 2
        assert result["relevant_sources"][0].arxiv_id == "1706.03762"
        assert result["relevant_sources"][1].arxiv_id == "2005.14165"
        assert len(result["grading_results"]) == 3
        assert result["grading_results"][0].is_relevant is True
        assert result["grading_results"][1].is_relevant is False
        assert result["grading_results"][2].is_relevant is True


# ---------------------------------------------------------------------------
# Tests: ainvoke_grade_documents_step — FR-1: None relevant
# ---------------------------------------------------------------------------


class TestGradeDocumentsNoneRelevant:
    @pytest.mark.asyncio
    async def test_all_no(self, source_a, source_b):
        """All 'no' → relevant_sources empty."""
        responses = [
            GradeDocuments(binary_score="no", reasoning="Off-topic"),
            GradeDocuments(binary_score="no", reasoning="Not related"),
        ]
        ctx, _ = _make_context(responses)
        state = _make_state([source_a, source_b])

        result = await ainvoke_grade_documents_step(state, ctx)

        assert len(result["relevant_sources"]) == 0
        assert len(result["grading_results"]) == 2
        assert all(not gr.is_relevant for gr in result["grading_results"])


# ---------------------------------------------------------------------------
# Tests: ainvoke_grade_documents_step — FR-1: Empty sources
# ---------------------------------------------------------------------------


class TestGradeDocumentsEmptySources:
    @pytest.mark.asyncio
    async def test_empty_sources_no_llm_calls(self):
        """No sources → empty results, no LLM calls."""
        ctx, provider = _make_context([])
        state = _make_state([])

        result = await ainvoke_grade_documents_step(state, ctx)

        assert result["relevant_sources"] == []
        assert result["grading_results"] == []
        # LLM should NOT have been called
        provider.get_langchain_model.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: ainvoke_grade_documents_step — FR-1: LLM failure
# ---------------------------------------------------------------------------


class TestGradeDocumentsLLMFailure:
    @pytest.mark.asyncio
    async def test_llm_failure_marks_not_relevant(self, source_a, source_b):
        """LLM raises for one doc → that doc marked not relevant, others still graded."""
        provider = MagicMock()
        mock_model = MagicMock()
        mock_structured = AsyncMock()
        # First call succeeds, second call fails
        mock_structured.ainvoke = AsyncMock(
            side_effect=[
                GradeDocuments(binary_score="yes", reasoning="Relevant"),
                RuntimeError("LLM timeout"),
            ]
        )
        mock_model.with_structured_output.return_value = mock_structured
        provider.get_langchain_model.return_value = mock_model
        ctx = AgentContext(llm_provider=provider, model_name="test-model")

        state = _make_state([source_a, source_b])
        result = await ainvoke_grade_documents_step(state, ctx)

        assert len(result["grading_results"]) == 2
        # First doc: graded successfully as relevant
        assert result["grading_results"][0].is_relevant is True
        # Second doc: LLM failed → not relevant
        assert result["grading_results"][1].is_relevant is False
        reasoning = result["grading_results"][1].reasoning.lower()
        assert "error" in reasoning or "fail" in reasoning
        # Only first doc in relevant_sources
        assert len(result["relevant_sources"]) == 1
        assert result["relevant_sources"][0].arxiv_id == "1706.03762"


# ---------------------------------------------------------------------------
# Tests: continue_after_grading — FR-3
# ---------------------------------------------------------------------------


class TestContinueAfterGrading:
    def test_has_relevant_returns_generate(self, source_a):
        """relevant_sources non-empty → 'generate'."""
        state = AgentState(relevant_sources=[source_a], retrieval_attempts=1)
        ctx = AgentContext(llm_provider=MagicMock(), max_retrieval_attempts=3)

        result = continue_after_grading(state, ctx)
        assert result == "generate"

    def test_no_relevant_returns_rewrite(self):
        """relevant_sources empty, attempts < max → 'rewrite'."""
        state = AgentState(relevant_sources=[], retrieval_attempts=1)
        ctx = AgentContext(llm_provider=MagicMock(), max_retrieval_attempts=3)

        result = continue_after_grading(state, ctx)
        assert result == "rewrite"

    def test_exhausted_retries_returns_generate(self):
        """relevant_sources empty, attempts >= max → 'generate' (force)."""
        state = AgentState(relevant_sources=[], retrieval_attempts=3)
        ctx = AgentContext(llm_provider=MagicMock(), max_retrieval_attempts=3)

        result = continue_after_grading(state, ctx)
        assert result == "generate"

    def test_missing_retrieval_attempts_defaults_to_zero(self):
        """Missing retrieval_attempts → defaults to 0, routes to 'rewrite'."""
        state = AgentState(relevant_sources=[])
        ctx = AgentContext(llm_provider=MagicMock(), max_retrieval_attempts=3)

        result = continue_after_grading(state, ctx)
        assert result == "rewrite"

    def test_missing_relevant_sources_defaults_empty(self):
        """Missing relevant_sources → defaults to empty, routes to 'rewrite'."""
        state = AgentState(retrieval_attempts=1)
        ctx = AgentContext(llm_provider=MagicMock(), max_retrieval_attempts=3)

        result = continue_after_grading(state, ctx)
        assert result == "rewrite"


# ---------------------------------------------------------------------------
# Tests: GRADING_PROMPT — FR-2
# ---------------------------------------------------------------------------


class TestGradingPrompt:
    def test_prompt_is_string(self):
        assert isinstance(GRADING_PROMPT, str)

    def test_prompt_has_query_placeholder(self):
        assert "{query}" in GRADING_PROMPT

    def test_prompt_has_document_placeholder(self):
        assert "{document}" in GRADING_PROMPT

    def test_prompt_formats_with_query_and_document(self):
        formatted = GRADING_PROMPT.format(query="How does BERT work?", document="BERT uses bidirectional attention.")
        assert "How does BERT work?" in formatted
        assert "BERT uses bidirectional attention." in formatted
        assert "{query}" not in formatted
        assert "{document}" not in formatted

    def test_prompt_mentions_relevance(self):
        lower = GRADING_PROMPT.lower()
        assert "relevant" in lower or "relevance" in lower


# ---------------------------------------------------------------------------
# Tests: GradingResult structure — FR-4
# ---------------------------------------------------------------------------


class TestGradingResultStructure:
    @pytest.mark.asyncio
    async def test_grading_result_has_all_fields(self, source_a):
        """Each GradingResult has document_id, is_relevant, score, reasoning."""
        responses = [GradeDocuments(binary_score="yes", reasoning="Highly relevant")]
        ctx, _ = _make_context(responses)
        state = _make_state([source_a])

        result = await ainvoke_grade_documents_step(state, ctx)

        gr = result["grading_results"][0]
        assert isinstance(gr, GradingResult)
        assert gr.document_id == "1706.03762"
        assert gr.is_relevant is True
        assert gr.score == 1.0
        assert gr.reasoning == "Highly relevant"

    @pytest.mark.asyncio
    async def test_no_grade_result_score_zero(self, source_a):
        """'no' grade → score=0.0, is_relevant=False."""
        responses = [GradeDocuments(binary_score="no", reasoning="Not related")]
        ctx, _ = _make_context(responses)
        state = _make_state([source_a])

        result = await ainvoke_grade_documents_step(state, ctx)

        gr = result["grading_results"][0]
        assert gr.is_relevant is False
        assert gr.score == 0.0
        assert gr.reasoning == "Not related"
