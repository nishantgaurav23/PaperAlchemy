"""Tests for the answer generation node (S6.6).

TDD: Red phase — these tests define the expected behavior before implementation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from src.services.agents.context import AgentContext
from src.services.agents.models import SourceItem
from src.services.agents.nodes.generate_answer_node import (
    GENERATION_PROMPT,
    NO_SOURCES_MESSAGE,
    ainvoke_generate_answer_step,
    build_generation_prompt,
    source_items_to_references,
)
from src.services.agents.state import AgentState
from src.services.rag.models import SourceReference

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_source(
    arxiv_id: str = "2301.00001",
    title: str = "Attention Is All You Need",
    authors: list[str] | None = None,
    url: str = "https://arxiv.org/abs/2301.00001",
    chunk_text: str = "Transformers use self-attention mechanisms.",
    relevance_score: float = 0.9,
) -> SourceItem:
    return SourceItem(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors or ["Vaswani", "Shazeer", "Parmar"],
        url=url,
        chunk_text=chunk_text,
        relevance_score=relevance_score,
    )


def _make_context(llm_response: str = "Answer with [1] citation.") -> AgentContext:
    """Create an AgentContext with a mocked LLM provider."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=llm_response))

    mock_provider = MagicMock()
    mock_provider.get_langchain_model.return_value = mock_llm

    return AgentContext(
        llm_provider=mock_provider,
        model_name="test-model",
        temperature=0.7,
    )


def _make_state(
    query: str = "What are transformers?",
    relevant_sources: list[SourceItem] | None = None,
    rewritten_query: str | None = None,
) -> AgentState:
    return AgentState(
        messages=[HumanMessage(content=query)],
        original_query=query,
        rewritten_query=rewritten_query,
        retrieval_attempts=1,
        guardrail_result=None,
        routing_decision=None,
        sources=relevant_sources or [],
        grading_results=[],
        relevant_sources=relevant_sources or [],
        metadata={},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildGenerationPrompt:
    """FR-1: Prompt construction."""

    def test_prompt_includes_sources(self):
        sources = [
            _make_source(arxiv_id="2301.00001", title="Paper A", chunk_text="Chunk A content"),
            _make_source(arxiv_id="2302.00002", title="Paper B", chunk_text="Chunk B content"),
        ]
        prompt = build_generation_prompt("What are transformers?", sources)

        assert "What are transformers?" in prompt
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "Chunk A content" in prompt
        assert "Chunk B content" in prompt
        assert "Paper A" in prompt
        assert "Paper B" in prompt

    def test_prompt_has_citation_instructions(self):
        sources = [_make_source()]
        prompt = build_generation_prompt("query", sources)
        # Should instruct LLM to use inline citations
        assert "[" in prompt and "]" in prompt

    def test_prompt_uses_generation_template(self):
        """GENERATION_PROMPT constant should be a non-empty string."""
        assert isinstance(GENERATION_PROMPT, str)
        assert len(GENERATION_PROMPT) > 50


class TestSourceItemsToReferences:
    """FR-3: SourceItem -> SourceReference conversion."""

    def test_converts_sources(self):
        sources = [
            _make_source(arxiv_id="2301.00001", title="Paper A"),
            _make_source(arxiv_id="2302.00002", title="Paper B"),
        ]
        refs = source_items_to_references(sources)

        assert len(refs) == 2
        assert isinstance(refs[0], SourceReference)
        assert refs[0].index == 1
        assert refs[0].arxiv_id == "2301.00001"
        assert refs[0].title == "Paper A"
        assert refs[1].index == 2
        assert refs[1].arxiv_id == "2302.00002"

    def test_empty_sources(self):
        refs = source_items_to_references([])
        assert refs == []

    def test_preserves_authors_and_url(self):
        source = _make_source(
            authors=["Alice", "Bob"],
            url="https://arxiv.org/abs/2301.00001",
        )
        refs = source_items_to_references([source])
        assert refs[0].authors == ["Alice", "Bob"]
        assert refs[0].arxiv_url == "https://arxiv.org/abs/2301.00001"


@pytest.mark.asyncio
class TestGenerateAnswerWithSources:
    """FR-2 + FR-3 + FR-4: Full generation flow."""

    async def test_generates_answer_with_citations(self):
        sources = [
            _make_source(arxiv_id="2301.00001", title="Paper A"),
            _make_source(arxiv_id="2302.00002", title="Paper B"),
        ]
        llm_answer = "Transformers [1] use self-attention [2] for sequence modeling."
        context = _make_context(llm_response=llm_answer)
        state = _make_state(relevant_sources=sources)

        result = await ainvoke_generate_answer_step(state, context)

        # Should return messages with AIMessage
        assert "messages" in result
        messages = result["messages"]
        assert len(messages) == 1
        ai_msg = messages[0]
        assert isinstance(ai_msg, AIMessage)
        # Answer should contain the text and sources section
        assert "Transformers" in ai_msg.content
        assert "**Sources:**" in ai_msg.content

    async def test_citation_metadata_stored(self):
        sources = [_make_source()]
        llm_answer = "Answer with [1] citation."
        context = _make_context(llm_response=llm_answer)
        state = _make_state(relevant_sources=sources)

        result = await ainvoke_generate_answer_step(state, context)

        assert "metadata" in result
        meta = result["metadata"]
        assert "citation_validation" in meta
        assert meta["citation_validation"]["is_valid"] is True

    async def test_llm_called_with_correct_model(self):
        sources = [_make_source()]
        context = _make_context(llm_response="Answer [1].")
        state = _make_state(relevant_sources=sources)

        await ainvoke_generate_answer_step(state, context)

        context.llm_provider.get_langchain_model.assert_called_once_with(
            model="test-model",
            temperature=0.7,
        )


@pytest.mark.asyncio
class TestNoSourcesFallback:
    """FR-5: No relevant sources."""

    async def test_returns_fallback_message(self):
        context = _make_context()
        state = _make_state(relevant_sources=[])

        result = await ainvoke_generate_answer_step(state, context)

        messages = result["messages"]
        assert len(messages) == 1
        ai_msg = messages[0]
        assert isinstance(ai_msg, AIMessage)
        assert NO_SOURCES_MESSAGE in ai_msg.content

    async def test_skips_llm_call(self):
        context = _make_context()
        state = _make_state(relevant_sources=[])

        await ainvoke_generate_answer_step(state, context)

        # LLM should NOT be called when there are no sources
        context.llm_provider.get_langchain_model.assert_not_called()


@pytest.mark.asyncio
class TestLLMFailure:
    """FR-2 edge case: LLM raises exception."""

    async def test_returns_error_message_gracefully(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM timeout"))

        mock_provider = MagicMock()
        mock_provider.get_langchain_model.return_value = mock_llm

        context = AgentContext(llm_provider=mock_provider, model_name="test-model")
        state = _make_state(relevant_sources=[_make_source()])

        result = await ainvoke_generate_answer_step(state, context)

        messages = result["messages"]
        assert len(messages) == 1
        ai_msg = messages[0]
        assert isinstance(ai_msg, AIMessage)
        assert "error" in ai_msg.content.lower() or "unable" in ai_msg.content.lower()


@pytest.mark.asyncio
class TestUsesRewrittenQuery:
    """FR-2: Prefers rewritten_query over original_query."""

    async def test_prefers_rewritten_query(self):
        sources = [_make_source()]
        context = _make_context(llm_response="Answer [1].")
        state = _make_state(
            query="original question",
            relevant_sources=sources,
            rewritten_query="rewritten better question",
        )

        await ainvoke_generate_answer_step(state, context)

        # Check that the LLM was invoked with a prompt containing the rewritten query
        call_args = context.llm_provider.get_langchain_model.return_value.ainvoke.call_args
        prompt_content = str(call_args)
        assert "rewritten better question" in prompt_content


@pytest.mark.asyncio
class TestCitationPostProcessing:
    """FR-3: Citation enforcement integration."""

    async def test_invalid_citations_detected(self):
        sources = [_make_source()]
        # LLM cites [1] and [3], but only [1] exists
        llm_answer = "Answer with [1] and [3] citations."
        context = _make_context(llm_response=llm_answer)
        state = _make_state(relevant_sources=sources)

        result = await ainvoke_generate_answer_step(state, context)

        meta = result["metadata"]
        assert "citation_validation" in meta
        # [3] is invalid since only 1 source
        assert 3 in meta["citation_validation"]["invalid_citations"]
