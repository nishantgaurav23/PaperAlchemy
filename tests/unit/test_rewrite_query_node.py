"""Tests for the query rewrite node (S6.5).

Tests cover:
- ainvoke_rewrite_query_step: LLM-based query optimization
- QueryRewriteOutput: structured output model
- Fallback behavior on LLM failure
- Metadata enrichment with rewrite details
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage
from src.services.agents.context import AgentContext
from src.services.agents.state import AgentState, create_initial_state

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider with get_langchain_model."""
    provider = MagicMock()
    mock_model = MagicMock()
    mock_structured = AsyncMock()
    mock_model.with_structured_output.return_value = mock_structured
    provider.get_langchain_model.return_value = mock_model
    return provider


@pytest.fixture
def context(mock_llm_provider):
    """Create an AgentContext with mocked LLM provider."""
    return AgentContext(
        llm_provider=mock_llm_provider,
        model_name="test-model",
        max_retrieval_attempts=3,
    )


@pytest.fixture
def base_state() -> AgentState:
    return create_initial_state("How does BERT work?")


@pytest.fixture
def state_with_rewrite() -> AgentState:
    """State that already went through one retrieval attempt."""
    state = create_initial_state("How does BERT work?")
    state["retrieval_attempts"] = 1
    return state


# ---------------------------------------------------------------------------
# Tests: ainvoke_rewrite_query_step — basic (FR-1)
# ---------------------------------------------------------------------------


class TestRewriteQueryBasic:
    @pytest.mark.asyncio
    async def test_rewrite_returns_partial_state(self, base_state, context, mock_llm_provider):
        """Rewrite should return dict with rewritten_query and messages."""
        from src.services.agents.nodes.rewrite_query_node import QueryRewriteOutput, ainvoke_rewrite_query_step

        mock_structured = mock_llm_provider.get_langchain_model().with_structured_output()
        mock_structured.ainvoke = AsyncMock(
            return_value=QueryRewriteOutput(
                rewritten_query="How does BERT pre-training and bidirectional transformers work in NLP?",
                reasoning="Expanded with technical terms",
            )
        )
        mock_llm_provider.get_langchain_model().with_structured_output.return_value = mock_structured

        result = await ainvoke_rewrite_query_step(base_state, context)

        assert isinstance(result, dict)
        assert "rewritten_query" in result
        assert "messages" in result
        assert result["rewritten_query"] == "How does BERT pre-training and bidirectional transformers work in NLP?"

    @pytest.mark.asyncio
    async def test_rewrite_appends_human_message(self, base_state, context, mock_llm_provider):
        """Rewritten query should be appended as HumanMessage."""
        from src.services.agents.nodes.rewrite_query_node import QueryRewriteOutput, ainvoke_rewrite_query_step

        rewritten = "BERT bidirectional encoder representations from transformers pre-training NLP"
        mock_structured = mock_llm_provider.get_langchain_model().with_structured_output()
        mock_structured.ainvoke = AsyncMock(return_value=QueryRewriteOutput(rewritten_query=rewritten, reasoning="Expanded"))
        mock_llm_provider.get_langchain_model().with_structured_output.return_value = mock_structured

        result = await ainvoke_rewrite_query_step(base_state, context)

        messages = result["messages"]
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == rewritten


# ---------------------------------------------------------------------------
# Tests: uses original_query (FR-1)
# ---------------------------------------------------------------------------


class TestRewriteUsesOriginalQuery:
    @pytest.mark.asyncio
    async def test_reads_original_query_not_latest_message(self, context, mock_llm_provider):
        """Should use original_query to prevent semantic drift on multi-rewrite."""
        from src.services.agents.nodes.rewrite_query_node import QueryRewriteOutput, ainvoke_rewrite_query_step

        state = create_initial_state("How does BERT work?")
        # Simulate a previous rewrite added a different HumanMessage
        state["messages"].append(HumanMessage(content="BERT bidirectional transformers NLP"))

        mock_structured = mock_llm_provider.get_langchain_model().with_structured_output()
        mock_structured.ainvoke = AsyncMock(return_value=QueryRewriteOutput(rewritten_query="improved query", reasoning="test"))
        mock_llm_provider.get_langchain_model().with_structured_output.return_value = mock_structured

        await ainvoke_rewrite_query_step(state, context)

        # Check the prompt was called with original_query, not the latest message
        call_args = mock_structured.ainvoke.call_args[0][0]
        assert "How does BERT work?" in call_args

    @pytest.mark.asyncio
    async def test_fallback_to_latest_message_when_no_original(self, context, mock_llm_provider):
        """If original_query is missing, fall back to latest HumanMessage."""
        from src.services.agents.nodes.rewrite_query_node import QueryRewriteOutput, ainvoke_rewrite_query_step

        state = AgentState(messages=[HumanMessage(content="What is attention?")])

        mock_structured = mock_llm_provider.get_langchain_model().with_structured_output()
        mock_structured.ainvoke = AsyncMock(
            return_value=QueryRewriteOutput(rewritten_query="attention mechanism query", reasoning="test")
        )
        mock_llm_provider.get_langchain_model().with_structured_output.return_value = mock_structured

        result = await ainvoke_rewrite_query_step(state, context)

        assert result["rewritten_query"] == "attention mechanism query"


# ---------------------------------------------------------------------------
# Tests: LLM failure fallback (FR-1)
# ---------------------------------------------------------------------------


class TestRewriteLLMFailure:
    @pytest.mark.asyncio
    async def test_llm_failure_returns_keyword_expanded_query(self, base_state, context, mock_llm_provider):
        """On LLM failure, should fallback to keyword expansion."""
        from src.services.agents.nodes.rewrite_query_node import ainvoke_rewrite_query_step

        mock_structured = mock_llm_provider.get_langchain_model().with_structured_output()
        mock_structured.ainvoke = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        mock_llm_provider.get_langchain_model().with_structured_output.return_value = mock_structured

        result = await ainvoke_rewrite_query_step(base_state, context)

        assert "rewritten_query" in result
        # Fallback should contain original query + academic expansion
        assert "How does BERT work?" in result["rewritten_query"]
        assert "research paper" in result["rewritten_query"].lower() or "arxiv" in result["rewritten_query"].lower()

    @pytest.mark.asyncio
    async def test_llm_failure_still_appends_message(self, base_state, context, mock_llm_provider):
        """Even on failure, should append a HumanMessage with the fallback query."""
        from src.services.agents.nodes.rewrite_query_node import ainvoke_rewrite_query_step

        mock_structured = mock_llm_provider.get_langchain_model().with_structured_output()
        mock_structured.ainvoke = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        mock_llm_provider.get_langchain_model().with_structured_output.return_value = mock_structured

        result = await ainvoke_rewrite_query_step(base_state, context)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], HumanMessage)


# ---------------------------------------------------------------------------
# Tests: structured output model (FR-1)
# ---------------------------------------------------------------------------


class TestQueryRewriteOutput:
    def test_model_valid(self):
        from src.services.agents.nodes.rewrite_query_node import QueryRewriteOutput

        output = QueryRewriteOutput(rewritten_query="improved query", reasoning="expanded terms")
        assert output.rewritten_query == "improved query"
        assert output.reasoning == "expanded terms"

    def test_model_requires_rewritten_query(self):
        from src.services.agents.nodes.rewrite_query_node import QueryRewriteOutput

        with pytest.raises(ValueError):
            QueryRewriteOutput(reasoning="missing query")

    def test_model_default_reasoning(self):
        from src.services.agents.nodes.rewrite_query_node import QueryRewriteOutput

        output = QueryRewriteOutput(rewritten_query="test")
        assert output.reasoning == ""


# ---------------------------------------------------------------------------
# Tests: temperature=0.3 (FR-1)
# ---------------------------------------------------------------------------


class TestRewriteTemperature:
    @pytest.mark.asyncio
    async def test_uses_temperature_03(self, base_state, context, mock_llm_provider):
        """Should call get_langchain_model with temperature=0.3."""
        from src.services.agents.nodes.rewrite_query_node import QueryRewriteOutput, ainvoke_rewrite_query_step

        mock_model = MagicMock()
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=QueryRewriteOutput(rewritten_query="improved", reasoning="test"))
        mock_model.with_structured_output.return_value = mock_structured
        mock_llm_provider.get_langchain_model.return_value = mock_model
        mock_llm_provider.get_langchain_model.reset_mock()

        await ainvoke_rewrite_query_step(base_state, context)

        mock_llm_provider.get_langchain_model.assert_called_once()
        call_kwargs = mock_llm_provider.get_langchain_model.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.3


# ---------------------------------------------------------------------------
# Tests: metadata enrichment (FR-3)
# ---------------------------------------------------------------------------


class TestRewriteMetadata:
    @pytest.mark.asyncio
    async def test_metadata_has_rewrite_details(self, base_state, context, mock_llm_provider):
        """Metadata should contain rewrite section with all required fields."""
        from src.services.agents.nodes.rewrite_query_node import QueryRewriteOutput, ainvoke_rewrite_query_step

        mock_structured = mock_llm_provider.get_langchain_model().with_structured_output()
        mock_structured.ainvoke = AsyncMock(
            return_value=QueryRewriteOutput(rewritten_query="improved query", reasoning="expanded abbreviations")
        )
        mock_llm_provider.get_langchain_model().with_structured_output.return_value = mock_structured

        result = await ainvoke_rewrite_query_step(base_state, context)

        assert "metadata" in result
        rewrite_meta = result["metadata"]["rewrite"]
        assert "original_query" in rewrite_meta
        assert "rewritten_query" in rewrite_meta
        assert "reasoning" in rewrite_meta
        assert "attempt_number" in rewrite_meta
        assert rewrite_meta["original_query"] == "How does BERT work?"
        assert rewrite_meta["rewritten_query"] == "improved query"
        assert rewrite_meta["reasoning"] == "expanded abbreviations"

    @pytest.mark.asyncio
    async def test_metadata_attempt_number_from_state(self, state_with_rewrite, context, mock_llm_provider):
        """Attempt number should reflect retrieval_attempts from state."""
        from src.services.agents.nodes.rewrite_query_node import QueryRewriteOutput, ainvoke_rewrite_query_step

        mock_structured = mock_llm_provider.get_langchain_model().with_structured_output()
        mock_structured.ainvoke = AsyncMock(return_value=QueryRewriteOutput(rewritten_query="improved", reasoning="test"))
        mock_llm_provider.get_langchain_model().with_structured_output.return_value = mock_structured

        result = await ainvoke_rewrite_query_step(state_with_rewrite, context)

        assert result["metadata"]["rewrite"]["attempt_number"] == 1


# ---------------------------------------------------------------------------
# Tests: empty query handling (FR-1)
# ---------------------------------------------------------------------------


class TestRewriteEmptyQuery:
    @pytest.mark.asyncio
    async def test_empty_original_query_uses_messages(self, context, mock_llm_provider):
        """If original_query is empty string, fall back to messages."""
        from src.services.agents.nodes.rewrite_query_node import QueryRewriteOutput, ainvoke_rewrite_query_step

        state = AgentState(
            messages=[HumanMessage(content="What is GPT?")],
            original_query="",
        )

        mock_structured = mock_llm_provider.get_langchain_model().with_structured_output()
        mock_structured.ainvoke = AsyncMock(
            return_value=QueryRewriteOutput(rewritten_query="GPT generative pre-trained transformer", reasoning="test")
        )
        mock_llm_provider.get_langchain_model().with_structured_output.return_value = mock_structured

        result = await ainvoke_rewrite_query_step(state, context)

        assert result["rewritten_query"] == "GPT generative pre-trained transformer"


# ---------------------------------------------------------------------------
# Tests: REWRITE_PROMPT (FR-2)
# ---------------------------------------------------------------------------


class TestRewritePrompt:
    def test_prompt_is_string(self):
        from src.services.agents.nodes.rewrite_query_node import REWRITE_PROMPT

        assert isinstance(REWRITE_PROMPT, str)

    def test_prompt_has_query_placeholder(self):
        from src.services.agents.nodes.rewrite_query_node import REWRITE_PROMPT

        assert "{query}" in REWRITE_PROMPT

    def test_prompt_formats_with_query(self):
        from src.services.agents.nodes.rewrite_query_node import REWRITE_PROMPT

        formatted = REWRITE_PROMPT.format(query="How does BERT work?")
        assert "How does BERT work?" in formatted
        assert "{query}" not in formatted

    def test_prompt_mentions_academic_context(self):
        from src.services.agents.nodes.rewrite_query_node import REWRITE_PROMPT

        lower = REWRITE_PROMPT.lower()
        assert "academic" in lower or "research" in lower or "paper" in lower
