"""Tests for the guardrail node (S6.2).

Tests cover:
- ainvoke_guardrail_step: LLM-based domain relevance scoring
- continue_after_guardrail: conditional edge routing
- get_latest_query: message extraction helper
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from src.services.agents.context import AgentContext
from src.services.agents.models import GuardrailScoring
from src.services.agents.nodes.guardrail_node import (
    GUARDRAIL_PROMPT,
    ainvoke_guardrail_step,
    continue_after_guardrail,
    get_latest_query,
)
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
        guardrail_threshold=40,
    )


@pytest.fixture
def on_topic_state() -> AgentState:
    return create_initial_state("How do transformers work in NLP?")


@pytest.fixture
def off_topic_state() -> AgentState:
    return create_initial_state("What is the best pizza in New York?")


# ---------------------------------------------------------------------------
# Tests: get_latest_query (FR-3)
# ---------------------------------------------------------------------------


class TestGetLatestQuery:
    def test_extracts_human_message(self):
        messages = [HumanMessage(content="How does BERT work?")]
        assert get_latest_query(messages) == "How does BERT work?"

    def test_extracts_last_human_message(self):
        messages = [
            HumanMessage(content="First question"),
            AIMessage(content="Some answer"),
            HumanMessage(content="Follow-up question"),
        ]
        assert get_latest_query(messages) == "Follow-up question"

    def test_empty_messages_raises(self):
        with pytest.raises(ValueError, match="[Nn]o.*[Hh]uman[Mm]essage"):
            get_latest_query([])

    def test_no_human_message_raises(self):
        messages = [AIMessage(content="I am an AI")]
        with pytest.raises(ValueError, match="[Nn]o.*[Hh]uman[Mm]essage"):
            get_latest_query(messages)


# ---------------------------------------------------------------------------
# Tests: continue_after_guardrail (FR-2)
# ---------------------------------------------------------------------------


class TestContinueAfterGuardrail:
    def test_above_threshold_returns_continue(self, context):
        state = AgentState(
            guardrail_result=GuardrailScoring(score=85, reason="Academic query"),
        )
        result = continue_after_guardrail(state, context)
        assert result == "continue"

    def test_below_threshold_returns_out_of_scope(self, context):
        state = AgentState(
            guardrail_result=GuardrailScoring(score=10, reason="Off-topic"),
        )
        result = continue_after_guardrail(state, context)
        assert result == "out_of_scope"

    def test_exact_threshold_returns_continue(self, context):
        state = AgentState(
            guardrail_result=GuardrailScoring(score=40, reason="Borderline"),
        )
        result = continue_after_guardrail(state, context)
        assert result == "continue"

    def test_no_guardrail_result_defaults_to_continue(self, context):
        state = AgentState()
        result = continue_after_guardrail(state, context)
        assert result == "continue"

    def test_custom_threshold(self, mock_llm_provider):
        ctx = AgentContext(llm_provider=mock_llm_provider, guardrail_threshold=70)
        state = AgentState(
            guardrail_result=GuardrailScoring(score=60, reason="Research-ish"),
        )
        result = continue_after_guardrail(state, ctx)
        assert result == "out_of_scope"


# ---------------------------------------------------------------------------
# Tests: ainvoke_guardrail_step (FR-1)
# ---------------------------------------------------------------------------


class TestAinvokeGuardrailStep:
    @pytest.mark.asyncio
    async def test_high_score_on_topic(self, on_topic_state, context, mock_llm_provider):
        """On-topic query should get high score."""
        mock_structured = mock_llm_provider.get_langchain_model().with_structured_output()
        mock_structured.ainvoke = AsyncMock(
            return_value=GuardrailScoring(score=92, reason="Transformer architecture is core NLP research")
        )
        mock_llm_provider.get_langchain_model().with_structured_output.return_value = mock_structured

        result = await ainvoke_guardrail_step(on_topic_state, context)

        assert "guardrail_result" in result
        assert result["guardrail_result"].score == 92
        assert result["guardrail_result"].reason != ""

    @pytest.mark.asyncio
    async def test_low_score_off_topic(self, off_topic_state, context, mock_llm_provider):
        """Off-topic query should get low score."""
        mock_structured = mock_llm_provider.get_langchain_model().with_structured_output()
        mock_structured.ainvoke = AsyncMock(return_value=GuardrailScoring(score=5, reason="Pizza is not academic research"))
        mock_llm_provider.get_langchain_model().with_structured_output.return_value = mock_structured

        result = await ainvoke_guardrail_step(off_topic_state, context)

        assert result["guardrail_result"].score == 5

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self, on_topic_state, context, mock_llm_provider):
        """LLM failure should fallback to score=50."""
        mock_structured = mock_llm_provider.get_langchain_model().with_structured_output()
        mock_structured.ainvoke = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        mock_llm_provider.get_langchain_model().with_structured_output.return_value = mock_structured

        result = await ainvoke_guardrail_step(on_topic_state, context)

        assert result["guardrail_result"].score == 50
        assert "failed" in result["guardrail_result"].reason.lower() or "fallback" in result["guardrail_result"].reason.lower()

    @pytest.mark.asyncio
    async def test_uses_zero_temperature(self, on_topic_state, context, mock_llm_provider):
        """Should call get_langchain_model with temperature=0.0."""
        mock_structured = mock_llm_provider.get_langchain_model().with_structured_output()
        mock_structured.ainvoke = AsyncMock(return_value=GuardrailScoring(score=80, reason="Research topic"))
        mock_llm_provider.get_langchain_model().with_structured_output.return_value = mock_structured
        mock_llm_provider.get_langchain_model.reset_mock()

        await ainvoke_guardrail_step(on_topic_state, context)

        mock_llm_provider.get_langchain_model.assert_called_once()
        call_kwargs = mock_llm_provider.get_langchain_model.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.0

    @pytest.mark.asyncio
    async def test_uses_structured_output(self, on_topic_state, context, mock_llm_provider):
        """Should call with_structured_output(GuardrailScoring)."""
        mock_model = MagicMock()
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=GuardrailScoring(score=80, reason="Research topic"))
        mock_model.with_structured_output.return_value = mock_structured
        mock_llm_provider.get_langchain_model.return_value = mock_model

        await ainvoke_guardrail_step(on_topic_state, context)

        mock_model.with_structured_output.assert_called_once_with(GuardrailScoring)

    @pytest.mark.asyncio
    async def test_returns_partial_state_dict(self, on_topic_state, context, mock_llm_provider):
        """Should return a dict with only 'guardrail_result' key."""
        mock_structured = mock_llm_provider.get_langchain_model().with_structured_output()
        mock_structured.ainvoke = AsyncMock(return_value=GuardrailScoring(score=75, reason="Relevant"))
        mock_llm_provider.get_langchain_model().with_structured_output.return_value = mock_structured

        result = await ainvoke_guardrail_step(on_topic_state, context)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"guardrail_result"}
        assert isinstance(result["guardrail_result"], GuardrailScoring)


# ---------------------------------------------------------------------------
# Tests: GUARDRAIL_PROMPT (FR-4)
# ---------------------------------------------------------------------------


class TestGuardrailPrompt:
    def test_prompt_is_string(self):
        assert isinstance(GUARDRAIL_PROMPT, str)

    def test_prompt_has_placeholder(self):
        assert "{question}" in GUARDRAIL_PROMPT

    def test_prompt_formats_with_query(self):
        formatted = GUARDRAIL_PROMPT.format(question="How does BERT work?")
        assert "How does BERT work?" in formatted
        assert "{question}" not in formatted

    def test_prompt_mentions_academic_research(self):
        lower = GUARDRAIL_PROMPT.lower()
        assert "research" in lower or "academic" in lower or "scientific" in lower
