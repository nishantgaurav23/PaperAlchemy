"""Tests for AgentState TypedDict and create_initial_state factory (FR-1, FR-4)."""

from __future__ import annotations

import typing
from typing import Annotated, get_type_hints

import pytest
from langchain_core.messages import HumanMessage
from src.services.agents.models import GuardrailScoring
from src.services.agents.state import AgentState, create_initial_state


class TestAgentState:
    def test_is_typeddict(self):
        """AgentState should be a TypedDict."""
        assert hasattr(AgentState, "__annotations__")
        # TypedDicts have __required_keys__ and __optional_keys__
        assert hasattr(AgentState, "__required_keys__")
        assert hasattr(AgentState, "__optional_keys__")

    def test_total_false(self):
        """AgentState should have total=False (all fields optional in partial returns)."""
        # With total=False, __required_keys__ should be empty
        assert len(AgentState.__required_keys__) == 0

    def test_has_messages_field(self):
        """AgentState must have a 'messages' field."""
        assert "messages" in AgentState.__annotations__

    def test_messages_uses_add_messages_reducer(self):
        """messages field should use Annotated[..., add_messages] for LangGraph."""
        hints = get_type_hints(AgentState, include_extras=True)
        messages_hint = hints["messages"]
        # Should be Annotated type
        assert typing.get_origin(messages_hint) is Annotated
        # The metadata should contain add_messages function
        metadata = typing.get_args(messages_hint)
        from langgraph.graph import add_messages

        assert add_messages in metadata

    def test_has_all_expected_fields(self):
        """AgentState should have all specified fields."""
        expected = {
            "messages",
            "original_query",
            "rewritten_query",
            "retrieval_attempts",
            "guardrail_result",
            "routing_decision",
            "sources",
            "grading_results",
            "relevant_sources",
            "metadata",
        }
        assert expected.issubset(set(AgentState.__annotations__.keys()))

    def test_partial_return_works(self):
        """Nodes should be able to return partial dicts as AgentState."""
        # This simulates what a node returns — just the fields it modifies
        partial: AgentState = {"routing_decision": "retrieve"}  # type: ignore[typeddict-item]
        assert partial["routing_decision"] == "retrieve"

    def test_partial_return_with_guardrail(self):
        gs = GuardrailScoring(score=80, reason="on topic")
        partial: AgentState = {"guardrail_result": gs}  # type: ignore[typeddict-item]
        assert partial["guardrail_result"].score == 80


class TestCreateInitialState:
    def test_creates_state_with_human_message(self):
        state = create_initial_state("What is attention mechanism?")
        assert len(state["messages"]) == 1
        assert isinstance(state["messages"][0], HumanMessage)
        assert state["messages"][0].content == "What is attention mechanism?"

    def test_sets_original_query(self):
        state = create_initial_state("test query")
        assert state["original_query"] == "test query"

    def test_default_values(self):
        state = create_initial_state("test")
        assert state["rewritten_query"] is None
        assert state["retrieval_attempts"] == 0
        assert state["guardrail_result"] is None
        assert state["routing_decision"] is None
        assert state["sources"] == []
        assert state["grading_results"] == []
        assert state["relevant_sources"] == []
        assert state["metadata"] == {}

    def test_empty_query_raises(self):
        with pytest.raises(ValueError, match="query"):
            create_initial_state("")

    def test_whitespace_only_query_raises(self):
        with pytest.raises(ValueError, match="query"):
            create_initial_state("   ")
