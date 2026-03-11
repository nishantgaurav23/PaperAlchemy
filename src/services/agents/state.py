"""Agent state schema for LangGraph workflow.

Defines the TypedDict that flows through every agent node, carrying
message history, retrieval results, grading outcomes, and routing decisions.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import add_messages
from src.services.agents.models import GradingResult, GuardrailScoring, SourceItem


class AgentState(TypedDict, total=False):
    """State flowing through the agentic RAG LangGraph workflow.

    Uses total=False so nodes can return partial dicts with only modified fields.
    LangGraph merges partials automatically.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    original_query: str | None
    rewritten_query: str | None
    retrieval_attempts: int
    guardrail_result: GuardrailScoring | None
    routing_decision: str | None
    sources: list[SourceItem]
    grading_results: list[GradingResult]
    relevant_sources: list[SourceItem]
    metadata: dict[str, Any]


def create_initial_state(query: str) -> AgentState:
    """Create a fresh AgentState for a new user query.

    Args:
        query: The user's research question. Must be non-empty.

    Returns:
        AgentState with HumanMessage in messages and all other fields at defaults.

    Raises:
        ValueError: If query is empty or whitespace-only.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    return AgentState(
        messages=[HumanMessage(content=query)],
        original_query=query,
        rewritten_query=None,
        retrieval_attempts=0,
        guardrail_result=None,
        routing_decision=None,
        sources=[],
        grading_results=[],
        relevant_sources=[],
        metadata={},
    )
