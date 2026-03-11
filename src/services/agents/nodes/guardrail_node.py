"""Guardrail node for domain relevance filtering.

First node in the agentic RAG LangGraph workflow. Scores the user query
on a 0-100 domain relevance scale using structured LLM output, then a
conditional edge routes the graph based on the score vs threshold.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.messages import AnyMessage, HumanMessage
from src.services.agents.context import AgentContext
from src.services.agents.models import GuardrailScoring
from src.services.agents.state import AgentState

logger = logging.getLogger(__name__)

GUARDRAIL_PROMPT = """You are a domain relevance classifier for an academic research assistant.

Rate the following question on a scale of 0-100 for relevance to academic and scientific research,
particularly in the areas of machine learning, artificial intelligence, natural language processing,
computer science, and topics commonly found on arXiv.

Scoring guide:
- 80-100: Directly about research papers, algorithms, models, methods, or scientific findings
- 60-79: Related to academic topics but more general (e.g., "What is deep learning?")
- 40-59: Tangentially related or ambiguous
- 20-39: Mostly off-topic but has some academic angle
- 0-19: Completely off-topic (cooking, sports, personal advice, etc.)

Question: {question}

Provide your score (0-100) and a brief reason for the rating."""


def get_latest_query(messages: list[AnyMessage]) -> str:
    """Extract the content of the last HumanMessage from the message list.

    Args:
        messages: List of LangChain messages.

    Returns:
        The text content of the last HumanMessage.

    Raises:
        ValueError: If no HumanMessage is found in the list.
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    raise ValueError("No HumanMessage found in messages")


def continue_after_guardrail(
    state: AgentState,
    context: AgentContext,
) -> Literal["continue", "out_of_scope"]:
    """Conditional edge: route based on guardrail score vs threshold.

    Args:
        state: Current agent state containing guardrail_result.
        context: Runtime context with guardrail_threshold.

    Returns:
        "continue" if score >= threshold, "out_of_scope" otherwise.
    """
    guardrail_result = state.get("guardrail_result")

    if not guardrail_result:
        logger.warning("No guardrail_result in state - defaulting to continue")
        return "continue"

    score = guardrail_result.score
    threshold = context.guardrail_threshold

    logger.info("Guardrail routing: score=%d, threshold=%d", score, threshold)

    return "continue" if score >= threshold else "out_of_scope"


async def ainvoke_guardrail_step(
    state: AgentState,
    context: AgentContext,
) -> dict[str, Any]:
    """Score query domain relevance using a structured LLM call.

    Extracts the latest HumanMessage, calls the LLM with structured output
    (GuardrailScoring), and returns a partial state dict for LangGraph to merge.

    On LLM failure, falls back to score=50 (above default threshold=40) so
    queries proceed to retrieval rather than crashing.

    Args:
        state: Current agent state (reads messages for the query).
        context: Runtime context (reads llm_provider, model_name).

    Returns:
        Partial state dict: {"guardrail_result": GuardrailScoring}.
    """
    logger.info("NODE: guardrail_validation")

    query = get_latest_query(state["messages"])
    logger.debug("Evaluating query: %s", query[:100])

    try:
        prompt = GUARDRAIL_PROMPT.format(question=query)

        llm = context.llm_provider.get_langchain_model(
            model=context.model_name,
            temperature=0.0,
        )
        structured_llm = llm.with_structured_output(GuardrailScoring)

        logger.info("Invoking LLM for guardrail scoring")
        response = await structured_llm.ainvoke(prompt)

        logger.info("Guardrail result - score: %d, reason: %s", response.score, response.reason)

    except Exception as e:
        logger.error("Guardrail LLM call failed: %s - falling back to score=50", e)
        response = GuardrailScoring(
            score=50,
            reason=f"LLM validation failed, using conservative fallback: {e}",
        )

    return {"guardrail_result": response}
