"""Query rewrite node for query optimization.

When document grading finds insufficient relevant results, this node rewrites
the user's query with synonym expansion, context refinement, and academic
terminology enrichment. Reads original_query (not latest message) to prevent
semantic drift across multiple rewrite cycles.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from src.services.agents.context import AgentContext
from src.services.agents.nodes.guardrail_node import get_latest_query
from src.services.agents.state import AgentState

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """You are a query optimization specialist for academic paper retrieval.

Your task is to rewrite the following research query to improve retrieval results.
The original query did not return enough relevant academic papers.

Rewriting strategies:
- Expand abbreviations (e.g., "NLP" → "natural language processing NLP")
- Add synonyms and related technical terms
- Include broader or narrower terms as appropriate
- Add domain context (e.g., "machine learning", "deep learning")
- Keep the query focused on the original intent — do NOT drift to unrelated topics

Original query: {query}

Provide the improved query and a brief explanation of what you changed."""


class QueryRewriteOutput(BaseModel):
    """Structured output for query rewrite LLM calls."""

    rewritten_query: str
    reasoning: str = ""


async def ainvoke_rewrite_query_step(
    state: AgentState,
    context: AgentContext,
) -> dict[str, Any]:
    """Rewrite the user query for better retrieval using structured LLM output.

    Reads original_query from state to prevent semantic drift on repeated
    rewrites. Falls back to keyword expansion if the LLM call fails.

    Args:
        state: Current agent state (reads original_query, messages, retrieval_attempts).
        context: Runtime context (reads llm_provider, model_name).

    Returns:
        Partial state dict with rewritten_query, messages (HumanMessage), and metadata.
    """
    logger.info("NODE: rewrite_query")

    # Use original_query to prevent drift; fall back to latest message
    original = state.get("original_query") or ""
    if not original.strip():
        try:
            original = get_latest_query(state.get("messages", []))
        except ValueError:
            original = ""

    attempt_number = state.get("retrieval_attempts", 0)
    logger.debug("Rewriting query (attempt %d): %s", attempt_number, original[:100])

    try:
        prompt = REWRITE_PROMPT.format(query=original)

        llm = context.llm_provider.get_langchain_model(
            model=context.model_name,
            temperature=0.3,
        )
        structured_llm = llm.with_structured_output(QueryRewriteOutput)

        logger.info("Invoking LLM for query rewrite")
        response: QueryRewriteOutput = await structured_llm.ainvoke(prompt)

        rewritten = response.rewritten_query
        reasoning = response.reasoning

        logger.info("Query rewritten: %s → %s", original[:80], rewritten[:80])

    except Exception as e:
        logger.error("Rewrite LLM call failed: %s — falling back to keyword expansion", e)
        rewritten = f"{original} research paper arxiv"
        reasoning = f"LLM rewrite failed, using keyword expansion fallback: {e}"

    metadata = dict(state.get("metadata", {}))
    metadata["rewrite"] = {
        "original_query": original,
        "rewritten_query": rewritten,
        "reasoning": reasoning,
        "attempt_number": attempt_number,
    }

    return {
        "rewritten_query": rewritten,
        "messages": [HumanMessage(content=rewritten)],
        "metadata": metadata,
    }
