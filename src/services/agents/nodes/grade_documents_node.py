"""Document grading node for binary relevance filtering.

After retrieval, this node evaluates each retrieved document against the
user's query using an LLM with structured output (GradeDocuments).
Documents graded "yes" are promoted to relevant_sources; "no" are filtered out.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from src.services.agents.context import AgentContext
from src.services.agents.models import GradeDocuments, GradingResult, SourceItem
from src.services.agents.state import AgentState

logger = logging.getLogger(__name__)

GRADING_PROMPT = """You are a document relevance grader for an academic research assistant.

Given a user query and a retrieved document chunk, assess whether the document is relevant
to answering the query.

Grade as "yes" if the document contains information that would help answer the query.
Grade as "no" if the document is not relevant or does not contain useful information.

User query: {query}

Document content:
{document}

Provide your binary grade ("yes" or "no") and a brief reasoning."""


def _get_latest_query(state: AgentState) -> str:
    """Extract the latest user query from state messages."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return state.get("original_query", "")


async def _grade_single_document(
    source: SourceItem,
    query: str,
    structured_llm: Any,
) -> GradingResult:
    """Grade a single document against the query.

    On LLM failure, returns a not-relevant result with error reasoning.
    """
    try:
        prompt = GRADING_PROMPT.format(query=query, document=source.chunk_text)
        response: GradeDocuments = await structured_llm.ainvoke(prompt)

        is_relevant = response.binary_score == "yes"
        return GradingResult(
            document_id=source.arxiv_id,
            is_relevant=is_relevant,
            score=1.0 if is_relevant else 0.0,
            reasoning=response.reasoning,
        )
    except Exception as e:
        logger.warning("LLM grading failed for %s: %s — marking as not relevant", source.arxiv_id, e)
        return GradingResult(
            document_id=source.arxiv_id,
            is_relevant=False,
            score=0.0,
            reasoning=f"Grading failed: {e}",
        )


async def ainvoke_grade_documents_step(
    state: AgentState,
    context: AgentContext,
) -> dict[str, Any]:
    """Grade each retrieved document for relevance to the user query.

    Iterates over state["sources"], calls the LLM with structured output
    (GradeDocuments) for each, and partitions into relevant/not-relevant.

    Args:
        state: Current agent state (reads sources, messages).
        context: Runtime context (reads llm_provider, model_name).

    Returns:
        Partial state dict with grading_results and relevant_sources.
    """
    logger.info("NODE: grade_documents")

    sources: list[SourceItem] = state.get("sources", [])

    if not sources:
        logger.info("No sources to grade — returning empty results")
        return {"grading_results": [], "relevant_sources": []}

    query = _get_latest_query(state)
    logger.debug("Grading %d documents for query: %s", len(sources), query[:100])

    llm = context.llm_provider.get_langchain_model(
        model=context.model_name,
        temperature=0.0,
    )
    structured_llm = llm.with_structured_output(GradeDocuments)

    # Grade all documents in parallel for speed
    import asyncio

    grading_tasks = [_grade_single_document(source, query, structured_llm) for source in sources]
    grading_results: list[GradingResult] = list(await asyncio.gather(*grading_tasks))
    relevant_sources: list[SourceItem] = [
        source for source, result in zip(sources, grading_results, strict=True) if result.is_relevant
    ]

    logger.info(
        "Grading complete: %d/%d documents relevant",
        len(relevant_sources),
        len(sources),
    )

    return {"grading_results": grading_results, "relevant_sources": relevant_sources}


def continue_after_grading(
    state: AgentState,
    context: AgentContext,
) -> Literal["generate", "rewrite", "web_search"]:
    """Conditional edge: route based on grading results.

    Args:
        state: Current agent state with relevant_sources, sources, and retrieval_attempts.
        context: Runtime context with max_retrieval_attempts, web_search_service.

    Returns:
        "generate" if relevant sources exist;
        "rewrite" if some docs were retrieved but none relevant, and retries remain;
        "web_search" if retrieval returned nothing (infra down / empty KB) or retries exhausted;
        "generate" as final fallback (no sources message).
    """
    relevant_sources = state.get("relevant_sources", [])
    retrieved_sources = state.get("sources", [])
    retrieval_attempts = state.get("retrieval_attempts", 0)
    max_attempts = context.max_retrieval_attempts
    has_web_search = getattr(context, "web_search_service", None) is not None

    if relevant_sources:
        logger.info("Routing to generate: %d relevant sources found", len(relevant_sources))
        return "generate"

    # If retrieval returned ZERO docs (infra down, empty KB), rewriting won't help.
    # Skip straight to web search instead of wasting LLM calls on rewrites.
    if not retrieved_sources and has_web_search:
        logger.info("Routing to web_search: retrieval returned 0 docs, rewriting won't help")
        return "web_search"

    # Docs were retrieved but none graded relevant — rewrite may help
    if retrieval_attempts < max_attempts:
        logger.info("Routing to rewrite: no relevant sources, attempt %d/%d", retrieval_attempts, max_attempts)
        return "rewrite"

    # Retries exhausted — try web search if available
    if has_web_search:
        logger.info("Routing to web_search: KB retries exhausted, falling back to web")
        return "web_search"

    logger.warning("Routing to generate: retries exhausted, no web search available (%d/%d)", retrieval_attempts, max_attempts)
    return "generate"
