"""Retrieval node for the agentic RAG LangGraph workflow (S6.3).

Second node in the pipeline (after guardrail). Invokes the advanced retrieval
pipeline to fetch relevant documents and populates the agent state with
SourceItem results for downstream grading and generation.

Retrieval is MANDATORY — the agent must always call the retrieval pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from src.schemas.api.search import SearchHit
from src.services.agents.context import AgentContext
from src.services.agents.models import SourceItem
from src.services.agents.nodes.guardrail_node import get_latest_query
from src.services.agents.state import AgentState

logger = logging.getLogger(__name__)


def convert_search_hits_to_sources(hits: list[SearchHit]) -> list[SourceItem]:
    """Convert retrieval pipeline SearchHits into SourceItems for agent state.

    Skips hits with empty arxiv_id. Constructs arXiv URL from arxiv_id when
    pdf_url is not available.

    Args:
        hits: List of SearchHit objects from the retrieval pipeline.

    Returns:
        List of SourceItem objects with proper field mapping.
    """
    sources: list[SourceItem] = []
    for hit in hits:
        if not hit.arxiv_id:
            logger.debug("Skipping hit with empty arxiv_id: %s", hit.title)
            continue

        url = hit.pdf_url if hit.pdf_url else f"https://arxiv.org/abs/{hit.arxiv_id}"

        sources.append(
            SourceItem(
                arxiv_id=hit.arxiv_id,
                title=hit.title,
                authors=hit.authors,
                url=url,
                relevance_score=hit.score,
                chunk_text=hit.chunk_text,
            )
        )
    return sources


def _get_query(state: AgentState) -> str:
    """Extract the best available query from state.

    Priority: rewritten_query > original_query > last HumanMessage.
    """
    rewritten = state.get("rewritten_query")
    if rewritten:
        return rewritten

    original = state.get("original_query")
    if original:
        return original

    return get_latest_query(state["messages"])


async def ainvoke_retrieve_step(
    state: AgentState,
    context: AgentContext,
) -> dict[str, Any]:
    """Invoke the retrieval pipeline and populate agent state with sources.

    Calls ``context.retrieval_pipeline.retrieve()`` with the current query,
    converts results to SourceItems, and enriches metadata with pipeline stats.

    On failure (None pipeline, exception), returns empty sources with error metadata
    rather than crashing the workflow.

    Args:
        state: Current agent state (reads query fields, retrieval_attempts).
        context: Runtime context (reads retrieval_pipeline, top_k).

    Returns:
        Partial state dict with sources, retrieval_attempts, and metadata.
    """
    logger.info("NODE: retrieve_documents")

    query = _get_query(state)
    current_attempts = state.get("retrieval_attempts", 0)
    existing_metadata = dict(state.get("metadata", {}))

    # Handle missing pipeline
    if context.retrieval_pipeline is None:
        logger.error("Retrieval pipeline is None — cannot retrieve documents")
        existing_metadata["retrieval"] = {
            "error": "retrieval_pipeline is None",
            "query_used": query,
            "num_results": 0,
        }
        return {
            "sources": [],
            "retrieval_attempts": current_attempts + 1,
            "metadata": existing_metadata,
        }

    # Invoke pipeline
    try:
        logger.info("Retrieving documents for query: %s", query[:100])
        retrieval_result = await context.retrieval_pipeline.retrieve(query, top_k=context.top_k)
    except Exception as e:
        logger.error("Retrieval pipeline failed: %s", e, exc_info=True)
        existing_metadata["retrieval"] = {
            "error": str(e),
            "query_used": query,
            "num_results": 0,
        }
        return {
            "sources": [],
            "retrieval_attempts": current_attempts + 1,
            "metadata": existing_metadata,
        }

    # Convert results
    sources = convert_search_hits_to_sources(retrieval_result.results)

    if not sources:
        logger.warning("Retrieval returned zero relevant sources for query: %s", query[:100])

    # Enrich metadata
    existing_metadata["retrieval"] = {
        "stages_executed": retrieval_result.stages_executed,
        "total_candidates": retrieval_result.total_candidates,
        "timings": retrieval_result.timings,
        "query_used": query,
        "num_results": len(sources),
    }

    logger.info("Retrieved %d sources (from %d candidates)", len(sources), retrieval_result.total_candidates)

    return {
        "sources": sources,
        "retrieval_attempts": current_attempts + 1,
        "metadata": existing_metadata,
    }
