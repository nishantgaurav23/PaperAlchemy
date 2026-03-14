"""Web search fallback node for agentic RAG.

When KB retrieval + grading exhausts retries with no relevant results,
this node searches the web (DuckDuckGo), fetches page content (Jina Reader),
and converts results into SourceItems for the generate_answer node.
"""

from __future__ import annotations

import logging
from typing import Any

from src.services.agents.context import AgentContext
from src.services.agents.models import SourceItem
from src.services.agents.state import AgentState
from src.services.web_search.content_fetcher import fetch_multiple_pages

logger = logging.getLogger(__name__)

_MAX_WEB_RESULTS = 5
_MAX_FETCH = 3  # only fetch content from top N URLs


async def ainvoke_web_search_step(
    state: AgentState,
    context: AgentContext,
) -> dict[str, Any]:
    """Search the web and fetch page content as fallback when KB has no results.

    Workflow:
    1. Search DuckDuckGo for the query
    2. Fetch actual page content from top results via Jina Reader
    3. Convert to SourceItems (with snippet fallback if fetch fails)
    4. Set as relevant_sources so generate_answer can use them

    Args:
        state: Current agent state (reads original_query, rewritten_query).
        context: Runtime context (reads web_search_service).

    Returns:
        Partial state dict with relevant_sources populated from web results.
    """
    logger.info("NODE: web_search (fallback)")

    web_search = getattr(context, "web_search_service", None)
    if web_search is None:
        logger.warning("No web search service available — skipping web fallback")
        return {"relevant_sources": [], "metadata": {"web_search": "no_service"}}

    # Use the best available query
    query = state.get("rewritten_query") or state.get("original_query") or ""
    if not query:
        return {"relevant_sources": [], "metadata": {"web_search": "no_query"}}

    # Step 1: Search DuckDuckGo
    try:
        search_response = await web_search.search(
            f"{query} research paper",
            max_results=_MAX_WEB_RESULTS,
        )
    except Exception as e:
        logger.warning("Web search failed: %s", e)
        return {"relevant_sources": [], "metadata": {"web_search": f"search_error: {e}"}}

    if not search_response.results:
        logger.info("Web search returned no results")
        return {"relevant_sources": [], "metadata": {"web_search": "no_results"}}

    # Step 2: Fetch actual page content from top URLs
    urls_to_fetch = [r.url for r in search_response.results[:_MAX_FETCH] if r.url]
    fetched_content: dict[str, str] = {}
    if urls_to_fetch:
        try:
            fetched_content = await fetch_multiple_pages(urls_to_fetch)
            logger.info("Fetched content from %d/%d URLs", len(fetched_content), len(urls_to_fetch))
        except Exception as e:
            logger.warning("Content fetching failed: %s", e)

    # Step 3: Convert to SourceItems
    web_sources: list[SourceItem] = []
    for result in search_response.results:
        # Use fetched content if available, otherwise fall back to snippet
        content = fetched_content.get(result.url, result.snippet)
        if not content:
            content = result.snippet

        web_sources.append(
            SourceItem(
                arxiv_id=f"web:{result.source or 'unknown'}",
                title=result.title,
                authors=[],
                url=result.url,
                chunk_text=content,
                relevance_score=0.3,  # lower trust than KB
            )
        )

    logger.info("Web search produced %d source items", len(web_sources))

    return {
        "relevant_sources": web_sources,
        "metadata": {
            "web_search": "success",
            "web_results_count": len(web_sources),
            "pages_fetched": len(fetched_content),
        },
    }
