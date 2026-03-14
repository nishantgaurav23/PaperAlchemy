"""Web search service using DuckDuckGo for real-time information retrieval."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    """A single web search result."""

    title: str
    url: str
    snippet: str
    source: str = ""  # domain name


@dataclass
class WebSearchResponse:
    """Response from a web search query."""

    query: str
    results: list[WebSearchResult] = field(default_factory=list)


class WebSearchService:
    """Web search via DuckDuckGo — no API key required."""

    def __init__(self, max_results: int = 5, region: str = "wt-wt") -> None:
        self._max_results = max_results
        self._region = region

    async def search(self, query: str, *, max_results: int | None = None) -> WebSearchResponse:
        """Search the web for a given query.

        DuckDuckGo search is synchronous, so we run it in a thread pool
        with a 15-second timeout to prevent indefinite hangs.
        """
        max_results = max_results or self._max_results

        def _search() -> list[dict]:
            try:
                from ddgs import DDGS

                return list(DDGS().text(query, region=self._region, max_results=max_results))
            except Exception as e:
                logger.warning("DuckDuckGo search failed: %s", e)
                return []

        try:
            raw_results = await asyncio.wait_for(asyncio.to_thread(_search), timeout=15.0)
        except asyncio.TimeoutError:
            logger.warning("DuckDuckGo search timed out after 15s for query: %s", query[:80])
            raw_results = []

        results = []
        for r in raw_results:
            results.append(
                WebSearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", r.get("link", "")),
                    snippet=r.get("body", r.get("snippet", "")),
                    source=r.get("source", ""),
                )
            )

        return WebSearchResponse(query=query, results=results)
