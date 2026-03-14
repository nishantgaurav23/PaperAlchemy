"""RAG chain service (S5.2).

Orchestrates: retrieve → prompt → generate, with citation enforcement.
Supports both single-shot (aquery) and streaming (aquery_stream) modes.

Enhanced to search three sources in parallel:
1. Knowledge base (OpenSearch hybrid search)
2. arXiv API (live paper search)
3. Web search (DuckDuckGo)

Always returns top 3 source links in the response.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from src.schemas.api.search import SearchHit
from src.services.arxiv.client import ArxivClient
from src.services.llm.provider import LLMProvider
from src.services.rag.citation import enforce_citations, stream_with_citations
from src.services.rag.models import LLMMetadata, RAGResponse, RetrievalMetadata, SourceReference
from src.services.rag.prompts import SYSTEM_PROMPT, build_mixed_user_prompt, build_user_prompt
from src.services.retrieval.pipeline import RetrievalPipeline, RetrievalResult
from src.services.web_search.service import WebSearchService

logger = logging.getLogger(__name__)

_NO_RESULTS_MESSAGE = (
    "I could not find any relevant information for your query from the knowledge base, "
    "arXiv, or web search. Try rephrasing your question or broadening your search terms."
)


class RAGChain:
    """Retrieval-Augmented Generation chain with multi-source retrieval and citation enforcement."""

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        retrieval_pipeline: RetrievalPipeline,
        arxiv_client: ArxivClient | None = None,
        web_search_service: WebSearchService | None = None,
    ) -> None:
        self._llm = llm_provider
        self._retrieval = retrieval_pipeline
        self._arxiv = arxiv_client
        self._web_search = web_search_service

    # ------------------------------------------------------------------
    # Multi-source retrieval
    # ------------------------------------------------------------------

    async def _retrieve_all_sources(
        self,
        query: str,
        *,
        top_k: int | None = None,
        categories: list[str] | None = None,
    ) -> tuple[list[SourceReference], RetrievalResult]:
        """Run KB + arXiv + web search in parallel, merge into unified source list.

        Returns (merged_sources, kb_retrieval_result).
        """
        # Launch all three searches in parallel

        async def _noop() -> list:
            return []

        kb_task = self._retrieval.retrieve(query, top_k=top_k, categories=categories)
        arxiv_task = self._search_arxiv(query) if self._arxiv else _noop()
        web_task = self._search_web(query) if self._web_search else _noop()

        try:
            results = await asyncio.wait_for(
                asyncio.gather(kb_task, arxiv_task, web_task, return_exceptions=True),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Multi-source retrieval timed out after 60s")
            results = [TimeoutError("timeout"), TimeoutError("timeout"), TimeoutError("timeout")]

        # Process KB results
        kb_result: RetrievalResult
        kb_sources: list[SourceReference] = []
        if isinstance(results[0], RetrievalResult):
            kb_result = results[0]
            kb_sources = self._build_sources(results[0].results, source_type="knowledge_base")
        else:
            kb_result = RetrievalResult(query=query)
            if isinstance(results[0], Exception):
                logger.warning("KB retrieval failed: %s", results[0])

        # Process arXiv results
        arxiv_sources: list[SourceReference] = []
        if isinstance(results[1], list):
            arxiv_sources = results[1]
        elif isinstance(results[1], Exception):
            logger.warning("arXiv search failed: %s", results[1])

        # Process web results
        web_sources: list[SourceReference] = []
        if isinstance(results[2], list):
            web_sources = results[2]
        elif isinstance(results[2], Exception):
            logger.warning("Web search failed: %s", results[2])

        # Merge: KB first (highest trust), then arXiv, then web
        # Deduplicate arXiv results that are already in KB
        kb_arxiv_ids = {s.arxiv_id for s in kb_sources if s.arxiv_id}
        arxiv_sources = [s for s in arxiv_sources if s.arxiv_id not in kb_arxiv_ids]

        all_sources = kb_sources + arxiv_sources + web_sources

        # Re-index 1-based
        for i, src in enumerate(all_sources, start=1):
            src.index = i

        return all_sources, kb_result

    async def _search_arxiv(self, query: str) -> list[SourceReference]:
        """Search arXiv API for live papers matching the query."""
        if not self._arxiv:
            return []

        try:
            papers = await self._arxiv.fetch_papers(
                search_query=query,
                max_results=5,
                sort_by="relevance",
                sort_order="descending",
                skip_default_category=True,
            )
        except Exception as e:
            logger.warning("arXiv search failed in RAG chain: %s", e)
            return []

        sources = []
        for paper in papers:
            sources.append(
                SourceReference(
                    index=0,  # re-indexed later
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    authors=paper.authors,
                    arxiv_url=f"https://arxiv.org/abs/{paper.arxiv_id}",
                    url=f"https://arxiv.org/abs/{paper.arxiv_id}",
                    chunk_text=paper.abstract,
                    score=0.5,
                    source_type="arxiv",
                )
            )

        return sources

    async def _search_web(self, query: str) -> list[SourceReference]:
        """Search the web via DuckDuckGo for supplementary information."""
        if not self._web_search:
            return []

        try:
            response = await self._web_search.search(f"{query} research paper", max_results=5)
        except Exception as e:
            logger.warning("Web search failed in RAG chain: %s", e)
            return []

        sources = []
        for result in response.results:
            sources.append(
                SourceReference(
                    index=0,  # re-indexed later
                    title=result.title,
                    url=result.url,
                    chunk_text=result.snippet,
                    score=0.3,
                    source_type="web",
                )
            )

        return sources

    # ------------------------------------------------------------------
    # Main query methods
    # ------------------------------------------------------------------

    async def aquery(
        self,
        query: str,
        *,
        top_k: int | None = None,
        categories: list[str] | None = None,
        temperature: float | None = None,
    ) -> RAGResponse:
        """Run the full RAG pipeline: multi-source retrieve → prompt → generate."""
        query = self._validate_query(query)

        # Stage 1: Retrieve from all sources
        all_sources, kb_result = await self._retrieve_all_sources(
            query, top_k=top_k, categories=categories
        )

        # Stage 2: Check for empty results
        if not all_sources:
            return self._empty_response(query, kb_result)

        # Stage 3: Build prompt and generate
        prompt = self._build_mixed_prompt(query, all_sources)

        llm_response = await self._llm.generate(prompt, temperature=temperature)

        raw_response = RAGResponse(
            answer=llm_response.text,
            sources=all_sources,
            query=query,
            retrieval_metadata=RetrievalMetadata(
                stages_executed=kb_result.stages_executed,
                total_candidates=kb_result.total_candidates,
                timings=kb_result.timings,
                expanded_queries=kb_result.expanded_queries,
            ),
            llm_metadata=LLMMetadata(
                provider=llm_response.provider,
                model=llm_response.model,
                prompt_tokens=llm_response.usage.prompt_tokens if llm_response.usage else None,
                completion_tokens=llm_response.usage.completion_tokens if llm_response.usage else None,
                total_tokens=llm_response.usage.total_tokens if llm_response.usage else None,
                latency_ms=llm_response.usage.latency_ms if llm_response.usage else None,
            ),
        )

        # S5.5: Enforce citations — parse, validate, format source list
        citation_result = enforce_citations(raw_response)
        raw_response.answer = citation_result.formatted_answer

        # Filter sources to only those actually cited in the answer
        if citation_result.validation.valid_citations:
            cited_set = set(citation_result.validation.valid_citations)
            raw_response.sources = [s for s in raw_response.sources if s.index in cited_set]
        else:
            # Keep top 3 sources even if LLM didn't cite them
            raw_response.sources = all_sources[:3]

        return raw_response

    async def aquery_stream(
        self,
        query: str,
        *,
        top_k: int | None = None,
        categories: list[str] | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream RAG response tokens, ending with a [SOURCES] JSON block."""
        query = self._validate_query(query)

        # Stage 1: Retrieve from all sources
        all_sources, kb_result = await self._retrieve_all_sources(
            query, top_k=top_k, categories=categories
        )

        # Stage 2: Check for empty results
        if not all_sources:
            yield _NO_RESULTS_MESSAGE
            return

        # Stage 3: Build prompt and stream
        prompt = self._build_mixed_prompt(query, all_sources)

        # S5.5: Wrap LLM token stream with citation enforcement
        llm_stream = self._llm.generate_stream(prompt, temperature=temperature)
        citation_stream = stream_with_citations(llm_stream, all_sources)

        async for token in citation_stream:
            yield token

        # Filter sources to only those actually cited in the streamed answer
        sources = all_sources
        if citation_stream.validation and citation_stream.validation.valid_citations:
            cited_set = set(citation_stream.validation.valid_citations)
            sources = [s for s in all_sources if s.index in cited_set]

        # Always include at least top 3 sources
        if len(sources) < 3:
            sources = all_sources[:3]

        # Yield sources as final JSON block
        sources_json = json.dumps([s.model_dump() for s in sources])
        yield f"\n\n[SOURCES]{sources_json}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_query(query: str) -> str:
        """Validate and clean the query string."""
        query = query.strip()
        if not query:
            msg = "Query must not be empty"
            raise ValueError(msg)
        return query

    @staticmethod
    def _build_sources(hits: list[SearchHit], source_type: str = "knowledge_base") -> list[SourceReference]:
        """Convert SearchHit list to numbered SourceReference list, deduplicated by paper.

        Multiple chunks from the same paper are collapsed into one source entry,
        keeping the highest score and merging chunk texts.
        """
        seen: dict[str, SourceReference] = {}
        for hit in hits:
            key = hit.arxiv_id or hit.chunk_id
            if key in seen:
                existing = seen[key]
                existing.chunk_text = (existing.chunk_text or "") + "\n\n" + (hit.chunk_text or "")
                if hit.score > (existing.score or 0.0):
                    existing.score = hit.score
            else:
                seen[key] = SourceReference(
                    index=0,
                    arxiv_id=hit.arxiv_id or "",
                    title=hit.title or "Unknown Title",
                    authors=hit.authors if hit.authors else [],
                    arxiv_url=f"https://arxiv.org/abs/{hit.arxiv_id}" if hit.arxiv_id else "",
                    url=f"https://arxiv.org/abs/{hit.arxiv_id}" if hit.arxiv_id else "",
                    chunk_text=hit.chunk_text or "",
                    score=hit.score,
                    source_type=source_type,
                )

        sources = list(seen.values())
        for i, src in enumerate(sources, start=1):
            src.index = i

        return sources

    @staticmethod
    def _build_mixed_prompt(query: str, sources: list[SourceReference]) -> str:
        """Combine system prompt + user prompt for mixed source types."""
        user_prompt = build_mixed_user_prompt(query=query, sources=sources)
        return f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    @staticmethod
    def _build_full_prompt(query: str, hits: list[SearchHit]) -> str:
        """Combine system prompt + user prompt into a single prompt string (legacy KB-only)."""
        user_prompt = build_user_prompt(query=query, search_hits=hits)
        return f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    @staticmethod
    def _empty_response(query: str, retrieval_result: RetrievalResult) -> RAGResponse:
        """Build a graceful response when no documents are found."""
        return RAGResponse(
            answer=_NO_RESULTS_MESSAGE,
            sources=[],
            query=query,
            retrieval_metadata=RetrievalMetadata(
                stages_executed=retrieval_result.stages_executed,
                total_candidates=retrieval_result.total_candidates,
                timings=retrieval_result.timings,
                expanded_queries=retrieval_result.expanded_queries,
            ),
        )
