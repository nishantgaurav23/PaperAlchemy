"""RAG chain service (S5.2).

Orchestrates: retrieve → prompt → generate, with citation enforcement.
Supports both single-shot (aquery) and streaming (aquery_stream) modes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from src.schemas.api.search import SearchHit
from src.services.llm.provider import LLMProvider
from src.services.rag.citation import enforce_citations, stream_with_citations
from src.services.rag.models import LLMMetadata, RAGResponse, RetrievalMetadata, SourceReference
from src.services.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from src.services.retrieval.pipeline import RetrievalPipeline, RetrievalResult

logger = logging.getLogger(__name__)

_NO_PAPERS_MESSAGE = (
    "I could not find any relevant papers in the knowledge base for your query. "
    "Try rephrasing your question or broadening your search terms."
)


class RAGChain:
    """Retrieval-Augmented Generation chain with citation enforcement."""

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        retrieval_pipeline: RetrievalPipeline,
    ) -> None:
        self._llm = llm_provider
        self._retrieval = retrieval_pipeline

    async def aquery(
        self,
        query: str,
        *,
        top_k: int | None = None,
        categories: list[str] | None = None,
        temperature: float | None = None,
    ) -> RAGResponse:
        """Run the full RAG pipeline: retrieve → prompt → generate."""
        query = self._validate_query(query)

        # Stage 1: Retrieve
        retrieval_result = await self._retrieval.retrieve(query, top_k=top_k, categories=categories)

        # Stage 2: Check for empty results
        if not retrieval_result.results:
            return self._empty_response(query, retrieval_result)

        # Stage 3: Build prompt and generate
        sources = self._build_sources(retrieval_result.results)
        prompt = self._build_full_prompt(query, retrieval_result.results)

        llm_response = await self._llm.generate(prompt, temperature=temperature)

        raw_response = RAGResponse(
            answer=llm_response.text,
            sources=sources,
            query=query,
            retrieval_metadata=RetrievalMetadata(
                stages_executed=retrieval_result.stages_executed,
                total_candidates=retrieval_result.total_candidates,
                timings=retrieval_result.timings,
                expanded_queries=retrieval_result.expanded_queries,
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

        # Stage 1: Retrieve
        retrieval_result = await self._retrieval.retrieve(query, top_k=top_k, categories=categories)

        # Stage 2: Check for empty results
        if not retrieval_result.results:
            yield _NO_PAPERS_MESSAGE
            return

        # Stage 3: Build prompt and stream
        sources = self._build_sources(retrieval_result.results)
        prompt = self._build_full_prompt(query, retrieval_result.results)

        # S5.5: Wrap LLM token stream with citation enforcement
        llm_stream = self._llm.generate_stream(prompt, temperature=temperature)
        citation_stream = stream_with_citations(llm_stream, sources)

        async for token in citation_stream:
            yield token

        # Yield sources as final JSON block
        sources_json = json.dumps([s.model_dump() for s in sources])
        yield f"\n\n[SOURCES]{sources_json}"

    @staticmethod
    def _validate_query(query: str) -> str:
        """Validate and clean the query string."""
        query = query.strip()
        if not query:
            msg = "Query must not be empty"
            raise ValueError(msg)
        return query

    @staticmethod
    def _build_sources(hits: list[SearchHit]) -> list[SourceReference]:
        """Convert SearchHit list to numbered SourceReference list."""
        sources = []
        for i, hit in enumerate(hits, start=1):
            sources.append(
                SourceReference(
                    index=i,
                    arxiv_id=hit.arxiv_id or "",
                    title=hit.title or "Unknown Title",
                    authors=hit.authors if hit.authors else [],
                    arxiv_url=f"https://arxiv.org/abs/{hit.arxiv_id}" if hit.arxiv_id else "",
                    chunk_text=hit.chunk_text or "",
                    score=hit.score,
                )
            )
        return sources

    @staticmethod
    def _build_full_prompt(query: str, hits: list[SearchHit]) -> str:
        """Combine system prompt + user prompt into a single prompt string."""
        user_prompt = build_user_prompt(query=query, search_hits=hits)
        return f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    @staticmethod
    def _empty_response(query: str, retrieval_result: RetrievalResult) -> RAGResponse:
        """Build a graceful response when no documents are found."""
        return RAGResponse(
            answer=_NO_PAPERS_MESSAGE,
            sources=[],
            query=query,
            retrieval_metadata=RetrievalMetadata(
                stages_executed=retrieval_result.stages_executed,
                total_candidates=retrieval_result.total_candidates,
                timings=retrieval_result.timings,
                expanded_queries=retrieval_result.expanded_queries,
            ),
        )
