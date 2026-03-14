"""Unified advanced retrieval pipeline (S4b.5).

Orchestrates: multi-query → hybrid search → re-rank → parent expansion → top-K.
Each stage can be independently enabled/disabled and gracefully degrades on failure.

This is the single entry point for document retrieval used by RAG chain and agents.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from src.config import RetrievalPipelineSettings
from src.schemas.api.search import SearchHit
from src.services.embeddings.client import JinaEmbeddingsClient
from src.services.indexing.parent_child import ParentChildChunker
from src.services.opensearch.client import OpenSearchClient
from src.services.reranking.service import RerankerService
from src.services.retrieval.hyde import HyDEResult, HyDEService
from src.services.retrieval.multi_query import MultiQueryResult, MultiQueryService

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Pipeline output with ranked results and metadata."""

    results: list[SearchHit] = field(default_factory=list)
    query: str = ""
    expanded_queries: list[str] = field(default_factory=list)
    hypothetical_document: str = ""
    stages_executed: list[str] = field(default_factory=list)
    total_candidates: int = 0
    timings: dict[str, float] = field(default_factory=dict)


class RetrievalPipeline:
    """Unified advanced retrieval pipeline composing all S4b sub-services."""

    def __init__(
        self,
        *,
        settings: RetrievalPipelineSettings,
        multi_query_service: MultiQueryService,
        hyde_service: HyDEService,
        reranker_service: RerankerService,
        parent_child_chunker: ParentChildChunker,
        opensearch_client: OpenSearchClient,
        embeddings_client: JinaEmbeddingsClient,
    ) -> None:
        self._settings = settings
        self._multi_query = multi_query_service
        self._hyde = hyde_service
        self._reranker = reranker_service
        self._parent_child = parent_child_chunker
        self._opensearch = opensearch_client
        self._embeddings = embeddings_client

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        categories: list[str] | None = None,
    ) -> RetrievalResult:
        """Run the full retrieval pipeline with graceful degradation.

        Stages:
        1. Query expansion (multi-query + HyDE, parallel)
        2. Hybrid search (fallback if expansion fails)
        3. Merge & deduplicate
        4. Re-rank
        5. Parent expansion
        """
        final_top_k = top_k or self._settings.final_top_k
        retrieval_top_k = self._settings.retrieval_top_k

        result = RetrievalResult(query=query)
        all_hits: list[SearchHit] = []

        # ── Stage 1: Query Expansion (parallel) ──────────────────────
        tasks = []
        task_labels = []

        if self._settings.multi_query_enabled:
            tasks.append(self._run_multi_query(query, retrieval_top_k))
            task_labels.append("multi_query")

        if self._settings.hyde_enabled:
            tasks.append(self._run_hyde(query, retrieval_top_k))
            task_labels.append("hyde")

        if tasks:
            start = time.monotonic()
            try:
                outcomes = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Query expansion timed out after 30s, skipping")
                outcomes = [TimeoutError("expansion timeout")] * len(tasks)
            expansion_time = time.monotonic() - start

            for label, outcome in zip(task_labels, outcomes, strict=True):
                if isinstance(outcome, Exception):
                    logger.warning("Stage %s failed: %s", label, outcome)
                    continue

                if label == "multi_query" and isinstance(outcome, MultiQueryResult):
                    result.expanded_queries = outcome.generated_queries
                    result.stages_executed.append("multi_query")
                    result.timings["multi_query"] = expansion_time
                    all_hits.extend(outcome.results)

                elif label == "hyde" and isinstance(outcome, HyDEResult):
                    result.hypothetical_document = outcome.hypothetical_document
                    result.stages_executed.append("hyde")
                    result.timings["hyde"] = expansion_time
                    all_hits.extend(outcome.results)

        # ── Stage 2: Hybrid Search (always runs as baseline) ─────────
        start = time.monotonic()
        try:
            embedding = await self._embeddings.embed_query(query)
            raw = await self._opensearch.asearch_chunks_hybrid(
                query=query,
                query_embedding=embedding,
                size=retrieval_top_k,
                categories=categories,
            )
            baseline_hits = [SearchHit(**hit) for hit in raw.get("hits", [])]
            all_hits.extend(baseline_hits)
        except Exception:
            logger.warning("Baseline hybrid search failed", exc_info=True)

        result.stages_executed.append("hybrid_search")
        result.timings["hybrid_search"] = time.monotonic() - start

        # ── Stage 3: Merge & Deduplicate ─────────────────────────────
        merged = self._deduplicate(all_hits)
        result.total_candidates = len(merged)

        # ── Stage 4: Re-rank ─────────────────────────────────────────
        if self._settings.reranker_enabled and merged:
            start = time.monotonic()
            try:
                merged = await self._reranker.rerank_search_hits(query, merged, top_k=final_top_k)
                result.stages_executed.append("rerank")
                result.timings["rerank"] = time.monotonic() - start
            except Exception:
                logger.warning("Re-ranking failed, using score-sorted results", exc_info=True)
                result.timings["rerank"] = time.monotonic() - start
                merged = sorted(merged, key=lambda h: h.score, reverse=True)[:final_top_k]
        else:
            merged = sorted(merged, key=lambda h: h.score, reverse=True)[:final_top_k]

        # ── Stage 5: Parent Expansion ────────────────────────────────
        if self._settings.parent_expansion_enabled and merged:
            start = time.monotonic()
            try:
                child_dicts = [h.model_dump() for h in merged]
                expanded = self._parent_child.expand_to_parents(child_dicts, self._opensearch)
                # Convert back to SearchHit objects
                merged = [SearchHit(**d) if isinstance(d, dict) else d for d in expanded]
                result.stages_executed.append("parent_expand")
                result.timings["parent_expand"] = time.monotonic() - start
            except Exception:
                logger.warning("Parent expansion failed, returning child chunks", exc_info=True)
                result.timings["parent_expand"] = time.monotonic() - start

        # ── Stage 6: Paper-level dedup ────────────────────────────
        merged = self._deduplicate_by_paper(merged)

        result.results = merged[:final_top_k]
        return result

    async def _run_multi_query(self, query: str, top_k: int) -> MultiQueryResult:
        """Run multi-query retrieval stage."""
        return await self._multi_query.retrieve_with_multi_query(query, top_k=top_k)

    async def _run_hyde(self, query: str, top_k: int) -> HyDEResult:
        """Run HyDE retrieval stage."""
        return await self._hyde.retrieve_with_hyde(query, top_k=top_k)

    @staticmethod
    def _deduplicate(hits: list[SearchHit]) -> list[SearchHit]:
        """Deduplicate hits by chunk_id, keeping the highest score."""
        best: dict[str, SearchHit] = {}
        for hit in hits:
            cid = hit.chunk_id
            if cid not in best or hit.score > best[cid].score:
                best[cid] = hit
        return list(best.values())

    @staticmethod
    def _deduplicate_by_paper(hits: list[SearchHit]) -> list[SearchHit]:
        """Deduplicate hits by arxiv_id, keeping the highest-scoring chunk per paper.

        Multiple chunks from the same paper are collapsed into one entry so that
        the final source list shows each paper only once.
        """
        best: dict[str, SearchHit] = {}
        for hit in hits:
            key = hit.arxiv_id or hit.chunk_id  # fall back to chunk_id if no arxiv_id
            if key not in best or hit.score > best[key].score:
                best[key] = hit
        return list(best.values())
