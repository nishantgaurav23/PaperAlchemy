"""Multi-query retrieval service.

Improves search recall by generating multiple query variations via LLM,
running parallel hybrid searches, deduplicating by chunk_id, and fusing
results using Reciprocal Rank Fusion (RRF).

Flow: query -> LLM variations -> parallel hybrid search -> dedup -> RRF -> top-K
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

from src.config import MultiQuerySettings
from src.schemas.api.search import SearchHit
from src.services.embeddings.client import JinaEmbeddingsClient
from src.services.llm.provider import LLMProvider
from src.services.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)

_MULTI_QUERY_PROMPT_TEMPLATE = """You are a research query expansion assistant. Given the following research question, \
generate {num_queries} alternative formulations that capture different aspects, synonyms, or angles of the same question. \
Each variation should be semantically distinct but preserve the original intent. \
Use different terminology, broader/narrower scope, or different perspectives.

Output each variation on a new line, numbered (1. 2. 3. etc.). Do not include any other text.

Original question: {query}

Alternative formulations:"""


@dataclass
class MultiQueryResult:
    """Result of a multi-query retrieval operation."""

    original_query: str = ""
    generated_queries: list[str] = field(default_factory=list)
    results: list[SearchHit] = field(default_factory=list)


class MultiQueryService:
    """Generates query variations via LLM and performs fused parallel retrieval."""

    def __init__(
        self,
        *,
        settings: MultiQuerySettings,
        llm_provider: LLMProvider,
        embeddings_client: JinaEmbeddingsClient,
        opensearch_client: OpenSearchClient,
    ) -> None:
        self._settings = settings
        self._llm = llm_provider
        self._embeddings = embeddings_client
        self._opensearch = opensearch_client

    async def generate_query_variations(self, query: str) -> list[str]:
        """Generate diverse query reformulations via LLM.

        Returns a list of query variations, or [original_query] on failure.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty")

        prompt = _MULTI_QUERY_PROMPT_TEMPLATE.format(
            query=query,
            num_queries=self._settings.num_queries,
        )

        try:
            response = await self._llm.generate(
                prompt,
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
            )
            if not response.text or not response.text.strip():
                logger.warning("LLM returned empty response for query variations, using original query")
                return [query]

            variations = self._parse_variations(response.text)
            if not variations:
                logger.warning("Failed to parse any query variations, using original query")
                return [query]

            return variations

        except Exception:
            logger.warning("Multi-query generation failed, using original query", exc_info=True)
            return [query]

    @staticmethod
    def _parse_variations(text: str) -> list[str]:
        """Parse numbered or bulleted list from LLM output into query strings."""
        lines = text.strip().split("\n")
        variations = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Strip numbered prefixes: "1. ", "1) ", "1 - ", "1: "
            cleaned = re.sub(r"^\d+[\.\)\-:]\s*", "", line)
            # Strip bullet prefixes: "- ", "* ", "• "
            cleaned = re.sub(r"^[\-\*•]\s*", "", cleaned)
            cleaned = cleaned.strip()
            if cleaned:
                variations.append(cleaned)
        return variations

    async def retrieve_with_multi_query(self, query: str, *, top_k: int = 20) -> MultiQueryResult:
        """Run full multi-query retrieval: variations -> parallel search -> dedup -> RRF.

        Falls back to single query search if variation generation fails.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty")

        # Step 1: Generate query variations
        variations = await self.generate_query_variations(query)

        # Step 2: Parallel hybrid search for each variation
        per_query_results = await self._parallel_search(variations, top_k=top_k)

        # Step 3: RRF fusion + deduplication
        fused = self._rrf_fuse(per_query_results, top_k=top_k)

        return MultiQueryResult(
            original_query=query,
            generated_queries=variations,
            results=fused,
        )

    async def _parallel_search(self, queries: list[str], *, top_k: int = 20) -> list[list[SearchHit]]:
        """Execute hybrid search for each query variation concurrently."""

        async def _search_one(q: str) -> list[SearchHit]:
            try:
                embedding = await self._embeddings.embed_query(q)
                raw = await self._opensearch.asearch_chunks_hybrid(
                    query=q,
                    query_embedding=embedding,
                    size=top_k,
                )
                return [SearchHit(**hit) for hit in raw.get("hits", [])]
            except Exception:
                logger.warning("Search failed for query variation: %s", q, exc_info=True)
                return []

        results = await asyncio.gather(*[_search_one(q) for q in queries])
        return list(results)

    def _rrf_fuse(self, per_query_results: list[list[SearchHit]], *, top_k: int = 20) -> list[SearchHit]:
        """Fuse results from multiple queries using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) across all query result lists.
        Deduplicates by chunk_id, keeping the hit with the best original score.
        """
        k = self._settings.rrf_k
        rrf_scores: dict[str, float] = {}
        best_hits: dict[str, SearchHit] = {}

        for query_results in per_query_results:
            for rank, hit in enumerate(query_results, start=1):
                cid = hit.chunk_id
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)

                # Keep the hit with the highest original score for metadata
                if cid not in best_hits or hit.score > best_hits[cid].score:
                    best_hits[cid] = hit

        # Sort by RRF score descending, update score field
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        fused: list[SearchHit] = []
        for cid in sorted_ids[:top_k]:
            hit = best_hits[cid].model_copy()
            hit.score = rrf_scores[cid]
            fused.append(hit)

        return fused
