"""HyDE (Hypothetical Document Embeddings) retrieval service.

Improves search recall by generating a hypothetical answer passage via LLM,
embedding that passage, then searching for real documents similar to it.

Flow: query -> LLM hypothetical doc -> embed -> KNN search -> real results
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.config import HyDESettings
from src.schemas.api.search import SearchHit
from src.services.embeddings.client import JinaEmbeddingsClient
from src.services.llm.provider import LLMProvider
from src.services.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)

_HYDE_PROMPT_TEMPLATE = """You are an academic research assistant. Given the following research question, \
write a short (150-200 word) passage that would appear in a real scientific paper answering this question. \
Write in a formal academic style with technical terminology. Do not include citations or references. \
Focus on providing a factual, informative answer as if from a published research paper.

Research question: {query}

Hypothetical passage:"""


@dataclass
class HyDEResult:
    """Result of a HyDE retrieval operation."""

    hypothetical_document: str = ""
    query_embedding: list[float] = field(default_factory=list)
    results: list[SearchHit] = field(default_factory=list)


class HyDEService:
    """Generates hypothetical documents via LLM and uses them for vector search."""

    def __init__(
        self,
        *,
        settings: HyDESettings,
        llm_provider: LLMProvider,
        embeddings_client: JinaEmbeddingsClient,
        opensearch_client: OpenSearchClient,
    ) -> None:
        self._settings = settings
        self._llm = llm_provider
        self._embeddings = embeddings_client
        self._opensearch = opensearch_client

    async def generate_hypothetical_document(self, query: str) -> str:
        """Generate a hypothetical academic passage answering the query.

        Returns the hypothetical passage, or falls back to the original query
        if LLM generation fails or returns empty.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty")

        prompt = _HYDE_PROMPT_TEMPLATE.format(query=query)

        try:
            response = await self._llm.generate(
                prompt,
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
            )
            if not response.text or not response.text.strip():
                logger.warning("LLM returned empty hypothetical document, falling back to original query")
                return query
            return response.text.strip()
        except Exception:
            logger.warning("HyDE generation failed, falling back to original query", exc_info=True)
            return query

    async def retrieve_with_hyde(self, query: str, *, top_k: int = 20) -> HyDEResult:
        """Run full HyDE retrieval: generate -> embed -> KNN search.

        Falls back to standard query embedding if hypothetical generation
        or embedding fails at any stage.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty")

        # Step 1: Generate hypothetical document (with fallback)
        hypothetical_doc = await self.generate_hypothetical_document(query)

        # Step 2: Embed the hypothetical document (with fallback)
        try:
            embedding = await self._embeddings.embed_query(hypothetical_doc)
        except Exception:
            logger.warning("Failed to embed hypothetical doc, falling back to original query embedding", exc_info=True)
            try:
                embedding = await self._embeddings.embed_query(query)
            except Exception:
                logger.error("Embedding fallback also failed", exc_info=True)
                return HyDEResult(hypothetical_document=hypothetical_doc)

        # Step 3: KNN search with the embedding
        raw_results = await self._opensearch.asearch_chunks_vectors(
            query_embedding=embedding,
            size=top_k,
        )

        hits = [SearchHit(**hit) for hit in raw_results.get("hits", [])]

        return HyDEResult(
            hypothetical_document=hypothetical_doc,
            query_embedding=embedding,
            results=hits,
        )
