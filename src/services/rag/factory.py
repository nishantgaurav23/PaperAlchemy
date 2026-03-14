"""Factory function for RAG chain (S5.2)."""

from __future__ import annotations

from src.services.arxiv.client import ArxivClient
from src.services.llm.provider import LLMProvider
from src.services.rag.chain import RAGChain
from src.services.retrieval.pipeline import RetrievalPipeline
from src.services.web_search.service import WebSearchService


def create_rag_chain(
    *,
    llm_provider: LLMProvider,
    retrieval_pipeline: RetrievalPipeline,
    arxiv_client: ArxivClient | None = None,
    web_search_service: WebSearchService | None = None,
) -> RAGChain:
    """Create a RAGChain with the given dependencies."""
    return RAGChain(
        llm_provider=llm_provider,
        retrieval_pipeline=retrieval_pipeline,
        arxiv_client=arxiv_client,
        web_search_service=web_search_service,
    )
