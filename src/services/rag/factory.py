"""Factory function for RAG chain (S5.2)."""

from __future__ import annotations

from src.services.llm.provider import LLMProvider
from src.services.rag.chain import RAGChain
from src.services.retrieval.pipeline import RetrievalPipeline


def create_rag_chain(
    *,
    llm_provider: LLMProvider,
    retrieval_pipeline: RetrievalPipeline,
) -> RAGChain:
    """Create a RAGChain with the given dependencies."""
    return RAGChain(
        llm_provider=llm_provider,
        retrieval_pipeline=retrieval_pipeline,
    )
