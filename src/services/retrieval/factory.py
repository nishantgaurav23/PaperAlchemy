"""Factory functions for retrieval services."""

from __future__ import annotations

from src.config import HyDESettings, MultiQuerySettings, RetrievalPipelineSettings
from src.services.embeddings.client import JinaEmbeddingsClient
from src.services.indexing.parent_child import ParentChildChunker
from src.services.llm.provider import LLMProvider
from src.services.opensearch.client import OpenSearchClient
from src.services.reranking.service import RerankerService
from src.services.retrieval.hyde import HyDEService
from src.services.retrieval.multi_query import MultiQueryService
from src.services.retrieval.pipeline import RetrievalPipeline


def create_hyde_service(
    *,
    settings: HyDESettings,
    llm_provider: LLMProvider,
    embeddings_client: JinaEmbeddingsClient,
    opensearch_client: OpenSearchClient,
) -> HyDEService:
    """Create a HyDEService with the given dependencies."""
    return HyDEService(
        settings=settings,
        llm_provider=llm_provider,
        embeddings_client=embeddings_client,
        opensearch_client=opensearch_client,
    )


def create_multi_query_service(
    *,
    settings: MultiQuerySettings,
    llm_provider: LLMProvider,
    embeddings_client: JinaEmbeddingsClient,
    opensearch_client: OpenSearchClient,
) -> MultiQueryService:
    """Create a MultiQueryService with the given dependencies."""
    return MultiQueryService(
        settings=settings,
        llm_provider=llm_provider,
        embeddings_client=embeddings_client,
        opensearch_client=opensearch_client,
    )


def create_retrieval_pipeline(
    *,
    settings: RetrievalPipelineSettings,
    multi_query_service: MultiQueryService,
    hyde_service: HyDEService,
    reranker_service: RerankerService,
    parent_child_chunker: ParentChildChunker,
    opensearch_client: OpenSearchClient,
    embeddings_client: JinaEmbeddingsClient,
) -> RetrievalPipeline:
    """Create a RetrievalPipeline with all sub-service dependencies."""
    return RetrievalPipeline(
        settings=settings,
        multi_query_service=multi_query_service,
        hyde_service=hyde_service,
        reranker_service=reranker_service,
        parent_child_chunker=parent_child_chunker,
        opensearch_client=opensearch_client,
        embeddings_client=embeddings_client,
    )
