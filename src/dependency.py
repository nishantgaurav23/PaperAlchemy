"""FastAPI dependency injection for PaperAlchemy services.

Provides typed Annotated[] aliases so routers declare dependencies concisely:
    async def my_route(settings: SettingsDep, session: SessionDep): ...

Dependencies are overridable via app.dependency_overrides for testing.
New services are added here as future specs implement them.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings, get_settings
from src.db import _get_database, get_db_session
from src.db.database import Database
from src.repositories.paper import PaperRepository
from src.services.arxiv.client import ArxivClient
from src.services.arxiv.factory import make_arxiv_client
from src.services.cache.client import CacheClient
from src.services.chat.follow_up import FollowUpHandler
from src.services.chat.memory import ConversationMemory
from src.services.embeddings.client import JinaEmbeddingsClient
from src.services.embeddings.factory import make_embeddings_client
from src.services.llm.factory import create_llm_provider
from src.services.llm.provider import LLMProvider
from src.services.opensearch.client import OpenSearchClient
from src.services.opensearch.factory import make_opensearch_client
from src.services.rag.chain import RAGChain
from src.services.rag.factory import create_rag_chain
from src.services.reranking.factory import create_reranker_service
from src.services.reranking.service import RerankerService
from src.services.retrieval.factory import create_hyde_service, create_multi_query_service, create_retrieval_pipeline
from src.services.retrieval.hyde import HyDEService
from src.services.retrieval.multi_query import MultiQueryService
from src.services.retrieval.pipeline import RetrievalPipeline


def get_database() -> Database:
    """Return the singleton Database instance. Raises RuntimeError if not initialised."""
    return _get_database()


# ---------------------------------------------------------------------------
# Annotated type aliases — use these in router function signatures
# ---------------------------------------------------------------------------

SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[Database, Depends(get_database)]
SessionDep = Annotated[AsyncSession, Depends(get_db_session)]


def get_paper_repository(session: SessionDep) -> PaperRepository:
    """Return a PaperRepository bound to the current session."""
    return PaperRepository(session)


PaperRepoDep = Annotated[PaperRepository, Depends(get_paper_repository)]

ArxivClientDep = Annotated[ArxivClient, Depends(make_arxiv_client)]


def get_opensearch_client() -> OpenSearchClient:
    """No-arg wrapper so FastAPI doesn't try to resolve factory params."""
    return make_opensearch_client()


def get_embeddings_client() -> JinaEmbeddingsClient:
    """No-arg wrapper so FastAPI doesn't try to resolve factory params."""
    return make_embeddings_client()


OpenSearchDep = Annotated[OpenSearchClient, Depends(get_opensearch_client)]

EmbeddingsDep = Annotated[JinaEmbeddingsClient, Depends(get_embeddings_client)]


def get_reranker_service() -> RerankerService:
    """No-arg wrapper so FastAPI doesn't try to resolve factory params."""
    return create_reranker_service()


RerankerDep = Annotated[RerankerService, Depends(get_reranker_service)]


def get_llm_provider(settings: SettingsDep) -> LLMProvider:
    """Return the preferred LLM provider based on configuration."""
    return create_llm_provider(settings)


LLMProviderDep = Annotated[LLMProvider, Depends(get_llm_provider)]


def get_hyde_service(
    settings: SettingsDep,
    llm_provider: LLMProviderDep,
    embeddings: EmbeddingsDep,
    opensearch: OpenSearchDep,
) -> HyDEService:
    """Create HyDE service with injected dependencies."""
    return create_hyde_service(
        settings=settings.hyde,
        llm_provider=llm_provider,
        embeddings_client=embeddings,
        opensearch_client=opensearch,
    )


HyDEDep = Annotated[HyDEService, Depends(get_hyde_service)]


def get_multi_query_service(
    settings: SettingsDep,
    llm_provider: LLMProviderDep,
    embeddings: EmbeddingsDep,
    opensearch: OpenSearchDep,
) -> MultiQueryService:
    """Create MultiQueryService with injected dependencies."""
    return create_multi_query_service(
        settings=settings.multi_query,
        llm_provider=llm_provider,
        embeddings_client=embeddings,
        opensearch_client=opensearch,
    )


MultiQueryDep = Annotated[MultiQueryService, Depends(get_multi_query_service)]


def get_retrieval_pipeline(
    settings: SettingsDep,
    multi_query: MultiQueryDep,
    hyde: HyDEDep,
    reranker: RerankerDep,
    opensearch: OpenSearchDep,
    embeddings: EmbeddingsDep,
) -> RetrievalPipeline:
    """Create RetrievalPipeline with injected dependencies."""
    from src.services.indexing.parent_child import ParentChildChunker

    return create_retrieval_pipeline(
        settings=settings.retrieval_pipeline,
        multi_query_service=multi_query,
        hyde_service=hyde,
        reranker_service=reranker,
        parent_child_chunker=ParentChildChunker(),
        opensearch_client=opensearch,
        embeddings_client=embeddings,
    )


RetrievalPipelineDep = Annotated[RetrievalPipeline, Depends(get_retrieval_pipeline)]


def get_rag_chain(
    llm_provider: LLMProviderDep,
    retrieval_pipeline: RetrievalPipelineDep,
) -> RAGChain:
    """Create RAGChain with injected dependencies."""
    return create_rag_chain(
        llm_provider=llm_provider,
        retrieval_pipeline=retrieval_pipeline,
    )


RAGChainDep = Annotated[RAGChain, Depends(get_rag_chain)]


# ---------------------------------------------------------------------------
# Cache (S5.4) — singleton set at startup, None if Redis unavailable
# ---------------------------------------------------------------------------

_cache_client: CacheClient | None = None


def set_cache_client(client: CacheClient | None) -> None:
    """Set the global cache client (called during app lifespan)."""
    global _cache_client  # noqa: PLW0603
    _cache_client = client


def get_cache_client() -> CacheClient | None:
    """Return the cache client (may be None if Redis is unavailable)."""
    return _cache_client


CacheDep = Annotated[CacheClient | None, Depends(get_cache_client)]


# ---------------------------------------------------------------------------
# Conversation Memory (S7.1) — singleton set at startup, None if Redis unavailable
# ---------------------------------------------------------------------------

_conversation_memory: ConversationMemory | None = None


def set_conversation_memory(memory: ConversationMemory | None) -> None:
    """Set the global conversation memory (called during app lifespan)."""
    global _conversation_memory  # noqa: PLW0603
    _conversation_memory = memory


def get_conversation_memory() -> ConversationMemory | None:
    """Return the conversation memory (may be None if Redis is unavailable)."""
    return _conversation_memory


ConversationMemoryDep = Annotated[ConversationMemory | None, Depends(get_conversation_memory)]


# ---------------------------------------------------------------------------
# Follow-up Handler (S7.2) — depends on LLM, RAG, and ConversationMemory
# ---------------------------------------------------------------------------


def get_follow_up_handler(
    llm_provider: LLMProviderDep,
    rag_chain: RAGChainDep,
    memory: ConversationMemoryDep,
) -> FollowUpHandler:
    """Create FollowUpHandler with injected dependencies."""
    return FollowUpHandler(llm_provider=llm_provider, rag_chain=rag_chain, memory=memory)


FollowUpHandlerDep = Annotated[FollowUpHandler, Depends(get_follow_up_handler)]


__all__ = [
    "ArxivClientDep",
    "CacheDep",
    "ConversationMemoryDep",
    "FollowUpHandlerDep",
    "EmbeddingsDep",
    "HyDEDep",
    "LLMProviderDep",
    "MultiQueryDep",
    "OpenSearchDep",
    "RAGChainDep",
    "RerankerDep",
    "RetrievalPipelineDep",
    "DatabaseDep",
    "PaperRepoDep",
    "SessionDep",
    "SettingsDep",
    "get_cache_client",
    "get_conversation_memory",
    "get_database",
    "get_db_session",
    "get_embeddings_client",
    "get_hyde_service",
    "get_llm_provider",
    "get_multi_query_service",
    "get_opensearch_client",
    "get_paper_repository",
    "get_rag_chain",
    "get_reranker_service",
    "get_retrieval_pipeline",
    "get_settings",
    "set_conversation_memory",
]
