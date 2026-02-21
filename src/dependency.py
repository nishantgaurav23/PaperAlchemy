"""
FastAPI dependency injection for PaperAlchemy services.

Why it's needed:
    Router functions need access to database connections, OpenSearch clients,
    and settings — but shouldn't create them directly. Dependency injection
    lets FastAPI automatically provide these objects to any route handler
    that declares them as parameters, keeping routers thin and testable.

What it does:
    - get_settings(): Reads Settings from app.state (set during lifespan startup)
    - get_database(): Reads Database from app.state
    - get_db_session(): Opens a SQLAlchemy session scoped to one request (yields + auto-closes)
    - get_opensearch_client(): Reads OpenSearchClient from app.state
    - Annotated type aliases (SettingsDep, DatabaseDep, etc.) let routers
      declare dependencies concisely: `def search(client: OpenSearchDep)`

How it helps:
    - Routers don't import factories or create clients — they just declare what they need
    - Testing: override dependencies with mocks using app.dependency_overrides
    - Lifecycle: sessions are opened per-request and closed automatically
    - Type safety: IDE autocomplete works because Annotated preserves the type
"""

from typing import Annotated, Generator, Optional

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from src.config import Settings
from src.db.database import Database
from src.services.agents.agentic_rag import AgenticRAGService
from src.services.opensearch.client import OpenSearchClient

# Future week dependencies (uncomment as services are added)
from src.services.arxiv.client import ArxivClient
from src.services.pdf_parser.service import PDFParserService
from src.services.embeddings.jina_client import JinaEmbeddingsClient
# from src.services.embeddings.client import EmbeddingsClient
from src.services.ollama.client import OllamaClient
from src.services.langfuse.client import LangfuseTracer
from src.services.cache.client import CacheClient

def get_settings(request: Request) -> Settings:
    """Get settings from the request state."""
    return request.app.state.settings

def get_database(request: Request) -> Database:
    """Get database from the request state."""
    return request.app.state.database

def get_db_session(database: Annotated[Database, Depends(get_database)]) -> Generator[Session, None, None]:
    """Get database session dependency."""
    with database.get_session() as session:
        yield session

def get_opensearch_client(request: Request) -> OpenSearchClient:
    """Get OpenSearch client from the request state."""
    return request.app.state.opensearch_client

# Future week dependencies (uncomment as services are added)
def get_arxiv_client(request: Request) -> ArxivClient:
    return request.app.state.arxiv_client

def get_pdf_parser(request: Request) -> PDFParserService:
    return request.app.state.pdf_parser

def get_embeddings_service(request: Request) -> JinaEmbeddingsClient:
     """Get Jina embeddings client from the request state.
     
     Set during lifespan startup via make_embeddings_service().
     Used by the hybrid search router to embed queries at searh time.
     """
     return request.app.state.embeddings_service

def get_ollama_client(request: Request) -> OllamaClient:
    """Get Ollama client from the request state.
    Set during lifespan startup. Used by the /ask router
    to generate RAG answers from retrieved paper chunks.
    """
    return request.app.state.ollama_client

def get_langfuse_tracer(request: Request) -> LangfuseTracer:
    """Get Langfuse tracer from the request state.
    Set during lifespan startup. Returns a disabled tracer
    if Langfuse is not configured.
    """
    return request.app.state.langfuse_tracer

def get_cache_client(request: Request) -> Optional[CacheClient]:
    """Get cache client from the request state.
    Returns None if Redis is unavailable (graceful degradation)
    """
    return getattr(request.app.state, "cache_client", None)

def get_agentic_rag_service(request: Request) -> AgenticRAGService:
    """Get AgenticRAGService from the request state.
    Set during lifespan startup via make_agentic_rag_service().
    Contains the compiled LangGraph workflow for agentic RAG.
    """
    return request.app.state.agentic_rag_service




# Dependency annotations
SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[Database, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
OpenSearchDep = Annotated[OpenSearchClient, Depends(get_opensearch_client)]
# Future week dependencies (uncomment as services are added)
ArxivDep = Annotated[ArxivClient, Depends(get_arxiv_client)]
PDFParserDep = Annotated[PDFParserService, Depends(get_pdf_parser)]
EmbeddingsDep = Annotated[JinaEmbeddingsClient, Depends(get_embeddings_service)]
OllamaDep = Annotated[OllamaClient, Depends(get_ollama_client)]
LangfuseDep = Annotated[LangfuseTracer, Depends(get_langfuse_tracer)]
CacheDep = Annotated[Optional[CacheClient], Depends(get_cache_client)]
AgenticRAGDep = Annotated[AgenticRAGService, Depends(get_agentic_rag_service)]