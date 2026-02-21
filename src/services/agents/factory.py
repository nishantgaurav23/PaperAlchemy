"""
What is needed:
    A factory function that wires up AgenticRAGService from PaperAlchemy's
    existing service clients. Called once from main.py's lifespan startup.

Why it is needed:
    AgenticRAGService needs 4-5 dependencies (opensearch, ollama, embeddings,
    langfuse, config). Constructing it inline in main.py would make the
    lifespan function hard to read. A factory encapsulates the construction
    logic in one place, consistent with how other services are created
    (make_opensearch_client, make_ollama_client, etc.).

How it helps:
    # In main.py lifespan:
    app.state.agentic_rag_service = make_agentic_rag_service(
        opensearch_client=opensearch_client,
        ollama_client=app.state.ollama_client,
        embeddings_client=app.state.embeddings_service,
        langfuse_tracer=app.state.langfuse_tracer,
    )
    # AgenticRAGService.__init__ compiles the LangGraph graph here (once).
    # All subsequent requests reuse the compiled graph.

____________________________________________________________________________________________________________
What it does:
    Defines make_agentic_rag_service() — a thin factory that creates a
    GraphConfig with the provided parameters, then passes it along with all
    service clients to AgenticRAGService.

┌────────────────────────────┬──────────────────────────────────────────────────────────────┐
│          Function          │                         Purpose                              │
├────────────────────────────┼──────────────────────────────────────────────────────────────┤
│ make_agentic_rag_service   │ Factory: assembles GraphConfig + AgenticRAGService from       │
│                            │ injected client instances and optional overrides.             │
└────────────────────────────┴──────────────────────────────────────────────────────────────┘

Why factory function (not class method):
    Follows PaperAlchemy's existing pattern — every service has a make_*()
    factory (make_opensearch_client, make_ollama_client, make_langfuse_tracer).
    This makes main.py's lifespan consistent: all services are created with
    make_*() calls.

Why GraphConfig created here (not in main.py):
    Graph configuration (top_k, use_hybrid, max_retrieval_attempts, etc.)
    belongs with the agent service, not the application entry point.
    main.py passes raw overrides (top_k=3); this factory wraps them in a
    typed GraphConfig with all defaults applied.

"""

import logging
from typing import Optional

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient

from .agentic_rag import AgenticRAGService
from .config import GraphConfig

logger = logging.getLogger(__name__)

def make_agentic_rag_service(
        opensearch_client: OpenSearchClient,
        ollama_client: OllamaClient,
        embeddings_client: JinaEmbeddingsClient,
        langfuse_tracer: Optional[LangfuseTracer] = None,
        top_k: int = 3,
        use_hybrid: bool = True,
) -> AgenticRAGService:
    """Create a fully initialized AgenticRAGService.

    What it does:
        1. Builds a GraphConfig with the porvided top_k and use_hybrid values
            (other config fields default from Settings via GraphConfig.__init__)
        2. Passes all service clients + config to AgenticRAGService
        3. AgenticRAGServie compiles the LangGraph graph during __init__
        4. Returns the ready-to-use service

    Why called at startup:
        Graph compilation (~50ms) happens once here. All requests reuse 
        the compiled graph via app.state.agentic_rag_service.

    Args:
        opensearch_client: Initialised OpenSearch client for paper retrieval
        ollama_client: Initialised Ollama client for LLM calls
        embeddings_client: Initialised Jina client for query embedding
        lagfuse_tracer: Optional LangfuseTracer - None disables all tracing
        top_k: Number of paper chunks to retrieve per search (default 3)
        use_hybrid: True =BM25 + vector hybrid search; False = BM25 only

    Returns:
        AgenticRAGService with compiled LangGraph graph, ready for ask()
    """
    logger.info(f"Creating AgenticRAGService (top_k={top_k}, use_hybrid={use_hybrid})")

    graph_config = GraphConfig(
        top_k=top_k,
        use_hybrid=use_hybrid
    )

    service = AgenticRAGService(
        opensearch_client=opensearch_client,
        ollama_client=ollama_client,
        embeddings_client=embeddings_client,
        langfuse_tracer=langfuse_tracer,
        graph_config=graph_config,
    )

    logger.info("AgenticRAGService created successfully")
    return service