"""
What it does: Defines the Context dataclass — the runtime dependency container injected into every LangGraph node. This is  
how nodes access services (Ollama, OpenSearch, Jina) without importing them directly or using global state.
Field: ollama_client                                                                                                        
Type: OllamaClient                                                                                                          
Purpose: LLM generation (guardrail scoring, grading, rewriting, answer generation)                                          
────────────────────────────────────────                                                                                    
Field: opensearch_client
Type: OpenSearchClient
Purpose: Document search (used by retriever tool)
────────────────────────────────────────
Field: embeddings_client
Type: JinaEmbeddingsClient
Purpose: Query→vector embedding (used by retriever tool)
────────────────────────────────────────
Field: langfuse_tracer
Type: Optional[LangfuseTracer]
Purpose: Tracing — None if disabled, nodes check before creating spans
────────────────────────────────────────
Field: trace
Type: Optional[Any]
Purpose: Active Langfuse trace/span for the current request
────────────────────────────────────────
Field: langfuse_enabled
Type: bool
Purpose: Quick boolean check — avoids tracer is not None and tracer.client is not None everywhere
────────────────────────────────────────
Field: model_name
Type: str
Purpose: Which Ollama model to use (overridable per-request)
────────────────────────────────────────
Field: temperature
Type: float
Purpose: Generation temperature for the answer node
────────────────────────────────────────
Field: top_k
Type: int
Purpose: Number of documents to retrieve
────────────────────────────────────────
Field: max_retrieval_attempts
Type: int
Purpose: Max retrieve→grade→rewrite loops before giving up
────────────────────────────────────────
Field: guardrail_threshold
Type: int
Purpose: Minimum guardrail score to proceed with retrieval
Why a dataclass and not just dict: Type safety. Every node accesses context.ollama_client with IDE autocompletion. A dict
would require context["ollama_client"] with zero type checking.

How LangGraph injects it: The StateGraph(AgentState, context_schema=Context) declaration tells LangGraph that nodes can
accept a context parameter. At invocation time, graph.ainvoke(state, context=runtime_context) passes the Context instance to
every node.

Runtime context for LangGraph node dependency injection

Problem this solves:
    LangGraph nodes are pure aync functions with signature:
        async def my_node(state: AgentState, *, context: context) -> dict

    But nodes need access to services (Ollama, OpenSearch, Jina, Langfuse).
    Without Context, you'd need either.
    - Global singletons (unteastable, no per-request config)
    - Closures capturing services (verbose, hard to type-check)
    - Passing services through AgentState (pollutes state with non-data)

How it works:
    1. AgenticRAGService creates a Context instance per request, populating
       it with the shared servies clients + per-request config (model, top_k)
    2. LangGraph's context_schema=Context declaration enables injection
    3. Evry node receives the same Context instance via keyword argument
    4. Nodes access services: context.ollama_client.generate(...)

Why dataclass over Pydantic:
    - Context holds live client instances (OllamaClient, etc.) that aren't
      serializable. pydantic would try to validate/serialize them.
    - Dataclass is a thin wrapper - zero overhead, no validation needed
      since we control the construction in AgenticRAGService.
    - slots=True gives faster attribution access (relevant when every node reads
       context on every invocation).
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient

@dataclass(slots=True)
class Context:
    """Runtime context injected into every LangGraph node.
    
    Created once per request by AgenticRAGService.ask() and passed
    to graph.ainvoke(state, context=runtime_context())

    Usage in nodes:
        async def my_node(state: AgenticState, *, context: Context) -> dict:
            response = await context.ollama_client.generate(
                model=context.model_name,
                prompt=prompt,

        )
    """
    # ─ Service clients (shared across requests, created at startup) ────
    ollama_client: OllamaClient
    opensearch_client: OpenSearchClient
    embeddings_client: JinaEmbeddingsClient

    # ── Observability (optional — graceful degradation) ─────────────────
    # langfuse_tracer: The client wrapper. None if Langfuse is disabled.
    # trace: The active span for this request. Created in ask(), closed
    #        after the grapgh finished. Nodes create child spans under it.
    # langfuse_enabled: Convinience flag to avoid None-checking everywhere.
    langfuse_tracer: Optional[LangfuseTracer] = None
    trace: Optional[Any] = None
    langfuse_enabled: bool = False

    # ── Per-request configuration ───────────────────────────────────────
    # These can differ per request (e.g.. user picks a different model).
    # Defaults match GraphConfig but can we overridden at invocation.
    # Defaults match GraphConfig but can be overridden at invocation.
    model_name: str ="llama3.2:1b"
    temperature: float = 0.7
    top_k: int = 3

    # ── Workflow control ────────────────────────────────────────────────
    # max_retrieval_attempts: Caps the retrieve->grade->rewrite loop.
    # After this many attempts, the generate node runs with whatever.
    # documents were found (even if grading said irrelevant)
    # guardrail_threshold: Minimum score (0-100) from guardrail node
    # to proceed with retrieval. Below this -> out_of_scope node.
    max_retrieval_attempt: int = 3
    guardrail_threshold: int = 40

