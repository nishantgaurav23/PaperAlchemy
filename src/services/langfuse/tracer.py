"""
RAG-specific tracer that wraps LangfuseTracer for pipeline stage tracking.

Why it's needed:
    The generic LangfuseTracer handles Langfuse SDK calls, but the RAG
    pipeline has specific stages (embedding, search, prompt, generation)
    that benefits from purpose-built tracing methods with typed metadata.

What it does:
    - trace_request(): Creates a top-level trace for the full /ask request.
    - trace_embedding() / trace_search() / trace_prompt_construction()/
      trace_generation() : Per-stage span creation with relevant metadata.
    - end_*() methods: Attach output and metrics when each stage completes.

How it helps:
    - Clean API for routers: ome method per pipeline stage
    - Consistent metadata structure across all traces
    - Easy to add new stages without touching the base tracer
"""

import logging
from typing import Any, Optional

from src.services.langfuse.client import LangfuseTracer

logger = logging.getLogger(__name__)

class RAGTracer:
    """RAG pipeline tracer wrapping LangFuseTracer with stage-specif methods."""

    def __init__(self, tracer: LangfuseTracer):
        self._tracer = tracer

    @property
    def enabled(self) -> bool:
        return self._tracer.enabled
    
    def trace_request(
            self,
            query: str,
            model: str,
            top_k: int,
            use_hybrid: bool,
            categories: Optional[list[str]] = None,
    ) -> Optional[Any]:
        """Start a top-level trace for a RAG request."""
        return self._tracer.create_trace(
            name="rag-request",
            metadata={
                "query": query,
                "model": model,
                "top_k": top_k,
                "use_hybrid": use_hybrid,
                "categories": categories or [],
            },
            input=query,
        )
    
    def trace_embedding(self, trace: Any, query: str) -> Optional[Any]:
        """Start a span for the query embedding stage."""
        return self._tracer.start_span(
            trace=trace,
            name="embedding",
            metadata={"query_length": len(query)},
            input=query,
        )
    
    def end_embedding(self, span: Any, embedding_dim: Optional[int] = None, fallback: bool = False) -> None:
        """End the embedding span with results."""
        self._tracer.update_span(
            span=span,
            output={"embedding_dim": embedding_dim, "fallback_to_bm25": fallback},
            metadata={"fallback": fallback},
        )

    def trace_search(
            self,
            trace: Any,
            search_mode: str,
            top_k: int
    ) -> Optional[Any]:
        """Start a span for OpenSearch retrieval."""
        return self._tracer.start_span(
            trace=trace,
            name="search",
            metadata={"search_mode": search_mode, "top_k": top_k}
        )
    
    def end_search_search(self, span: Any, chunks_found: int, search_mode: str) -> None:
        """End thesearch span with results."""
        self._tracer.update_span(
            span=span,
            output={"chunks_found": chunks_found, "search_mode": search_mode},
        )

    def trace_prompt_construction(self, trace: Any, chunks_count: int) -> Optional[Any]:
        """Start a span for RAG prompt building."""
        return self._tracer.start_span(
            trace=trace,
            name="prompt-construction",
            metadata={"chunks_count": chunks_count},
        )
    
    def end_prompt_construction(self, span: Any, prompt_length: Optional[int] = None) -> None:
        """End the prompt construction span."""
        self._tracer.update_span(
            span=span,
            output={"prompt_length": prompt_length},
        )

    def trace_generation(self, trace: Any, model: str, prompt: Optional[str] = None) -> Optional[Any]:
        """Start a span for LLM generation."""
        return self._tracer.start_span(
            trace=trace,
            name="llm-generation",
            model=model,
            input=prompt,
        )
    
    def end_genertaion(self, generation: Any, answer: str, usage: Optional[dict] = None) -> None:
        """End the generation span with the LLM response."""
        self._tracer.update_generation(
            generation=generation,
            output=answer,
            usage=usage,
        )

    def flush(self) -> None:
        """Flush pending traces."""
        self._tracer.flush()

    def shutdown(self) -> None:
        """Shut down the tracer."""
        self._tracer.shutdown()
