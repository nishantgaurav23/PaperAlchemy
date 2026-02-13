"""
What it does: Defines GraphConfig — a Pydantic settings model that controls how the agentic RAG workflow behaves. Separates 
tunable parameters from code so you can adjust behavior via environment variables without touching node logic.
┌────────────────────────┬───────────────┬───────────────────────────────────────────────────────────┐                      
│         Field          │    Default    │                          Purpose                          │
├────────────────────────┼───────────────┼───────────────────────────────────────────────────────────┤                      
│ model                  │ "llama3.2:1b" │ Default Ollama model (overridable per-request)            │                      
├────────────────────────┼───────────────┼───────────────────────────────────────────────────────────┤
│ temperature            │ 0.7           │ Generation temperature for the answer node                │
├────────────────────────┼───────────────┼───────────────────────────────────────────────────────────┤
│ top_k                  │ 3             │ Number of chunks to retrieve per attempt                  │
├────────────────────────┼───────────────┼───────────────────────────────────────────────────────────┤
│ use_hybrid             │ True          │ Use hybrid search (BM25 + vector) vs BM25 only            │
├────────────────────────┼───────────────┼───────────────────────────────────────────────────────────┤
│ max_retrieval_attempts │ 3             │ Max retrieve→grade→rewrite loops before forced generation │
├────────────────────────┼───────────────┼───────────────────────────────────────────────────────────┤
│ guardrail_threshold    │ 40            │ Minimum guardrail score to proceed (below → out_of_scope) │
└────────────────────────┴───────────────┴───────────────────────────────────────────────────────────┘
Why Pydantic BaseModel over plain dataclass: Unlike Context (which holds live client instances), GraphConfig holds only
primitive values that benefit from Pydantic validation. For example, guardrail_threshold must be 0-100 — Pydantic enforces
this at construction time.

Relationship to Context: GraphConfig is created once when AgenticRAGService is initialized. Its values are then copied into
each per-request Context instance (which can override model_name if the user specifies a different model in the API
  request).

Configuration for the agentic RAG LangGraph workflow.

This config is created once at servie startup (in factory.py) and stored
on AgenticRAGServie. When a request arrives, relevant values are copied
into the per-request Context instance.

Separation from Context:
    GraphConfig = startup-time, validated, shared across requests
    Context = per-request, holds live clients + request-speciifc overrides

Why these defaults:
    - top_k=3: Fewer documents means faster grading (each document requires)
      an LLM call. 3 is enough for most queries; user can override via API.
    - max_retrieval_attempts=3: Prevents infite loops while giving the
      rewrite node 2 chances to improve the query.
    - guardrail_threshold=40: Intentionally permissive. Borderline queries
      (score 40-59) still get retrieval - it's better to retrieve and find
      nothing than to reject a valid research question.
    - temperature=0.7: Only used for the final answer generation, Guardrail
      and grading nodes overrides this to 0.0 for determinsitic routing.
"""

import logging
from typing import Optional

from pydantic import BaseModel, Field

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

class GraphConfig(BaseModel):
    """Configuration for the agentic RAG LangGraph workflow.
    
    Created once at startup. Immutable during request processing.
    Per-request overrides (like model) happen in Context, not here.

    Example:
        config: GraphConfig(top_k=5, use_hybrid=True)
        service= AgenticRAGService(..., graph_config=config)
    """

    # ── Model settings ──────────────────────────────────────────────────
    model: str = Field(
        default="llama3.2:1b",
        description="Default Ollama model for all LLM calls",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Generation temperarure (answer node only; guardrail/grading use 0.0)"
    )
    # ── Retrieval settings ──────────────────────────────────────────────
    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of chunks to retrieve per attempt",
    )
    use_hybrid: bool = Field(
        default=True,
        description="Use hybrid search (BM25 + vector) vs BM25 only."
    )

    # ── Workflow control ────────────────────────────────────────────────
    max_retrieval_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max retrieve-> rewrite cycles before force generation",
    )
    guardrail_threshold: int = Field(
        default=40,
        ge=0,
        le=100,
        description="Minimum guardrail score to procced with retrieval",
    )

    # ── Application settings reference ──────────────────────────────────
    # Stored so nodes can access environment info (e.g., for Langfuse metadata).
    # Not serialized in API responses.
    settings: Optional[Settings] = Field(
        default=None,
        exclude=True,
        description="Application settings refrence (internal use)",
    )
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        """Initialize with application settings if not provided."""
        if "settings" not in data or data["Settings"] is None:
            data["Settings"] = get_settings()

        # Use Ollama default model from settings if not explicitly provided
        if "model" not in data:
            data["model"] = data["Settings"].ollama.default_model

        super().__init__(**data)
        logger.info(
            f"GraphConfig initialized: model={self.model}, top_k={self.top_k}, "
            f"hybrid={self.use_hybrid}, max_attempts={self.max_retrieval_attempts}, "
            f"guardrail_threshold={self.guardrail_threshold}"
        )

