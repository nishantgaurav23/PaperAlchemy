"""
What is needed:
    Pydantic schemas for the agentic RAG /ask-agentic and /feedback API endpoints.
    Separate from ask.py schemas because the agentic response carries richer
    metadata: reasoning steps, retrieval attempts, guardrail score, rewritten query.

Why it is needed:
    FastAPI uses these models for:
    1. Request validation — rejects malformed input before it reaches the service
    2. Response serialisation — converts the service's dict to typed JSON
    3. OpenAPI documentation — auto-generates the Swagger UI schema

How it helps:
    # Router uses AgenticAskRequest for validation:
    @router.post("/ask-agentic", response_model=AgenticAskResponse)
    async def ask_agentic(request: AgenticAskRequest, ...):
        result = await agentic_rag.ask(query=request.query, model=request.model)
        return AgenticAskResponse(**result, model=model_used, search_mode="hybrid")

____________________________________________________________________________________________________________
What it does:
    Defines two schemas for the agentic RAG endpoint:
    1. AgenticAskRequest — validates the incoming POST body
    2. AgenticAskResponse — structures the JSON response

┌─────────────────────┬──────────────────────────────────────────────────────────────────────┐
│        Model        │                            Purpose                                   │
├─────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ AgenticAskRequest   │ Request body: query (required), model (optional override)             │
├─────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ AgenticAskResponse  │ Response: answer + rich metadata (sources as dicts, reasoning steps, │
│                     │ retrieval_attempts, guardrail_score, rewritten_query, timing)         │
└─────────────────────┴──────────────────────────────────────────────────────────────────────┘

Why sources: List[dict] (not List[str]):
    The agentic pipeline returns rich source metadata (arxiv_id, title, authors,
    url, relevance_score) via SourceItem.to_dict(). Unlike the basic /ask endpoint
    which returns only PDF URLs, the agentic response includes enough metadata
    to display rich citations in the UI.

Why model is Optional in the request:
    The agentic service uses graph_config.model as the default (set from
    settings.ollama.default_model at startup). Per-request model override
    is provided for testing different models without restarting the service.

Why top_k is not in AgenticAskRequest:
    Unlike the basic /ask endpoint where top_k controls the retrieval, the
    agentic service bakes top_k into the ToolNode at graph-compile time.
    It cannot be changed per-request without recompiling the graph. The
    graph_config.top_k (set at startup) controls this.

"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

class AgenticAskRequest(BaseModel):
    """Request model for the agentic RAG /ask-agentic endpoint.

    Required:
        query: User question (1-2000 characters)

    Optional:
        model: Override the Ollama model for this request only.
            If None, uses the model configured in GraphConfig (from settings).
    """
    query: str = Field(
        ...,
        description="User question about CS/AI/ML research papers",
        min_length=1,
        max_length=2000,
    )
    model: Optional[str] = Field(
        default=None,
        description=(
            "Ollama model override for this request (e.g., 'llama3.2:1b', 'mistral:7b'). "
            "Uses server default from settings if not specified."
        ),
    )
    class Config:
        json_schema_extra = {
            "example": {
                "query": "How do transformer attention mechanisms work?",
                "model": None,
            }
        }
    
class AgenticAskResponse(BaseModel):
    """Response model for the agentic RAG /ask-agentic endpoint.

    Fields:
        query:              Echo of the original user question
        answer:             LLM-generated answer from retrieved paper context
        sources:            List of source paper dicts (arxiv_id, title, authors, url, score)
        reasoning_steps:    Human-readable agent decision trace (guardrail, retrieval, grading)
        retrieval_attempts: How many retrieve→grade→rewrite loops ran
        rewritten_query:    The rewritten query if the agent rewrote it, else None
        execution_time:     Wall-clock seconds for the full pipeline
        guardrail_score:    Domain relevance score 0-100 from the guardrail node
        search_mode:        Search strategy used ('hybrid' or 'bm25')
        model:              Which Ollama model generated the answer
    """
    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Generated answer from retrieved paper context")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Retrieved paper sources. Each dict contains: "
            "arxiv_id, title, authors (list), url, relevance_score."
        ),
    )
    reasoning_steps: List[str] = Field(
        default_factory=list,
        description="Human-readable trace of agent decisions (guardrail, retrieval, grading)",
    )
    retrieval_attempts: int = Field(
        default=0,
        description="Number of retrieve→grade→rewrite loops that ran",
    )
    rewritten_query: Optional[str] = Field(
        default=None,
        description="Improved query produced by rewrite_query_node, or None if no rewrite",
    )
    execution_time: float = Field(
        ...,
        description="Total pipeline execution time in seconds",
    )
    guardrail_score: Optional[int] = Field(
        default=None,
        description="Domain relevance score 0-100 from the guardrail node",
    )
    search_mode: str = Field(
        default="hybrid",
        description="Search strategy used: 'hybrid' (BM25 + vector) or 'bm25'",
    )
    model: str = Field(..., description="Ollama model that generated the answer")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How do transformer attention mechanisms work?",
                "answer": (
                    "Transformer attention mechanisms allow the model to focus on different "
                    "parts of the input sequence when generating each output token..."
                ),
                "sources": [
                    {
                        "arxiv_id": "1706.03762",
                        "title": "Attention Is All You Need",
                        "authors": ["Vaswani, A.", "Shazeer, N."],
                        "url": "https://arxiv.org/pdf/1706.03762.pdf",
                        "relevance_score": 0.92,
                    }
                ],
                "reasoning_steps": [
                    "Domain validation: scored 95/100 — Clearly about ML/NLP research",
                    "Retrieval: 1 attempt(s) made",
                    "Grading: 3/3 documents rated relevant",
                    "Answer generated from retrieved context",
                ],
                "retrieval_attempts": 1,
                "rewritten_query": None,
                "execution_time": 4.23,
                "guardrail_score": 95,
                "search_mode": "hybrid",
                "model": "llama3.2:1b",
            }
        }
