"""
What is needed:
    A FastAPI router that exposes the agentic RAG pipeline as a REST endpoint.
    Receives questions via POST /ask-agentic, calls AgenticRAGService.ask(),
    and returns AgenticAskResponse with answer, sources, and reasoning trace.

Why it is needed:
    FastAPI routers are the translation layer between HTTP and service logic.
    Without this file, AgenticRAGService.ask() has no public API surface.
    The router handles HTTP concerns (status codes, error mapping) so the
    service stays pure Python.

How it helps:
    POST /api/v1/ask-agentic {"query": "How does BERT work?"} →
        → AgenticRAGService.ask(query="How does BERT work?")
        → Full LangGraph pipeline (guardrail → retrieve → grade → generate)
        → AgenticAskResponse(answer=..., sources=[...], reasoning_steps=[...])
        → HTTP 200 JSON

____________________________________________________________________________________________________________
What it does:
    Defines agentic_router with a single POST endpoint for agentic RAG queries.

┌──────────────────────────────┬────────────────────────────────────────────────────────────────┐
│           Endpoint           │                          Purpose                               │
├──────────────────────────────┼────────────────────────────────────────────────────────────────┤
│ POST /ask-agentic            │ Run the full agentic RAG pipeline for a user query             │
└──────────────────────────────┴────────────────────────────────────────────────────────────────┘

Why no streaming endpoint here:
    The agentic pipeline is inherently multi-step (guardrail → retrieve → grade
    → generate). Token-level streaming from generate_answer_node would require
    restructuring the graph to yield tokens incrementally. This is a future
    enhancement — the non-streaming endpoint is correct for the current architecture.

Why prefix="/api/v1" on the router (not on include_router):
    Consistent with how other PaperAlchemy routers declare their prefix.
    The prefix is defined here (next to the routes) so it is immediately
    visible when reading the router file.

Why SettingsDep for model resolution:
    When the user doesn't specify a model, we fall back to
    settings.ollama.default_model rather than hardcoding a model name
    in the router. This respects the 3-tier config hierarchy.

Error mapping:
    ValueError (from service validation) → 422 Unprocessable Entity
    All other exceptions → 500 Internal Server Error with detail

"""

import logging

from fastapi import APIRouter, HTTPException

from src.dependency import AgenticRAGDep, SettingsDep
from src.schemas.api.agentic import AgenticAskRequest, AgenticAskResponse

logger = logging.getLogger(__name__)

agentic_router = APIRouter(prefix="/api/v1", tags=["agentic-rag"])

@agentic_router.post("/ask-agentic", response_model=AgenticAskResponse)
async def ask_agentic(
    request: AgenticAskRequest,
    agentic_rag: AgenticRAGDep,
    settings: SettingsDep,
) -> AgenticAskResponse:
    """Run the agentic RAG pipeline for a research paper question.

    What it does:
        1. Validates the request (Pydantic handles this before the function body)
        2. Resolves the model: request.model or settings.ollama.default_model
        3. Calls AgenticRAGService.ask() with the validated query and model
        4. Maps the result dict to AgenticAskResponse
        5. Returns the structured response

    The agent pipeline:
        guardrail (domain check) → retrieve → tool_retrieve → grade_documents
            ↓ (irrelevant)                                        ↓ (relevant)
        rewrite_query → retrieve (retry)               generate_answer → END

    Args:
        request: Validated request body (query, optional model override)
        agentic_rag: Injected AgenticRAGService from app.state
        settings: Injected Settings for model fallback

    Returns:
        AgenticAskResponse with answer, sources, reasoning_steps, and metadata

    Raises:
        HTTPException 422: If query is empty or invalid
        HTTPException 500: If the agentic pipeline raises an unexpected error
    """
    model_to_use = request.model or settings.ollama.default_model
    search_mode = "hybrid" # AgenticRAGService always uses graph_config.use_hybrid

    logger.info(f"Post /ask-agentic | query={request.query[:80]!r} | model={model_to_use}")

    try:
        result = await agentic_rag.ask(
            query=request.query,
            model=model_to_use,
        )
        
        return AgenticAskResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result.get("sources", []),
            reasoning_steps=result.get("reasoning_steps", []),
            retrieval_attempts=result.get("retrieval_attempts", 0),
            rewritten_query=result.get("rewritten_query"),
            execution_time=result.get("execution_time", 0.0),
            guardrail_score=result.get("guardrail_score"),
            search_mode=search_mode,
            model=model_to_use,
        )
    except ValueError as e:
        logger.warning(f"Invalid request: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Agentic pipeline error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error procesisng question: {str(e)}",
        )