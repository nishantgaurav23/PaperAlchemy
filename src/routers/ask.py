"""                                                                                                         
RAG (Retrieval-Augmented Generation) router for handling question-answering requests.                       
                                                                                                            
Why it's needed:
    This is the core user-facing feature of PaperAlchemy. User ask questions
    about academic papers, and the system retrueves relevant chunks from
    OpeanSearch , then generates an answer using Ollama. Without this router,
    users would have to manually search and read papers themselves.

What it does:
    - POST /ask : Accepts a question, retrieves relevant paper chunks via
    hybrid search, build a RAG prompt, generates an answer with Ollama,
    and returns the answer with source citations.
    - POST /stream: Same as /ask but streams the response token-by-token
    for real-time UI updates (better UX for longer responses).

How it helps:
    - Natural language interface to academic papers
    - Source attribution enables verification of claims
    - Hybrid search finds semantically relevant content
    - Streaming provides responsive user experience
    - Graceful degradation: embedding failures fall back to BM25
    - Redis caching: repeated queries return instantly (150-400x speedup)
    - Langfuse tracing: full pipeline observability per stage

Architecture:
    User Question
            │
            ▼
    AskRequest (validation)
            │
            ▼
    Cache check ──[hit]──► Return cached AskResponse
            │ [miss]
            ▼
    Embed query (Jina) ──[fail]──► BM25 fallback
            │
            ▼
    search_unified (OpenSearch)
            │
            ▼
    Retrieved Chunks
            │
            ▼
    RAGPromptBuilder
            │
            ▼
    OllamaClient.generate_rag_answer()
            │
            ▼
    Store in cache
            │
            ▼
    AskResponse (answer + sources)
"""
import json
import logging
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.dependency import CacheDep, EmbeddingsDep, LangfuseDep, OllamaDep, OpenSearchDep, SettingsDep
from src.exceptions import OllamaConnectionError, OllamaException, OllamaTimeoutError
from src.schemas.api.ask import AskRequest, AskResponse
from src.services.langfuse.tracer import RAGTracer


logger = logging.getLogger(__name__)

# Two routers: one for regular ask, one for streaming
ask_router = APIRouter(tags=["ask"])
stream_router = APIRouter(tags=["stream"])

async def _retrieve_chunks(
        request: AskRequest,
        opensearch_client,
        embeddings_service,
        rag_tracer=None,
        trace=None,
) -> tuple[List[Dict], List[str], str]:

    """
    Retrieve relevant chunks for RAG.

    This helper handles:
    1. Query embedding generation (with graceful degradation)
    2. Unified seacrch execution (hybrid or BM25)
    3. Source URL extraction from results

    Args:
        - request: The incoming AskRequest containing the user's question.
        - opensearch_client: An instance of the OpenSearch client for retrieval.
        - emebddings_service: Jina client for generating query embeddings.
        - rag_tracer: Optional RAGTracer for pipeline observability.
        - trace: Optional Langfuse trace for span attachment.

    Returns:
        Tuple of (chunks, source_urls, search_mode)
    """

    # Generate query embedding for hybrid search
    query_embedding = None
    if request.use_hybrid:
        embedding_span = rag_tracer.trace_embedding(trace, request.query) if rag_tracer else None
        try:
            query_embedding = await embeddings_service.embed_query(request.query)
            logger.info("Generated query embedding for hybrid search")
            if rag_tracer:
                rag_tracer.end_embedding(
                    embedding_span,
                    embedding_dim=len(query_embedding) if query_embedding else None,
                )

        except Exception as e:
            # Graceful degradation: fall back to BM25 if ambeddings fails.
            logger.warning(f"Failed to generate embeddings, falling back to BM25: {e}")
            query_embedding = None
            if rag_tracer:
                rag_tracer.end_embedding(embedding_span, fallback=True)

    # Determine actual search mode
    is_hybrid = request.use_hybrid and query_embedding is not None
    search_mode = "hybrid" if is_hybrid else "bm25"

    # Trace search
    search_span = rag_tracer.trace_search(trace, search_mode, request.top_k) if rag_tracer else None

    # Execute search — use a relevance threshold to filter out off-topic queries.
    # BM25 TF-IDF scores for relevant academic content are typically 2+;
    # irrelevant queries (e.g. "2 + 2") score near 0. Hybrid RRF scores are 0-1.
    min_score = 0.01 if is_hybrid else 1.5

    results = opensearch_client.search_unified(
        query=request.query,
        query_embedding=query_embedding,
        size=request.top_k,
        from_=0,
        categories=request.categories,
        use_hybrid=is_hybrid,
        min_score=min_score,
    )

    # Extract chunks and sources
    chunks = []
    source_set = set()

    for hit in results.get("hits", []):
        arxiv_id = hit.get("arxiv_id")

        # Build minimal chunk for lLM
        chunks.append({
            "arxiv_id": arxiv_id,
            "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
        })

        # Build source URL
        if arxiv_id:
            arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
            source_set.add(f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf")

    if rag_tracer:
        rag_tracer.end_search(search_span, chunks_found=len(chunks), search_mode=search_mode)

    logger.info(f"Retrieved {len(chunks)} chunks for RAG  (mode: {search_mode})")
    return chunks, list(source_set), search_mode

@ask_router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    settings: SettingsDep,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
) -> AskResponse:
    """Answer a question using RAG over indexed papers.

    Flow:
    1. Check cache for exact match (early return on hit)
    2. Retrive relevant chunks via hybrid search
    3. Build RAG prompt with chunks as context
    4. Generate answer with Ollama
    5. Store response in cache
    6. Return answer with soure citations
    """
    model = request.model or settings.ollama.default_model

    # Initialize RAG tracer (graceful — None if Langfuse disabled)
    rag_tracer = RAGTracer(langfuse_tracer) if langfuse_tracer and langfuse_tracer.enabled else None
    trace = rag_tracer.trace_request(
        query=request.query,
        model=model,
        top_k=request.top_k,
        use_hybrid=request.use_hybrid,
        categories=request.categories,
    ) if rag_tracer else None

    try:
        # Step 1: Check cache
        if cache_client:
            cached = await cache_client.find_cached_response(
                query=request.query,
                model=model,
                top_k=request.top_k,
                use_hybrid=request.use_hybrid,
                categories=request.categories,
            )
            if cached is not None:
                logger.info(f"Cache hit for query: {request.query[:50]}...")
                if rag_tracer:
                    rag_tracer.flush()
                return cached

        # Step 2: Check OpenSearch health
        if not opensearch_client.health_check():
            raise HTTPException(
                status_code=503,
                detail="Search servie is currently unavailable."
                )
        # Step 3: Retrieve relevant chunks
        chunks, sources, search_mode = await _retrieve_chunks(
            request, opensearch_client, embeddings_service, rag_tracer, trace
        )

        # Step 4: Handle no results
        if not chunks:
            return AskResponse(
                query=request.query,
                answer=(
                    "I could not find any relevant information in the indexed papers to answer your question. "
                    "Try rephrasing your question or expanding the search criteria by using different keywords or categories."
                ),
                sources=[],
                chunks_used=0,
                search_mode=search_mode,
                model=model,
            )

        # Step 5: Trace prompt construction
        prompt_span = rag_tracer.trace_prompt_construction(trace, len(chunks)) if rag_tracer else None
        if rag_tracer:
            rag_tracer.end_prompt_construction(prompt_span)

        # Step 6: Generate answer with Ollama
        generation_span = rag_tracer.trace_generation(trace, model) if rag_tracer else None

        rag_response = await ollama_client.generate_rag_answer(
            query=request.query,
            chunks=chunks,
            model=model,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        answer = rag_response.get("answer", "Unable to generate answer.")

        if rag_tracer:
            rag_tracer.end_generation(generation_span, answer=answer)

        # Step 7: Build response
        response = AskResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            chunks_used=len(chunks),
            search_mode=search_mode,
            model=model,
        )

        # Step 8: Store in cache
        if cache_client:
            await cache_client.store_response(
                query=request.query,
                model=model,
                top_k=request.top_k,
                use_hybrid=request.use_hybrid,
                response=response,
                categories=request.categories,
            )

        if rag_tracer:
            rag_tracer.flush()

        return response

    except OllamaConnectionError as e:
        logger.error(f"Ollama connection error: {e}")
        raise HTTPException(
            status_code=503,
            detail="LLM service is currently unavailable. Please try again later."
        )
    except OllamaTimeoutError as e:
        logger.error(f"Ollama timeout error: {e}")
        raise HTTPException(
            status_code=504,
            detail="LLM service timed out while generating the answer.Try a shorter question."
        )
    except OllamaException as e:
        logger.error(f"Ollama error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"LLM generation error: {str(e)}"
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions as is
    except Exception as e:
        logger.error(f"Unexpected error in /ask: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}."
        )

@stream_router.post("/stream")
async def ask_question_stream(
    request: AskRequest,
    settings: SettingsDep,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
) -> StreamingResponse:
    """Stream a RAG answer token-by-token.

    Same as /ask but returns a Server-Sent Events stream for real-time
    UI updates. Cache check returns full response as a single SSE event.
    """

    async def generate_stream():
        model = request.model or settings.ollama.default_model

        # Initialize RAG tracer
        rag_tracer = RAGTracer(langfuse_tracer) if langfuse_tracer and langfuse_tracer.enabled else None
        trace = rag_tracer.trace_request(
            query=request.query,
            model=model,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
            categories=request.categories,
        ) if rag_tracer else None

        try:
            # Check cache first
            if cache_client:
                cached = await cache_client.find_cached_response(
                    query=request.query,
                    model=model,
                    top_k=request.top_k,
                    use_hybrid=request.use_hybrid,
                    categories=request.categories,
                )
                if cached is not None:
                    logger.info(f"Cache hit (stream) for query: {request.query[:50]}...")
                    yield f"data: {cached.model_dump_json()}\n\n"
                    if rag_tracer:
                        rag_tracer.flush()
                    return

            # Check OpenSearch health
            if not opensearch_client.health_check():
                yield f"data: {json.dumps({'error': 'Search service is currently unavailable.'})}\n\n"
                return

            # Retrieve chunks
            chunks, sources, search_mode = await _retrieve_chunks(
                request, opensearch_client, embeddings_service, rag_tracer, trace
            )

            # Handle no results
            if not chunks:
                no_results = {
                    "answer": "No relevant information found in indexed papers.",
                    "sources": [],
                    "chunks_used": 0,
                    "search_mode": search_mode,
                    "done": True,
                }
                yield f"data: {json.dumps(no_results)}\n\n"
                return

            # Send metadata first
            metadata = {
                "sources": sources,
                "chunks_used": len(chunks),
                "search_mode": search_mode,
                "model": model,
            }
            yield f"data: {json.dumps(metadata)}\n\n"

            # Trace generation
            generation_span = rag_tracer.trace_generation(trace, model) if rag_tracer else None

            # Stream generation
            full_response = ""
            async for chunk in ollama_client.generate_rag_answer_stream(
                query=request.query,
                chunks=chunks,
                model=model,
                temperature=request.temperature,
                top_p=request.top_p,
            ):

                # Stream text chunks
                if chunk.get("response"):
                    text_chunk = chunk["response"]
                    full_response += text_chunk
                    yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"

                    # Handle completion
                if chunk.get("done", False):
                    yield f"data: {json.dumps({'done': True, 'answer': full_response})}\n\n"
                    break

            # End generation trace
            if rag_tracer:
                rag_tracer.end_generation(generation_span, answer=full_response)

            # Cache the full response
            if cache_client and full_response:
                response = AskResponse(
                    query=request.query,
                    answer=full_response,
                    sources=sources,
                    chunks_used=len(chunks),
                    search_mode=search_mode,
                    model=model,
                )
                await cache_client.store_response(
                    query=request.query,
                    model=model,
                    top_k=request.top_k,
                    use_hybrid=request.use_hybrid,
                    response=response,
                    categories=request.categories,
                )

            if rag_tracer:
                rag_tracer.flush()

        except OllamaConnectionError as e:
            logger.error(f"Ollama connection error in stream: {e}")
            yield f"data: {json.dumps({'error': 'LLM service is currently unavailable.'})}\n\n"
        except OllamaTimeoutError as e:
            logger.error(f"Ollama timeout error in stream: {e}")
            yield f"data: {json.dumps({'error': 'LLM generation timed out.'})}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )