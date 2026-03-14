"""Agent orchestrator — fast parallel RAG with web search fallback.

Runs KB retrieval and web search in parallel. Uses the best available sources
to generate a citation-backed answer in a single LLM call. No guardrail or
grading LLM calls — those are deferred to keep latency low.

The LangGraph StateGraph is retained for complex flows (rewrite loops, grading)
but the primary `ask()` path bypasses it for speed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from src.services.agents.context import AgentContext
from src.services.agents.models import SourceItem
from src.services.agents.nodes.generate_answer_node import (
    ainvoke_generate_answer_step,
    build_generation_prompt,
    source_items_to_references,
    _deduplicate_sources,
    _is_web_source,
)
from src.services.agents.nodes.grade_documents_node import ainvoke_grade_documents_step, continue_after_grading
from src.services.agents.nodes.guardrail_node import ainvoke_guardrail_step, continue_after_guardrail
from src.services.agents.nodes.retrieve_node import ainvoke_retrieve_step, convert_search_hits_to_sources
from src.services.agents.nodes.rewrite_query_node import ainvoke_rewrite_query_step
from src.services.agents.nodes.web_search_node import ainvoke_web_search_step
from src.services.agents.state import AgentState, create_initial_state
from src.services.rag.citation import enforce_citations
from src.services.rag.models import RAGResponse, RetrievalMetadata, SourceReference

logger = logging.getLogger(__name__)

OUT_OF_SCOPE_MESSAGE = (
    "I'm a research assistant focused on academic papers. I can't help with that topic, "
    "but I'd be happy to answer questions about scientific research papers in my knowledge base."
)


class AgenticRAGResponse(BaseModel):
    """Response from the agentic RAG workflow."""

    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    reasoning_steps: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def ainvoke_out_of_scope_step(state: AgentState) -> dict[str, Any]:
    """Return a polite rejection for off-topic queries."""
    return {"messages": [AIMessage(content=OUT_OF_SCOPE_MESSAGE)]}


def _extract_context(config: RunnableConfig) -> AgentContext:
    """Extract AgentContext from LangGraph RunnableConfig."""
    return config.get("configurable", {}).get("context")


async def _guardrail_wrapper(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    context = _extract_context(config)
    return await ainvoke_guardrail_step(state, context)


def _guardrail_routing(state: AgentState, config: RunnableConfig) -> str:
    context = _extract_context(config)
    return continue_after_guardrail(state, context)


async def _retrieve_wrapper(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    context = _extract_context(config)
    return await ainvoke_retrieve_step(state, context)


async def _grade_wrapper(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    context = _extract_context(config)
    return await ainvoke_grade_documents_step(state, context)


def _grade_routing(state: AgentState, config: RunnableConfig) -> str:
    context = _extract_context(config)
    return continue_after_grading(state, context)


async def _rewrite_wrapper(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    context = _extract_context(config)
    return await ainvoke_rewrite_query_step(state, context)


async def _web_search_wrapper(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    context = _extract_context(config)
    return await ainvoke_web_search_step(state, context)


async def _generate_wrapper(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    context = _extract_context(config)
    return await ainvoke_generate_answer_step(state, context)


class AgenticRAGService:
    """Fast parallel RAG orchestrator with web search fallback.

    Primary ``ask()`` path runs KB retrieval + web search in parallel,
    then generates a single LLM answer — one LLM call total.
    """

    def __init__(
        self,
        llm_provider: Any,
        retrieval_pipeline: Any | None = None,
        cache_service: Any | None = None,
        web_search_service: Any | None = None,
        arxiv_client: Any | None = None,
        max_retrieval_attempts: int = 1,
        guardrail_threshold: int = 40,
        default_model: str = "",
        default_top_k: int = 5,
    ) -> None:
        self._llm_provider = llm_provider
        self._retrieval_pipeline = retrieval_pipeline
        self._cache_service = cache_service
        self._web_search_service = web_search_service
        self._arxiv_client = arxiv_client
        self._max_retrieval_attempts = max_retrieval_attempts
        self._guardrail_threshold = guardrail_threshold
        self._default_model = default_model
        self._default_top_k = default_top_k

        self._compiled_graph = self._build_graph()

    def _build_graph(self):
        """Construct and compile the LangGraph StateGraph (used for complex flows)."""
        graph = StateGraph(AgentState)

        graph.add_node("guardrail", _guardrail_wrapper)
        graph.add_node("out_of_scope", ainvoke_out_of_scope_step)
        graph.add_node("retrieve", _retrieve_wrapper)
        graph.add_node("grade_documents", _grade_wrapper)
        graph.add_node("rewrite_query", _rewrite_wrapper)
        graph.add_node("web_search", _web_search_wrapper)
        graph.add_node("generate_answer", _generate_wrapper)

        graph.add_edge(START, "guardrail")
        graph.add_conditional_edges(
            "guardrail",
            _guardrail_routing,
            {"continue": "retrieve", "out_of_scope": "out_of_scope"},
        )
        graph.add_edge("out_of_scope", END)
        graph.add_edge("retrieve", "grade_documents")
        graph.add_conditional_edges(
            "grade_documents",
            _grade_routing,
            {"generate": "generate_answer", "rewrite": "rewrite_query", "web_search": "web_search"},
        )
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("web_search", "generate_answer")
        graph.add_edge("generate_answer", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Fast path: parallel retrieval + single LLM call
    # ------------------------------------------------------------------

    async def _retrieve_kb(self, query: str, top_k: int) -> list[SourceItem]:
        """Retrieve from KB. Returns empty list on failure (no exception)."""
        if self._retrieval_pipeline is None:
            return []
        try:
            result = await self._retrieval_pipeline.retrieve(query, top_k=top_k)
            return convert_search_hits_to_sources(result.results)
        except Exception as e:
            logger.warning("KB retrieval failed: %s", e)
            return []

    async def _search_web(self, query: str) -> list[SourceItem]:
        """Search web + fetch content. Returns empty list on failure."""
        if self._web_search_service is None:
            return []
        try:
            from src.services.agents.nodes.web_search_node import ainvoke_web_search_step

            # Build a minimal state/context for the web search node
            state: AgentState = {"original_query": query}  # type: ignore[typeddict-item]
            context = AgentContext(
                llm_provider=self._llm_provider,
                web_search_service=self._web_search_service,
            )
            result = await ainvoke_web_search_step(state, context)
            return result.get("relevant_sources", [])
        except Exception as e:
            logger.warning("Web search failed: %s", e)
            return []

    async def ask(
        self,
        query: str,
        user_id: str = "api_user",
        model_name: str | None = None,
        top_k: int | None = None,
    ) -> AgenticRAGResponse:
        """Fast parallel RAG: KB + web search in parallel → single LLM call.

        Flow:
          1. KB retrieval + web search run in PARALLEL (60s timeout)
          2. If KB has results → use KB sources (higher trust)
          3. If KB is empty → use web sources (fallback)
          4. Single LLM call to generate answer (90s timeout)
          5. Citation enforcement + source list appended

        One LLM call total. No guardrail. No grading. No rewrite loop.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        start_time = time.monotonic()
        effective_top_k = top_k or self._default_top_k
        steps: list[str] = []

        # Stage 1: Parallel retrieval (KB + web at the same time) with timeout
        kb_task = self._retrieve_kb(query, effective_top_k)
        web_task = self._search_web(query)

        try:
            kb_sources, web_sources = await asyncio.wait_for(
                asyncio.gather(kb_task, web_task, return_exceptions=True),
                timeout=60.0,
            )
            # Handle per-task exceptions from return_exceptions=True
            if isinstance(kb_sources, BaseException):
                logger.warning("KB retrieval failed: %s", kb_sources)
                kb_sources = []
            if isinstance(web_sources, BaseException):
                logger.warning("Web search failed: %s", web_sources)
                web_sources = []
        except asyncio.TimeoutError:
            logger.warning("Retrieval timed out after 60s")
            kb_sources, web_sources = [], []
            steps.append("Retrieval timed out (60s)")

        retrieval_time = time.monotonic() - start_time
        steps.append(f"Retrieval: KB={len(kb_sources)}, Web={len(web_sources)} ({retrieval_time:.1f}s)")

        # Log KB source scores for debugging relevance filtering
        if kb_sources:
            score_summary = ", ".join(f"{s.title[:30]}={s.relevance_score:.3f}" for s in kb_sources[:5])
            logger.info("KB source scores: %s", score_summary)

        # Stage 2: Filter KB sources by minimum relevance score.
        # Without this, hybrid search always returns top_k results even when
        # none are relevant (e.g., asking about BERT when KB has no NLP papers).
        #
        # Score ranges depend on pipeline stages:
        # - After reranking (cross-encoder): 0.0–1.0 (sigmoid), threshold 0.3
        # - RRF fusion scores: typically 0.01–0.05, threshold 0.015
        # - Raw BM25 scores: 1–20+, threshold 3.0
        # We detect the scale from the actual score values.
        pre_filter_count = len(kb_sources)
        if kb_sources:
            max_score = max(s.relevance_score for s in kb_sources)
            if max_score > 1.0:
                # Raw BM25 or unnormalized scores — use relative threshold (30% of top)
                min_threshold = max_score * 0.3
            elif max_score <= 0.1:
                # RRF fusion scores (typically 0.01–0.05)
                min_threshold = max_score * 0.3
            else:
                # Reranker (sigmoid) normalized 0–1 scores — 0.3 is a good cutoff
                min_threshold = 0.3
            kb_sources = [s for s in kb_sources if s.relevance_score >= min_threshold]
            if len(kb_sources) < pre_filter_count:
                steps.append(f"Filtered KB: {pre_filter_count} → {len(kb_sources)} (threshold={min_threshold:.3f})")

        # Stage 3: Merge sources — KB (higher trust) + web (supplementary)
        # Always include web results so the LLM has broader context to work with,
        # rather than discarding them when KB returns low-relevance hits.
        sources = kb_sources + web_sources
        if sources:
            steps.append(f"Using {len(kb_sources)} KB + {len(web_sources)} web sources ({len(sources)} total)")
        else:
            elapsed = time.monotonic() - start_time
            steps.append("No relevant sources from KB or web after filtering")
            return AgenticRAGResponse(
                answer="I couldn't find any relevant information for your question from the knowledge base or web search. "
                       "Try rephrasing your question or searching for a different topic.",
                reasoning_steps=steps,
                metadata={"elapsed_seconds": round(elapsed, 3), "user_id": user_id},
            )

        # Stage 4: Single LLM call — generate answer
        deduped = _deduplicate_sources(sources)
        prompt = build_generation_prompt(query, deduped)

        gen_start = time.monotonic()
        try:
            llm = self._llm_provider.get_langchain_model(
                model=model_name or self._default_model,
                temperature=0.7,
            )
            response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=90.0)
            raw_answer = response.content if hasattr(response, "content") else str(response)
        except asyncio.TimeoutError:
            logger.error("LLM generation timed out after 90s")
            elapsed = time.monotonic() - start_time
            return AgenticRAGResponse(
                answer="The answer generation timed out. Please try a simpler question or try again later.",
                reasoning_steps=steps + ["LLM generation timed out (90s)"],
                metadata={"error": "timeout", "elapsed_seconds": round(elapsed, 3)},
            )
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            elapsed = time.monotonic() - start_time
            return AgenticRAGResponse(
                answer="I was unable to generate an answer due to a processing error. Please try again.",
                reasoning_steps=steps,
                metadata={"error": str(e), "elapsed_seconds": round(elapsed, 3)},
            )

        gen_time = time.monotonic() - gen_start
        steps.append(f"Generated answer ({gen_time:.1f}s)")

        # Stage 5: Detect "no relevant papers" response — don't attach sources
        _NO_PAPERS_PHRASES = (
            "i don't have papers on that topic",
            "i don't have papers on that topic",
            "i could not find any relevant",
            "no relevant papers",
            "no papers found",
        )
        answer_lower = raw_answer.strip().lower()
        is_no_papers = any(phrase in answer_lower for phrase in _NO_PAPERS_PHRASES)

        if is_no_papers:
            elapsed = time.monotonic() - start_time
            steps.append("LLM found no relevant sources — returning without citations")
            steps.append(f"Total: {elapsed:.1f}s")
            return AgenticRAGResponse(
                answer=raw_answer.strip(),
                sources=[],
                reasoning_steps=steps,
                metadata={
                    "elapsed_seconds": round(elapsed, 3),
                    "user_id": user_id,
                    "model_name": model_name or self._default_model,
                    "kb_sources": len(kb_sources),
                    "web_sources": len(web_sources),
                    "no_relevant_sources": True,
                },
            )

        # Stage 6: Citation enforcement — validate inline [N] refs, strip LLM-generated sources
        source_refs = source_items_to_references(deduped)
        rag_response = RAGResponse(answer=raw_answer, sources=source_refs, query=query)
        citation_result = enforce_citations(rag_response)

        # Use the cleaned answer WITHOUT the appended sources markdown.
        # Sources are returned as structured data — the frontend renders them as cards.
        # Including sources in the text AND as data causes duplicate display.
        answer_text = citation_result.formatted_answer
        if citation_result.sources_markdown and answer_text.endswith(citation_result.sources_markdown):
            answer_text = answer_text[: -len(citation_result.sources_markdown)].rstrip()

        elapsed = time.monotonic() - start_time
        steps.append(f"Total: {elapsed:.1f}s")

        return AgenticRAGResponse(
            answer=answer_text,
            sources=source_refs,
            reasoning_steps=steps,
            metadata={
                "elapsed_seconds": round(elapsed, 3),
                "user_id": user_id,
                "model_name": model_name or self._default_model,
                "kb_sources": len(kb_sources),
                "web_sources": len(web_sources),
            },
        )

    @staticmethod
    def _extract_answer(state: dict[str, Any]) -> str:
        """Extract the answer from the last AIMessage in the final state."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return str(msg.content)
        return "Unable to generate an answer. Please try again."

    @staticmethod
    def _extract_sources(state: dict[str, Any]) -> list[SourceReference]:
        """Convert relevant SourceItems to SourceReferences, deduplicated by arxiv_id.

        Multiple chunks from the same paper are merged into a single source entry,
        keeping the highest relevance score and combining chunk texts.
        """
        relevant: list[SourceItem] = state.get("relevant_sources", [])

        # Deduplicate by arxiv_id — keep first occurrence, merge chunk texts
        seen: dict[str, int] = {}  # arxiv_id -> index in deduped list
        deduped: list[SourceReference] = []

        for s in relevant:
            key = s.arxiv_id
            if key in seen:
                # Merge: combine chunk text, keep higher score
                existing = deduped[seen[key]]
                existing.chunk_text = (existing.chunk_text or "") + "\n\n" + (s.chunk_text or "")
                if s.relevance_score and (existing.score is None or s.relevance_score > existing.score):
                    existing.score = s.relevance_score
            else:
                seen[key] = len(deduped)
                deduped.append(
                    SourceReference(
                        index=0,  # will be re-indexed below
                        arxiv_id=s.arxiv_id,
                        title=s.title,
                        authors=s.authors,
                        arxiv_url=s.url,
                        chunk_text=s.chunk_text,
                        score=s.relevance_score,
                    )
                )

        # Re-index 1-based
        for i, ref in enumerate(deduped):
            ref.index = i + 1

        return deduped

    def _extract_reasoning_steps(self, state: dict[str, Any]) -> list[str]:
        """Build a human-readable list of reasoning steps from final state."""
        steps: list[str] = []

        # Guardrail
        guardrail = state.get("guardrail_result")
        if guardrail:
            verdict = "passed" if guardrail.score >= self._guardrail_threshold else "rejected"
            steps.append(f"Guardrail: {verdict} (score={guardrail.score}, reason={guardrail.reason})")

        # Retrieval
        attempts = state.get("retrieval_attempts", 0)
        sources = state.get("sources", [])
        if attempts > 0:
            steps.append(f"Retrieved {len(sources)} documents (attempt {attempts})")

        # Grading
        grading_results = state.get("grading_results", [])
        relevant = state.get("relevant_sources", [])
        if grading_results:
            steps.append(f"Grading: {len(relevant)}/{len(grading_results)} relevant")

        # Rewrite (if happened)
        if state.get("rewritten_query"):
            steps.append(f"Rewrote query: {state['rewritten_query'][:80]}")

        # Generation
        messages = state.get("messages", [])
        has_ai_message = any(isinstance(m, AIMessage) for m in messages)
        if has_ai_message:
            steps.append(f"Generated answer with {len(relevant)} citations")

        return steps

    # ------------------------------------------------------------------
    # RAGChain-compatible interface (used by FollowUpHandler)
    # ------------------------------------------------------------------

    async def aquery(
        self,
        query: str,
        *,
        top_k: int | None = None,
        categories: list[str] | None = None,
        temperature: float | None = None,
    ) -> RAGResponse:
        """RAGChain-compatible query method. Runs the agentic pipeline and returns RAGResponse."""
        result = await self.ask(query, top_k=top_k)
        return RAGResponse(
            answer=result.answer,
            sources=result.sources,
            query=query,
            retrieval_metadata=RetrievalMetadata(
                stages_executed=result.metadata.get("stages_executed", []),
                total_candidates=result.metadata.get("total_candidates", 0),
                timings=result.metadata.get("timings", {}),
            ),
        )

    async def aquery_stream(
        self,
        query: str,
        *,
        top_k: int | None = None,
        categories: list[str] | None = None,
        temperature: float | None = None,
    ):
        """RAGChain-compatible streaming method.

        Runs the full agentic pipeline (non-streaming), then yields the answer
        followed by a [SOURCES] JSON block. True token streaming requires
        LangGraph streaming support which can be added later.
        """
        import json

        result = await self.ask(query, top_k=top_k)

        # Yield the answer text
        yield result.answer

        # Yield sources as final JSON block (same format as RAGChain)
        sources_json = json.dumps([s.model_dump() for s in result.sources])
        yield f"\n\n[SOURCES]{sources_json}"
