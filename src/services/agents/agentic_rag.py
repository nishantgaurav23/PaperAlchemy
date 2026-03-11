"""Agent orchestrator — LangGraph StateGraph for agentic RAG (S6.7).

Compiles a LangGraph StateGraph wiring together guardrail, retrieval, grading,
rewrite, and generation nodes into a complete agentic RAG workflow.
The graph is compiled once at startup and reused for every request.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from src.services.agents.context import AgentContext
from src.services.agents.models import SourceItem
from src.services.agents.nodes.generate_answer_node import ainvoke_generate_answer_step
from src.services.agents.nodes.grade_documents_node import ainvoke_grade_documents_step, continue_after_grading
from src.services.agents.nodes.guardrail_node import ainvoke_guardrail_step, continue_after_guardrail
from src.services.agents.nodes.retrieve_node import ainvoke_retrieve_step
from src.services.agents.nodes.rewrite_query_node import ainvoke_rewrite_query_step
from src.services.agents.state import AgentState, create_initial_state
from src.services.rag.models import SourceReference

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


async def _generate_wrapper(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    context = _extract_context(config)
    return await ainvoke_generate_answer_step(state, context)


class AgenticRAGService:
    """LangGraph-based agentic RAG orchestrator.

    Compiles a StateGraph once at init time, exposes ``ask()`` for queries.
    """

    def __init__(
        self,
        llm_provider: Any,
        retrieval_pipeline: Any | None = None,
        cache_service: Any | None = None,
        max_retrieval_attempts: int = 3,
        guardrail_threshold: int = 40,
        default_model: str = "",
        default_top_k: int = 5,
    ) -> None:
        self._llm_provider = llm_provider
        self._retrieval_pipeline = retrieval_pipeline
        self._cache_service = cache_service
        self._max_retrieval_attempts = max_retrieval_attempts
        self._guardrail_threshold = guardrail_threshold
        self._default_model = default_model
        self._default_top_k = default_top_k

        self._compiled_graph = self._build_graph()

    def _build_graph(self):
        """Construct and compile the LangGraph StateGraph."""
        graph = StateGraph(AgentState)

        # Register nodes
        graph.add_node("guardrail", _guardrail_wrapper)
        graph.add_node("out_of_scope", ainvoke_out_of_scope_step)
        graph.add_node("retrieve", _retrieve_wrapper)
        graph.add_node("grade_documents", _grade_wrapper)
        graph.add_node("rewrite_query", _rewrite_wrapper)
        graph.add_node("generate_answer", _generate_wrapper)

        # Edges
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
            {"generate": "generate_answer", "rewrite": "rewrite_query"},
        )
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("generate_answer", END)

        return graph.compile()

    async def ask(
        self,
        query: str,
        user_id: str = "api_user",
        model_name: str | None = None,
        top_k: int | None = None,
    ) -> AgenticRAGResponse:
        """Run a research question through the agentic RAG pipeline.

        Args:
            query: The user's research question. Must be non-empty.
            user_id: User identifier for auditing.
            model_name: Optional LLM model override.
            top_k: Optional override for number of documents to retrieve.

        Returns:
            AgenticRAGResponse with answer, sources, reasoning_steps, metadata.

        Raises:
            ValueError: If query is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        start_time = time.monotonic()

        # Create per-request context
        context = AgentContext(
            llm_provider=self._llm_provider,
            retrieval_pipeline=self._retrieval_pipeline,
            cache_service=self._cache_service,
            model_name=model_name or self._default_model,
            top_k=top_k or self._default_top_k,
            max_retrieval_attempts=self._max_retrieval_attempts,
            guardrail_threshold=self._guardrail_threshold,
            user_id=user_id,
        )

        # Create initial state
        state = create_initial_state(query)

        # Run graph
        try:
            final_state = await self._compiled_graph.ainvoke(
                state,
                config={"configurable": {"context": context}},
            )
        except Exception as e:
            logger.error("Graph execution failed: %s", e, exc_info=True)
            return AgenticRAGResponse(
                answer="I encountered an error while processing your question. Please try again.",
                metadata={"error": str(e), "elapsed_seconds": time.monotonic() - start_time},
            )

        # Extract results
        answer = self._extract_answer(final_state)
        sources = self._extract_sources(final_state)
        reasoning_steps = self._extract_reasoning_steps(final_state)

        elapsed = time.monotonic() - start_time

        return AgenticRAGResponse(
            answer=answer,
            sources=sources,
            reasoning_steps=reasoning_steps,
            metadata={
                "elapsed_seconds": round(elapsed, 3),
                "user_id": user_id,
                "model_name": model_name or self._default_model,
                **final_state.get("metadata", {}),
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
        """Convert relevant SourceItems to SourceReferences."""
        relevant: list[SourceItem] = state.get("relevant_sources", [])
        return [
            SourceReference(
                index=i + 1,
                arxiv_id=s.arxiv_id,
                title=s.title,
                authors=s.authors,
                arxiv_url=s.url,
                chunk_text=s.chunk_text,
                score=s.relevance_score,
            )
            for i, s in enumerate(relevant)
        ]

    @staticmethod
    def _extract_reasoning_steps(state: dict[str, Any]) -> list[str]:
        """Build a human-readable list of reasoning steps from final state."""
        steps: list[str] = []

        # Guardrail
        guardrail = state.get("guardrail_result")
        if guardrail:
            verdict = "passed" if guardrail.score >= 40 else "rejected"
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
