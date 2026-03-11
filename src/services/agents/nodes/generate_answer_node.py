"""Answer generation node for citation-backed responses (S6.6).

Takes relevant sources from the agent state, constructs a citation-enforcing
prompt, invokes the LLM, and post-processes the output with S5.5 citation
enforcement to ensure inline [N] references and a formatted source list.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from src.services.agents.context import AgentContext
from src.services.agents.models import SourceItem
from src.services.agents.state import AgentState
from src.services.rag.citation import enforce_citations
from src.services.rag.models import RAGResponse, SourceReference

logger = logging.getLogger(__name__)

NO_SOURCES_MESSAGE = (
    "I don't have papers on that topic in my knowledge base. "
    "Please try rephrasing your question or searching for a different topic."
)

GENERATION_PROMPT = """You are a research assistant that answers questions using ONLY the provided academic paper excerpts.
You MUST cite sources inline using [N] notation (e.g., [1], [2]) corresponding to the numbered sources below.
Every factual claim MUST have at least one citation. Do NOT fabricate information or cite sources not provided.

If the sources do not contain enough information to answer the question, say so honestly.

## Sources

{sources_block}

## Question

{query}

## Instructions

1. Answer the question using ONLY information from the sources above.
2. Use inline citations [1], [2], etc. for every claim.
3. Be concise but thorough.
4. Do NOT add a "Sources" section at the end — it will be appended automatically."""


def source_items_to_references(sources: list[SourceItem]) -> list[SourceReference]:
    """Convert agent SourceItems to RAG SourceReferences with 1-based indices."""
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
        for i, s in enumerate(sources)
    ]


def build_generation_prompt(query: str, sources: list[SourceItem]) -> str:
    """Build the generation prompt with numbered source chunks.

    Args:
        query: The user's research question.
        sources: Relevant source documents to include in the prompt.

    Returns:
        Formatted prompt string with numbered sources and citation instructions.
    """
    source_lines = []
    for i, source in enumerate(sources, 1):
        source_lines.append(f"[{i}] {source.title} (arxiv: {source.arxiv_id})\n{source.chunk_text}")
    sources_block = "\n\n".join(source_lines)

    return GENERATION_PROMPT.format(sources_block=sources_block, query=query)


def _get_effective_query(state: AgentState) -> str:
    """Get the best available query: rewritten_query > original_query > last HumanMessage."""
    if state.get("rewritten_query"):
        return state["rewritten_query"]
    if state.get("original_query"):
        return state["original_query"]
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


async def ainvoke_generate_answer_step(
    state: AgentState,
    context: AgentContext,
) -> dict[str, Any]:
    """Generate a citation-backed answer from relevant sources.

    Workflow:
    1. If no relevant sources → return fallback "no papers" message.
    2. Build citation-enforcing prompt with numbered source chunks.
    3. Invoke LLM to generate answer.
    4. Post-process with enforce_citations() from S5.5.
    5. Return AIMessage with formatted answer + citation metadata.

    Args:
        state: Current agent state (reads relevant_sources, messages, queries).
        context: Runtime context (reads llm_provider, model_name, temperature).

    Returns:
        Partial state dict with messages (AIMessage) and metadata (citation_validation).
    """
    logger.info("NODE: generate_answer")

    relevant_sources: list[SourceItem] = state.get("relevant_sources", [])

    # FR-5: No-sources fallback
    if not relevant_sources:
        logger.info("No relevant sources — returning fallback message")
        return {
            "messages": [AIMessage(content=NO_SOURCES_MESSAGE)],
            "metadata": {"citation_validation": {"is_valid": True, "reason": "no_sources"}},
        }

    query = _get_effective_query(state)
    logger.debug("Generating answer for query: %s", query[:100])

    # FR-1: Build prompt
    prompt = build_generation_prompt(query, relevant_sources)

    # FR-2: Invoke LLM
    try:
        llm = context.llm_provider.get_langchain_model(
            model=context.model_name,
            temperature=context.temperature,
        )
        response = await llm.ainvoke(prompt)
        raw_answer = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        return {
            "messages": [AIMessage(content="I was unable to generate an answer due to a processing error. Please try again.")],
            "metadata": {"citation_validation": {"is_valid": False, "error": str(e)}},
        }

    # FR-3: Citation post-processing
    source_refs = source_items_to_references(relevant_sources)
    rag_response = RAGResponse(answer=raw_answer, sources=source_refs, query=query)
    citation_result = enforce_citations(rag_response)

    # FR-4: Build state update
    logger.info(
        "Generation complete: citations valid=%s, coverage=%.1f%%",
        citation_result.validation.is_valid,
        citation_result.validation.citation_coverage * 100,
    )

    return {
        "messages": [AIMessage(content=citation_result.formatted_answer)],
        "metadata": {
            "citation_validation": {
                "is_valid": citation_result.validation.is_valid,
                "valid_citations": citation_result.validation.valid_citations,
                "invalid_citations": citation_result.validation.invalid_citations,
                "uncited_sources": citation_result.validation.uncited_sources,
                "citation_coverage": citation_result.validation.citation_coverage,
            },
        },
    }
