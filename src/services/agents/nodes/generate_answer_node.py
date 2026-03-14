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

GENERATION_PROMPT_KB = """You are a research assistant. Read ALL the provided source excerpts, then synthesize a comprehensive answer that combines insights from multiple sources into a coherent response.

If the provided sources do NOT contain information relevant to the question, respond with:
"I don't have papers on that topic in my knowledge base."
Do NOT answer from general knowledge. Do NOT force an answer from unrelated papers.

## Sources

{sources_block}

## Question

{query}

## Instructions

1. Read and understand ALL sources first.
2. If the sources are NOT relevant, say so. Do NOT force an answer.
3. Write a unified, synthesized answer that draws from multiple sources — do NOT just summarize each source separately.
4. Use inline citations [1], [2], etc. to attribute key claims to their sources.
5. Be thorough but concise. Combine related findings across sources.
6. Do NOT add a "Sources" section at the end — it will be appended automatically.
7. When the answer involves mathematical equations or formulas, use LaTeX notation: inline math with `$...$` and display math with `$$...$$`. For example: $\\mathcal{{L}}_{{GRPO}}$ or $$\\nabla_\\theta J(\\theta) = \\mathbb{{E}}[\\nabla_\\theta \\log \\pi_\\theta(a|s) \\cdot A(s,a)]$$
8. When the answer benefits from a diagram (architecture, flowchart, pipeline), use a mermaid code block.
   IMPORTANT Mermaid syntax rules:
   - In node labels, use double-quotes for labels with special characters like parentheses: `A["Encoder (N layers)"]`
   - In subgraph titles, do NOT use parentheses: use `subgraph Encoder Block` not `subgraph Encoder Block (per layer)`
   - In edge labels with commas, use double-quotes: `A -- "K, V from Encoder" --> B`
   Example:
   ```mermaid
   graph TD
     A["Input Tokens"] --> B["Encoder (N layers)"] --> C["Output"]
   ```"""

GENERATION_PROMPT_WEB = """You are a research assistant. The knowledge base did not have relevant papers, so these sources were retrieved from the web. Read ALL sources, then synthesize a comprehensive answer.

If the sources do NOT contain useful information to answer the question, say so honestly.

## Sources

{sources_block}

## Question

{query}

## Instructions

1. Read and understand ALL sources first.
2. Write a unified, synthesized answer that combines information from multiple sources — do NOT just list what each source says separately.
3. Use inline citations [1], [2], etc. to attribute key claims.
4. Mention that these results come from web search (not the knowledge base).
5. Be thorough but concise.
6. Do NOT add a "Sources" section at the end — it will be appended automatically.
7. When the answer involves mathematical equations or formulas, use LaTeX notation: inline math with `$...$` and display math with `$$...$$`. For example: $\\mathcal{{L}}_{{GRPO}}$ or $$\\nabla_\\theta J(\\theta) = \\mathbb{{E}}[\\nabla_\\theta \\log \\pi_\\theta(a|s) \\cdot A(s,a)]$$
8. When the answer benefits from a diagram (architecture, flowchart, pipeline), use a mermaid code block.
   IMPORTANT Mermaid syntax rules:
   - In node labels, use double-quotes for labels with special characters like parentheses: `A["Encoder (N layers)"]`
   - In subgraph titles, do NOT use parentheses: use `subgraph Encoder Block` not `subgraph Encoder Block (per layer)`
   - In edge labels with commas, use double-quotes: `A -- "K, V from Encoder" --> B`
   Example:
   ```mermaid
   graph TD
     A["Input Tokens"] --> B["Encoder (N layers)"] --> C["Output"]
   ```"""


def _deduplicate_sources(sources: list[SourceItem]) -> list[SourceItem]:
    """Deduplicate source items by arxiv_id, merging chunk texts from the same paper."""
    seen: dict[str, int] = {}  # arxiv_id -> index in deduped list
    deduped: list[SourceItem] = []

    for s in sources:
        key = s.arxiv_id
        if key in seen:
            existing = deduped[seen[key]]
            existing.chunk_text = (existing.chunk_text or "") + "\n\n" + (s.chunk_text or "")
            if s.relevance_score and (existing.relevance_score is None or s.relevance_score > existing.relevance_score):
                existing.relevance_score = s.relevance_score
        else:
            seen[key] = len(deduped)
            # Create a copy to avoid mutating the original state
            deduped.append(SourceItem(
                arxiv_id=s.arxiv_id,
                title=s.title,
                authors=s.authors,
                url=s.url,
                chunk_text=s.chunk_text,
                relevance_score=s.relevance_score,
            ))

    return deduped


def source_items_to_references(sources: list[SourceItem]) -> list[SourceReference]:
    """Convert agent SourceItems to RAG SourceReferences with 1-based indices (deduplicated)."""
    deduped = _deduplicate_sources(sources)
    refs = []
    for i, s in enumerate(deduped):
        is_web = _is_web_source(s)
        refs.append(
            SourceReference(
                index=i + 1,
                arxiv_id="" if is_web else s.arxiv_id,
                title=s.title,
                authors=s.authors,
                arxiv_url="" if is_web else s.url,
                url=s.url,
                chunk_text=s.chunk_text,
                score=s.relevance_score,
                source_type="web" if is_web else "knowledge_base",
            )
        )
    return refs


def _is_web_source(source: SourceItem) -> bool:
    """Check if a source came from web search (not KB)."""
    return source.arxiv_id.startswith("web:")


def _format_source_line(index: int, source: SourceItem) -> str:
    """Format a single source line for the prompt, adapting to source type."""
    if _is_web_source(source):
        return f"[{index}] {source.title} (url: {source.url})\n{source.chunk_text}"
    return f"[{index}] {source.title} (arxiv: {source.arxiv_id})\n{source.chunk_text}"


def build_generation_prompt(query: str, sources: list[SourceItem]) -> str:
    """Build the generation prompt with numbered source chunks.

    Automatically selects the web-aware or KB-only prompt template
    based on whether sources include web results.

    Args:
        query: The user's research question.
        sources: Relevant source documents to include in the prompt.

    Returns:
        Formatted prompt string with numbered sources and citation instructions.
    """
    source_lines = [_format_source_line(i, s) for i, s in enumerate(sources, 1)]
    sources_block = "\n\n".join(source_lines)

    has_web = any(_is_web_source(s) for s in sources)
    template = GENERATION_PROMPT_WEB if has_web else GENERATION_PROMPT_KB

    return template.format(sources_block=sources_block, query=query)


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

    # Deduplicate sources by arxiv_id before prompt construction
    deduped_sources = _deduplicate_sources(relevant_sources)
    logger.info("Deduplicated %d chunks into %d unique papers", len(relevant_sources), len(deduped_sources))

    query = _get_effective_query(state)
    logger.debug("Generating answer for query: %s", query[:100])

    # FR-1: Build prompt
    prompt = build_generation_prompt(query, deduped_sources)

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

    # FR-3: Citation post-processing (use already-deduplicated sources)
    source_refs = source_items_to_references(deduped_sources)
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
