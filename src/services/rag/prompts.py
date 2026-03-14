"""Citation-enforcing prompt templates for the RAG chain (S5.2).

System prompt instructs the LLM to always cite sources using [N] notation.
User prompt builder formats retrieved chunks with numbered metadata.
Supports mixed source types: knowledge base papers, live arXiv results, and web pages.
"""

from __future__ import annotations

from src.schemas.api.search import SearchHit
from src.services.rag.models import SourceReference

SYSTEM_PROMPT = """\
You are a research assistant that answers questions using the provided sources.
Sources come from three places: the knowledge base (indexed papers), live arXiv search, and web search.
You MUST cite sources inline using [N] notation (e.g., [1], [2]) corresponding to the numbered sources below.
Every factual claim MUST have at least one citation. Do NOT fabricate information or cite sources not provided.

IMPORTANT: If the provided sources do NOT contain information that directly answers the question, you MUST respond with:
"I don't have enough information from my sources to answer this question. Please try a different query."
Do NOT attempt to answer from general knowledge. Do NOT force an answer from unrelated sources.

## Instructions

1. First, assess whether the sources actually contain relevant information to answer the question.
2. If the sources are NOT relevant to the question, clearly state that. Do NOT try to force an answer.
3. If the sources ARE relevant, answer using ONLY information from the sources above.
4. Use inline citations [1], [2], etc. for every claim.
5. Structure your answer with clear paragraphs and use **bold** for key terms.
6. Be thorough — provide detailed explanations, not just one-line summaries.
7. When multiple sources discuss the same topic, synthesize their findings and note agreements or differences.
8. Do NOT include a "Sources" or "References" section at the end — it will be appended automatically by the system.
9. When the answer involves mathematical equations or formulas, use LaTeX notation:
   inline math with `$...$` and display math with `$$...$$`.
   For example: $\\mathcal{L}_{CE}$ or $$\\nabla_\\theta J(\\theta) = \\mathbb{E}[\\nabla_\\theta \\log \\pi_\\theta(a|s) \\cdot A(s,a)]$$
10. When the answer benefits from a diagram (architecture, flowchart, pipeline), use a mermaid code block.
    IMPORTANT Mermaid syntax rules:
    - In node labels, use double-quotes for labels with special characters like parentheses: `A["Encoder (N layers)"]`
    - In subgraph titles, do NOT use parentheses: use `subgraph Encoder Block` not `subgraph Encoder Block (per layer)`
    - In edge labels with commas, use double-quotes: `A -- "K, V from Encoder" --> B`
    Example:
    ```mermaid
    graph TD
      A["Input Tokens"] --> B["Encoder (N layers)"] --> C["Output"]
    ```
"""


def build_user_prompt(*, query: str, search_hits: list[SearchHit]) -> str:
    """Build the user prompt with numbered context chunks and the question.

    Each chunk is labeled with its paper title, authors, and arXiv ID
    so the LLM can reference them by number.
    """
    context_parts: list[str] = []

    for i, hit in enumerate(search_hits, start=1):
        title = hit.title or "Unknown Title"
        authors = ", ".join(hit.authors) if hit.authors else "Unknown Authors"
        arxiv_id = hit.arxiv_id or "unknown"
        text = hit.chunk_text or hit.abstract or ""

        context_parts.append(f"[{i}] Title: {title}\n    Authors: {authors}\n    arXiv ID: {arxiv_id}\n    Content: {text}")

    context_block = "\n\n".join(context_parts)

    return (
        f"## Sources\n\n"
        f"{context_block}\n\n"
        f"## Question\n\n"
        f"{query}\n\n"
        f"Answer the question thoroughly using ONLY the sources above. "
        f"Use inline citations [1], [2], etc. for every claim. "
        f"Structure your answer clearly with paragraphs. "
        f"Do NOT add a Sources section — it is generated automatically."
    )


def build_mixed_user_prompt(*, query: str, sources: list[SourceReference]) -> str:
    """Build user prompt from mixed source types (KB + arXiv + web).

    Each source is labeled with its type, title, and content so the
    LLM can reference them by number.
    """
    context_parts: list[str] = []

    for src in sources:
        label_parts = [f"[{src.index}]"]

        if src.source_type == "knowledge_base":
            label_parts.append("[Knowledge Base]")
        elif src.source_type == "arxiv":
            label_parts.append("[arXiv]")
        elif src.source_type == "web":
            label_parts.append("[Web]")

        label_parts.append(f"Title: {src.title}")

        if src.authors:
            label_parts.append(f"Authors: {', '.join(src.authors)}")
        if src.arxiv_id:
            label_parts.append(f"arXiv ID: {src.arxiv_id}")
        if src.url and src.source_type == "web":
            label_parts.append(f"URL: {src.url}")

        content = src.chunk_text or ""
        label_parts.append(f"Content: {content}")

        context_parts.append("\n    ".join(label_parts))

    context_block = "\n\n".join(context_parts)

    return (
        f"## Sources\n\n"
        f"{context_block}\n\n"
        f"## Question\n\n"
        f"{query}\n\n"
        f"Answer the question thoroughly using ONLY the sources above. "
        f"Use inline citations [1], [2], etc. for every claim. "
        f"Structure your answer clearly with paragraphs. "
        f"Do NOT add a Sources section — it is generated automatically."
    )
